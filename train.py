import argparse 
import logging 
import yaml 
import os 
import torch 
import numpy as np 
from tqdm import tqdm 

import torch 
from torch import nn, optim 
from torch.utils.data import DataLoader 
from torch.nn.utils import clip_grad_norm_ 

from transformers import AdamW 
from transformers import get_linear_schedule_with_warmup 
from transformers import get_cosine_schedule_with_warmup  
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup 
from tensorboardX import SummaryWriter 

from gquest.dataset import QuestDataset 
from gquest.model import QuestModel 
from gquest.metrics import spearmans_rho 
from gquest.utils.checkpoint import CheckpointManager 



# For reproducibility.
my_seed=606
np.random.seed(my_seed)
os.environ['PYTHONHASHSEED'] = str(my_seed)
torch.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='configs/config.yml',
        help='Path to config yml file.'
    )
    parser.add_argument(
        "--gpu_ids",
        nargs="+",
        type=int,
        default=0,
        help="List of ids of GPUs to use.",
    )
    parser.add_argument(
        "--cpu_workers",
        type=int,
        default=4,
        help="Number of CPU workers for dataloader.",
    )
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Overfit model on 4 examples, meant for debugging.",
    )
    parser.add_argument(
        "--save_dirpath",
        default="checkpoints/",
        help="Path of directory to create checkpoint directory and save "
        "checkpoints.",
    )
    return parser.parse_args()


def prepare_logger(dirpath):
    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(os.path.join(dirpath, 'log.txt'), mode='w')

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(message)s')
    f_format = logging.Formatter('%(asctime)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


def train(fold, config, args, device, logger):
    # Dataset
    train_dataset = QuestDataset(
        config['dataset'], 'train', fold=fold, overfit=args.overfit
    )
    val_dataset = QuestDataset(
        config['dataset'], 'val', fold=fold, overfit=args.overfit
    )
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=config['solver']['batch_size'], 
                                  shuffle=True,
                                  num_workers=args.cpu_workers)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=config['solver']['batch_size'],
                                shuffle=False,
                                num_workers=args.cpu_workers)

    # Model
    model = QuestModel(config['model']).to(device)
    if -1 not in args.gpu_ids:
        model = nn.DataParallel(model, args.gpu_ids)

    if config['model']['loss'] == 'mse':
        criterion = nn.MSELoss()
    elif config['model']['loss'] == 'bce':
        criterion = nn.BCELoss()
    else:
       raise NotImplementedError
    
    # Weight decay
    if 'no_decay' in config['solver'].keys():
        no_decay = config['solver']['no_decay']
        grouped_parameters = [
            {'params': [
                p for n, p in model.named_parameters() 
                if not any(nd in n for nd in no_decay)
            ], 
            'weight_decay': config['solver']['weight_decay']}, 
            {'params': [
                p for n, p in model.named_parameters() 
                if any(nd in n for nd in no_decay)
            ], 
            'weight_decay': 0.0}
        ]
    else:
        grouped_parameters = model.parameters()

    # Optimizer
    if config['solver']['optimizer'] == 'AdamW':
        optimizer = AdamW(
            grouped_parameters, 
            lr=config["solver"]["initial_lr"], 
            weight_decay=config['solver']['weight_decay']
        )
    else:
        raise NotImplementedError()

    # Learning rate schedule
    iterations = len(train_dataloader)
    if config['solver']['lr_schedule'] == 'warmup_linear':
        warmup_steps = iterations * config["solver"]["warmup_epochs"]
        t_total = iterations * config["solver"]["n_epochs"]
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, t_total
        )
    elif config['solver']['lr_schedule'] == 'warmup_cosine':
        warmup_steps = iterations * config["solver"]["warmup_epochs"]
        t_total = iterations * config["solver"]["n_epochs"]
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, t_total
        )
    elif config['solver']['lr_schedule'] == 'warmup_cosine_with_hard_restarts':
        warmup_steps = iterations * config["solver"]["warmup_epochs"]
        t_total = iterations * config["solver"]["n_epochs"]
        cycles = config['solver']['cycles']
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer, warmup_steps, t_total, num_cycles=cycles
        )
    else:
        raise NotImplementedError()

    # Setup before training
    save_dirpath = os.path.join(args.save_dirpath, 'fold{}'.format(fold))
    summary_writer = SummaryWriter(log_dir=save_dirpath)
    checkpoint_manager = CheckpointManager(
        model, optimizer, save_dirpath, config=config
    )
    global_iteration_step = 0
    epochs_val_pred = []
    epochs_val_true = []

    # Training loop
    model.train()
    for epoch in range(config['solver']['n_epochs']):
        logger.info('\n')
        logger.info('Training for epoch {} begin...'.format(epoch))
        for batch in tqdm(train_dataloader):
            for key in batch:
                batch[key] = batch[key].to(device)

            optimizer.zero_grad()
            batch_output = model(batch)
            batch_loss = criterion(batch_output, batch['targets'])
            batch_loss.backward()
            if config['solver']['max_grad_norm'] > 0:
                clip_grad_norm_(model.parameters(), 
                                config['solver']['max_grad_norm'])
            optimizer.step()
            scheduler.step()

            summary_writer.add_scalar(
                "train/loss", batch_loss, global_iteration_step
            )
            summary_writer.add_scalar(
                "train/lr", optimizer.param_groups[0]["lr"], global_iteration_step
            )
            global_iteration_step += 1
            torch.cuda.empty_cache()

        # On epoch end, save checkpoint and evaluate
        logger.info('Training end. Save checkpoint to {}'.format(save_dirpath))
        checkpoint_manager.step()
        model.eval()
        val_pred = []
        val_true = []
        for batch in tqdm(val_dataloader):
            for key in batch:
                batch[key] = batch[key].to(device)
            with torch.no_grad():
                batch_pred = model(batch)
            val_pred.append(batch_pred)
            val_true.append(batch['targets'])

        val_pred = torch.cat(val_pred, dim=0)
        val_true = torch.cat(val_true, dim=0)
        val_loss = criterion(val_pred, val_true)

        val_pred = val_pred.clone().detach().cpu().numpy()
        val_true = val_true.clone().detach().cpu().numpy()
        val_rho = spearmans_rho(val_true, val_pred)
        epochs_val_pred.append(val_pred)
        epochs_val_true.append(val_true)

        logger.info(
            'Epoch {} evaluate result: val_loss = {}, val_rho = {}'.format(
                epoch, val_loss, val_rho
            )
        )

        summary_writer.add_scalar(
            "val/loss", val_loss, global_iteration_step
        )
        summary_writer.add_scalar(
            "val/rho", val_rho, global_iteration_step
        )

        model.train()
        torch.cuda.empty_cache()

    summary_writer.close()
    epochs_val_pred = np.array(epochs_val_pred)
    epochs_val_true = np.array(epochs_val_true)
    
    return epochs_val_pred, epochs_val_true


if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.save_dirpath, exist_ok=True)
    logger = prepare_logger(args.save_dirpath)
    config = yaml.load(open(args.config))

    logger.info(yaml.dump(config, default_flow_style=False))
    for arg in vars(args):
        logger.info("{:<20}: {}".format(arg, getattr(args, arg)))

    if isinstance(args.gpu_ids, int):
        args.gpu_ids = [args.gpu_ids]
    device = (
        torch.device("cuda", args.gpu_ids[0])
        if args.gpu_ids[0] >= 0
        else torch.device("cpu")
    )

    oof_pred = []
    oof_true = []
    for fold in range(config['dataset']['n_folds']):
        logger.info('\n')
        logger.info('Training for fold {} begin...'.format(fold))
        fold_val_pred, fold_val_true = train(fold, config, args, device, logger)
        oof_pred.append(fold_val_pred)
        oof_true.append(fold_val_true)
    oof_pred = np.concatenate(oof_pred, axis=1)
    oof_true = np.concatenate(oof_true, axis=1)
    logger.info('\n')
    logger.info('OOF rho:')
    for i in range(config['solver']['n_epochs']):
        oof_rho = spearmans_rho(oof_true[i], oof_pred[i])
        logger.info('\tEpoch {}: {}'.format(i, oof_rho))
