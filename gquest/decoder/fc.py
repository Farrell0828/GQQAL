import torch 
from torch import nn 


class FCDecoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_layer = nn.Sequential(
            nn.Linear(config['input_size'], config['hidden_size']),
            nn.ELU()
        )
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config['hidden_size'], config['hidden_size']),
                nn.ELU()
            )
            for n in range(config['n_hidden_layers'] - 1)
        ])
        self.output_layer = nn.Sequential(
            nn.Linear(config['hidden_size'], config['output_size']),
            nn.Sigmoid()
        )

    def forward(self, input_feature):
        output = self.input_layer(input_feature)
        for hidden_layer in self.hidden_layers:
            output = hidden_layer(output)
        output = self.output_layer(output)
        return output