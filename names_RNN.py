import torch as pt
from torch import Tensor
from torch import nn


class RNN(pt.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input: Tensor, hidden: Tensor):
        combined = pt.cat((input, hidden), 1)

        output = self.i2o(combined)
        output = self.softmax(output)

        hidden_output = self.i2h(combined)

        return output, hidden_output

    def init_hidden(self):
        return pt.zeros(1, self.hidden_size)
