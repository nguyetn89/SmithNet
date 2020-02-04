import torch
import torch.nn as nn


######################################################################
# https://github.com/happyjin/ConvGRU-pytorch/blob/master/convGRU.py #
# My modification: (1)
######################################################################
class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_channel, hidden_channel, kernel_size, bias):
        """
        Initialize the ConvLSTM cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_channel: int
            Number of channels of input tensor.
        :param hidden_channel: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        """
        super(ConvGRUCell, self).__init__()
        self.input_channel = input_channel
        self.height, self.width = input_size
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.hidden_channel = hidden_channel
        self.bias = bias

        self.conv_gates = nn.Conv2d(in_channels=input_channel + hidden_channel,
                                    out_channels=2*self.hidden_channel,  # for update_gate,reset_gate respectively
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_can = nn.Conv2d(in_channels=input_channel + hidden_channel,
                                  out_channels=self.hidden_channel,  # for candidate neural memory
                                  kernel_size=kernel_size,
                                  padding=self.padding,
                                  bias=self.bias)

    def get_attribute(self, attribute):
        assert isinstance(attribute, str)
        if attribute == "input_channel":
            return self.input_channel
        elif attribute == "hidden_channel":
            return self.hidden_channel
        elif attribute == "input_size":
            return (self.height, self.width)
        elif attribute == "height":
            return self.height
        elif attribute == "width":
            return self.width

    def forward(self, input_tensor, hidden_tensor):
        """
        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :return: h_next,
            next hidden state
        """

        combined = torch.cat([input_tensor, hidden_tensor], dim=1)
        combined_conv = self.conv_gates(combined)

        z, r = torch.split(combined_conv, self.hidden_channel, dim=1)
        reset_gate = torch.sigmoid(z)
        update_gate = torch.sigmoid(r)

        combined = torch.cat([input_tensor, reset_gate * hidden_tensor], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        out_tensor = (1 - update_gate) * hidden_tensor + update_gate * cnm

        return out_tensor
