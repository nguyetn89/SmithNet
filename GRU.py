import torch
import torch.nn as nn
from torch.autograd import Variable


######################################################################
# https://github.com/happyjin/ConvGRU-pytorch/blob/master/convGRU.py #
# My modification: (1)
######################################################################
class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_channel, hidden_channel, kernel_size, bias, device, preprocess=None, postprocess=None):
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
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super(ConvGRUCell, self).__init__()
        self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_channel = hidden_channel
        self.bias = bias
        self.device = device
        self.preprocess = preprocess
        self.postprocess = postprocess

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

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_channel, self.height, self.width)).to(self.device))

    def forward(self, input_tensor, h_cur):
        """
        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """

        context_raw, reconstruction, instant_pred, long_term_pred = None, None, None, None

        # pre-processing
        if self.preprocess is not None:
            processed_input_tensor, context_raw, reconstruction, instant_pred = self.preprocess(input_tensor)
        else:
            processed_input_tensor = input_tensor
        # print("processed_input_tensor:", processed_input_tensor.shape, "; h_cur:", h_cur.shape)
        combined = torch.cat([processed_input_tensor, h_cur], dim=1)

        combined_conv = self.conv_gates(combined)

        z, r = torch.split(combined_conv, self.hidden_channel, dim=1)
        reset_gate = torch.sigmoid(z)
        update_gate = torch.sigmoid(r)

        combined = torch.cat([processed_input_tensor, reset_gate*h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm

        # post-processing
        if self.postprocess is not None:
            long_term_pred = self.postprocess(h_next)

        return h_next, processed_input_tensor, context_raw, reconstruction, instant_pred, long_term_pred


class ConvGRU(nn.Module):
    def __init__(self, input_size, input_channel, hidden_channel, kernel_size, num_layers,
                 device, batch_first=False, bias=True, return_all_layers=False, preprocess=None, postprocess=None):
        """
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_channel: int e.g. 256
            Number of channels of input tensor.
        :param hidden_channel: int e.g. 1024
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param num_layers: int
            Number of ConvLSTM layers
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        :param alexnet_path: str
            pretrained alexnet parameters
        :param batch_first: bool
            if the first position of array is batch or not
        :param bias: bool
            Whether or not to add the bias.
        :param return_all_layers: bool
            if return hidden and cell states for all layers
        """
        super(ConvGRU, self).__init__()

        # Make sure that both `kernel_size` and `hidden_channel` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_channel = self._extend_for_multilayer(hidden_channel, num_layers)
        if not len(kernel_size) == len(hidden_channel) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size
        self.input_channel = input_channel
        self.hidden_channel = hidden_channel
        self.kernel_size = kernel_size
        self.device = device
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.preprocess = preprocess
        self.postprocess = postprocess

        cell_list = []
        for i in range(self.num_layers):
            cur_input_channel = input_channel if i == 0 else hidden_channel[i-1]
            cell_list.append(ConvGRUCell(input_size=(self.height, self.width),
                                         input_channel=cur_input_channel,
                                         hidden_channel=self.hidden_channel[i],
                                         kernel_size=self.kernel_size[i],
                                         bias=self.bias,
                                         device=self.device,
                                         preprocess=self.preprocess if i == 0 else None,
                                         postprocess=self.postprocess if i == self.num_layers-1 else None))

        # convert python list to pytorch module
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        :param input_tensor: (b, t, c, h, w) or (t,b,c,h,w) depends on if batch first or not
            extracted features from alexnet
        :param hidden_state:
        :return: layer_output_list, last_state_list
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []
        processed_input_list = []
        #
        context_raw_list = []
        reconstruction_list = []
        instant_pred_list = []
        long_term_pred_list = []
        #

        # print("input_tensor:", input_tensor.shape)

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                # print("layer", layer_idx, ": t =", t)
                # input current hidden and cell state then compute the next hidden and cell state through ConvLSTMCell forward function

                h, processed_input, context_raw, reconstruction, instant_pred, long_term_pred = \
                    self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],  # (b,t,c,h,w)
                                              h_cur=h)
                output_inner.append(h)

                if layer_idx == 0:
                    assert processed_input is not None
                    processed_input_list.append(processed_input)
                    assert context_raw is not None
                    context_raw_list.append(context_raw)
                    assert reconstruction is not None
                    reconstruction_list.append(reconstruction)
                    assert instant_pred is not None
                    instant_pred_list.append(instant_pred)

                if layer_idx == self.num_layers-1:
                    assert long_term_pred is not None
                    long_term_pred_list.append(long_term_pred)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append(h)

        processed_input_list = torch.stack(processed_input_list, dim=1)
        context_raw_list = torch.stack(context_raw_list, dim=1)
        reconstruction_list = torch.stack(reconstruction_list, dim=1)
        instant_pred_list = torch.stack(instant_pred_list, dim=1)
        long_term_pred_list = torch.stack(long_term_pred_list, dim=1)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list, processed_input_list, context_raw_list, reconstruction_list, instant_pred_list, long_term_pred_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


if __name__ == '__main__':
    height, width = 16, 24
    channels = 16
    hidden_channel = [32, 64, 32]
    kernel_size = (3, 3)  # kernel size for two stacked hidden layer
    num_layers = 3  # number of stacked hidden layer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ConvGRU(input_size=(height, width),
                    input_channel=channels,
                    hidden_channel=hidden_channel,
                    kernel_size=kernel_size,
                    num_layers=num_layers,
                    device=device,
                    batch_first=True,
                    bias=True,
                    return_all_layers=False)

    batch_size = 2
    time_steps = 3
    input_tensor = torch.rand(batch_size, time_steps, channels, height, width)  # (b,t,c,h,w)
    layer_output_list, last_state_list = model(input_tensor)
    print([x.size() for x in layer_output_list])
    output = layer_output_list[0]
    y0, y1 = torch.split(output, 20, dim=2)
    print(y0.size(), y1.size())
    print([x.size() for x in last_state_list])
