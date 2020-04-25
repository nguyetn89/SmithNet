import torch
import torch.nn as nn
# import torch.nn.functional as F

#####################################

from GRU import ConvGRUCell
from utils import get_img_shape, summary


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# adapted Inception
class adapted_Inception(nn.Module):
    def __init__(self, in_channel, out_channel, max_filter_size=7):
        super(adapted_Inception, self).__init__()
        assert max_filter_size % 2 == 1 and max_filter_size < 8
        n_branch = (max_filter_size + 1) // 2
        assert out_channel % n_branch == 0
        nf_branch = out_channel // n_branch
        self.n_branch = n_branch
        self.nf_branch = nf_branch

        # 1x1 branch
        self.branch_1x1 = nn.Conv2d(in_channel, nf_branch, kernel_size=(1, 1))
        if n_branch == 1:
            return

        # 3x3
        self.branch_3x3 = nn.Sequential(nn.Conv2d(in_channel, nf_branch, kernel_size=(1, 1)),
                                        nn.Conv2d(nf_branch, nf_branch, kernel_size=(1, 3), padding=(0, 1)),
                                        nn.Conv2d(nf_branch, nf_branch, kernel_size=(3, 1), padding=(1, 0)))
        if n_branch == 2:
            return

        # 5x5
        self.branch_5x5 = nn.Sequential(nn.Conv2d(in_channel, nf_branch, kernel_size=(1, 1)),
                                        nn.Conv2d(nf_branch, nf_branch, kernel_size=(1, 3), padding=(0, 1)),
                                        nn.Conv2d(nf_branch, nf_branch, kernel_size=(3, 1), padding=(1, 0)),
                                        nn.Conv2d(nf_branch, nf_branch, kernel_size=(1, 3), padding=(0, 1)),
                                        nn.Conv2d(nf_branch, nf_branch, kernel_size=(3, 1), padding=(1, 0)))
        if n_branch == 3:
            return

        # 7x7
        self.branch_7x7 = nn.Sequential(nn.Conv2d(in_channel, nf_branch, kernel_size=(1, 1)),
                                        nn.Conv2d(nf_branch, nf_branch, kernel_size=(1, 3), padding=(0, 1)),
                                        nn.Conv2d(nf_branch, nf_branch, kernel_size=(3, 1), padding=(1, 0)),
                                        nn.Conv2d(nf_branch, nf_branch, kernel_size=(1, 3), padding=(0, 1)),
                                        nn.Conv2d(nf_branch, nf_branch, kernel_size=(3, 1), padding=(1, 0)),
                                        nn.Conv2d(nf_branch, nf_branch, kernel_size=(1, 3), padding=(0, 1)),
                                        nn.Conv2d(nf_branch, nf_branch, kernel_size=(3, 1), padding=(1, 0)))

    def forward(self, x):
        features_1x1 = self.branch_1x1(x)
        features = [features_1x1]
        if self.n_branch > 1:
            features_3x3 = self.branch_3x3(x)
            features.append(features_3x3)
        if self.n_branch > 2:
            features_5x5 = self.branch_5x5(x)
            features.append(features_5x5)
        if self.n_branch > 3:
            features_7x7 = self.branch_7x7(x)
            features.append(features_7x7)
        return torch.cat(features, dim=1)

    def get_out_channels(self):
        return self.n_branch * self.nf_branch


# conv-batchnorm-lRelu
class EncodingBlock(nn.Module):
    def __init__(self, output_size, conv_params, use_batchnorm, slope,
                 per_element_norm, sigmoid_instead_tanh, per_channel_norm, device):
        super().__init__()
        self.device = device
        self.use_batchnorm = use_batchnorm
        self.use_element_norm = per_element_norm
        self.use_sigmoid_instead_tanh = sigmoid_instead_tanh
        self.use_channel_norm = per_channel_norm

        assert len(conv_params) == 5
        assert isinstance(slope, (int, float))
        in_channel, out_channel, kernel_size, stride, padding = conv_params

        # convolution
        self.conv2d = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)

        # element-normalization
        self.W_soft = None
        if self.use_element_norm:
            self.W_soft = torch.autograd.Variable(torch.zeros(1, out_channel, output_size[0], output_size[1])).to(self.device)
            self.element_norm = nn.Sigmoid() if self.use_sigmoid_instead_tanh else nn.Tanh()

        # channel-normalization
        if self.use_channel_norm:
            self.channel_norm = nn.AvgPool2d(output_size)

        # normalize number of channels for block output
        self.n_stream = 1 + int(self.use_element_norm) + int(self.use_channel_norm)    # 3 streams
        if self.n_stream > 1:
            self.conv2d_concat = nn.Conv2d(self.n_stream * out_channel, out_channel, kernel_size=1, stride=1, padding=0)

        # batch normalization
        if self.use_batchnorm:
            self.batchnorm = nn.BatchNorm2d(out_channel)
        # activation
        self.activation = nn.ReLU() if slope == 0 else nn.LeakyReLU(negative_slope=slope)

    def set_W_soft(self, W_soft):
        if W_soft is not None:
            self.W_soft = W_soft.to(self.device)

    def get_W_soft(self):
        if self.W_soft is None:
            return None
        return self.W_soft.cpu()

    # gamma: annealing factor for element norm
    def forward(self, x, gamma=0):
        # first convolution
        output = self.conv2d(x)

        # stream of element norm
        if self.use_element_norm:
            W = self.element_norm(output)
            self.W_soft = repackage_hidden(self.W_soft)
            self.W_soft *= gamma
            self.W_soft += (1-gamma) * torch.mean(W, 0, keepdim=True)
            element_norm_output = self.W_soft * W

        # stream of channel norm
        if self.use_channel_norm:
            channel_weights = self.channel_norm(output)
            channel_norm_output = channel_weights * output

        # concatenate multiple streams
        if self.use_element_norm:
            output = torch.cat([output, element_norm_output], dim=1)
        if self.use_channel_norm:
            output = torch.cat([output, channel_norm_output], dim=1)

        # last convolution if multiple streams
        if self.n_stream > 1:
            output = self.conv2d_concat(output)

        # batch normalization
        if self.use_batchnorm:
            output = self.batchnorm(output)

        return self.activation(output)


# deconv-batchnorm-dropout-relu
class DecodingBlock(nn.Module):
    def __init__(self, deconv_params, drop_prob):
        super().__init__()
        assert len(deconv_params) == 6
        assert drop_prob >= 0 and drop_prob <= 1
        in_channel, out_channel, kernel_size, stride, padding, output_padding = deconv_params
        self.network = nn.Sequential(nn.ConvTranspose2d(in_channel, out_channel,
                                                        kernel_size, stride,
                                                        padding, output_padding),
                                     nn.BatchNorm2d(out_channel),
                                     nn.Dropout2d(p=drop_prob),
                                     nn.ReLU())

    def forward(self, x):
        return self.network(x)


# extension_params: ["skip:x-x-x", "RNN", "cat_latent", "element_norm", "sigmoid_instead_tanh", "channel_norm"]
class AnomaNet(nn.Module):
    def __init__(self, im_size, device, drop_prob, extension_params, prt_summary=True):
        super().__init__()
        # set device & input shape
        self.device = device
        self.IM_SIZE = get_img_shape(im_size)

        # extensions
        assert isinstance(extension_params, (list, tuple))
        assert extension_params[0][:5] == "skip:"
        self.use_RNN = "RNN" in extension_params
        self.use_RNN_cat_latent = "cat_latent" in extension_params
        self.use_element_norm = "element_norm" in extension_params
        self.use_sigmoid_instead_tanh = "sigmoid_instead_tanh" in extension_params
        self.use_channel_norm = "channel_norm" in extension_params
        self.skip_blocks = [int(block) for block in extension_params[0][5:].split('-')] if "-" in extension_params[0][5:] else []
        #
        n_filter = 64
        kernel_size = 5
        padding = kernel_size // 2

        # Inception
        self.Inception = adapted_Inception(3, n_filter)

        # encoding
        in_channel = self.Inception.get_out_channels()
        self.enc_block_1 = EncodingBlock(self.IM_SIZE,
                                         [in_channel, n_filter, kernel_size, 1, padding],
                                         use_batchnorm=False,
                                         slope=0.2,
                                         per_element_norm=self.use_element_norm if 1 not in self.skip_blocks else False,
                                         sigmoid_instead_tanh=self.use_sigmoid_instead_tanh,
                                         per_channel_norm=self.use_channel_norm if 1 not in self.skip_blocks else False,
                                         device=self.device)
        self.enc_block_2 = EncodingBlock((self.IM_SIZE[0]//2, self.IM_SIZE[1]//2),
                                         [n_filter, n_filter*2, kernel_size, 2, padding],
                                         use_batchnorm=True,
                                         slope=0.2,
                                         per_element_norm=self.use_element_norm if 2 not in self.skip_blocks else False,
                                         sigmoid_instead_tanh=self.use_sigmoid_instead_tanh,
                                         per_channel_norm=self.use_channel_norm if 2 not in self.skip_blocks else False,
                                         device=self.device)
        self.enc_block_3 = EncodingBlock((self.IM_SIZE[0]//4, self.IM_SIZE[1]//4),
                                         [n_filter*2, n_filter*4, kernel_size, 2, padding],
                                         use_batchnorm=True,
                                         slope=0.2,
                                         per_element_norm=self.use_element_norm if 3 not in self.skip_blocks else False,
                                         sigmoid_instead_tanh=self.use_sigmoid_instead_tanh,
                                         per_channel_norm=self.use_channel_norm if 3 not in self.skip_blocks else False,
                                         device=self.device)
        self.enc_block_4 = EncodingBlock((self.IM_SIZE[0]//8, self.IM_SIZE[1]//8),
                                         [n_filter*4, n_filter*8, kernel_size, 2, padding],
                                         use_batchnorm=True,
                                         slope=0.2,
                                         per_element_norm=self.use_element_norm if 4 not in self.skip_blocks else False,
                                         sigmoid_instead_tanh=self.use_sigmoid_instead_tanh,
                                         per_channel_norm=self.use_channel_norm if 4 not in self.skip_blocks else False,
                                         device=self.device)
        self.enc_block_5 = EncodingBlock((self.IM_SIZE[0]//16, self.IM_SIZE[1]//16),
                                         [n_filter*8, n_filter*8, kernel_size, 2, padding],
                                         use_batchnorm=True,
                                         slope=0.2,
                                         per_element_norm=self.use_element_norm if 5 not in self.skip_blocks else False,
                                         sigmoid_instead_tanh=self.use_sigmoid_instead_tanh,
                                         per_channel_norm=self.use_channel_norm if 5 not in self.skip_blocks else False,
                                         device=self.device)

        # decoding frame
        self.dec_b4_frame = DecodingBlock([n_filter*8, n_filter*4, kernel_size, 2, padding, 1], drop_prob)
        self.dec_b3_frame = DecodingBlock([n_filter*4, n_filter*4, kernel_size, 2, padding, 1], drop_prob)
        self.dec_b2_frame = DecodingBlock([n_filter*4, n_filter*2, kernel_size, 2, padding, 1], drop_prob)
        self.dec_b1_frame = DecodingBlock([n_filter*2, n_filter, kernel_size, 2, padding, 1], drop_prob)
        self.out_frame = nn.Conv2d(n_filter, 3, kernel_size, stride=1, padding=padding)

        # RNN for flow estimation (if used)
        if self.use_RNN:
            latent_channel = n_filter*8
            self.latent_channel = latent_channel
            RNN_input_size = (self.IM_SIZE[0]//16, self.IM_SIZE[1]//16)
            self.RNN = ConvGRUCell(input_size=RNN_input_size,
                                   input_channel=latent_channel,
                                   hidden_channel=latent_channel,
                                   kernel_size=(3, 3),
                                   bias=True)
            self.RNN_hidden_tensor = torch.autograd.Variable(
                torch.zeros(1, latent_channel, RNN_input_size[0], RNN_input_size[1])).to(self.device)

            if self.use_RNN_cat_latent:
                self.RNN_postprocess = nn.Sequential(nn.Conv2d(latent_channel*2,
                                                               latent_channel,
                                                               kernel_size=1,
                                                               stride=1,
                                                               padding=0),
                                                     nn.BatchNorm2d(latent_channel),
                                                     nn.LeakyReLU(negative_slope=0.2))

        # decoding optical flow
        self.dec_b4_optic = DecodingBlock([n_filter*8, n_filter*4, kernel_size, 2, padding, 1], drop_prob)
        self.dec_b3_optic = DecodingBlock([n_filter*12, n_filter*4, kernel_size, 2, padding, 1], drop_prob)
        self.dec_b2_optic = DecodingBlock([n_filter*8, n_filter*2, kernel_size, 2, padding, 1], drop_prob)
        self.dec_b1_optic = DecodingBlock([n_filter*4, n_filter, kernel_size, 2, padding, 1], drop_prob)
        self.out_optic = nn.Conv2d(n_filter*2, 3, kernel_size, stride=1, padding=padding)

        self.to(self.device)
        #
        if prt_summary:
            summary(self, (3, self.IM_SIZE[0], self.IM_SIZE[1]), print_details=True)
            print("Model's state_dict:")
            for param_tensor in self.state_dict():
                print(param_tensor, "\t", self.state_dict()[param_tensor].size())

    def reset_hidden_tensor(self):
        if self.use_RNN:
            RNN_input_size = (self.IM_SIZE[0]//16, self.IM_SIZE[1]//16)
            self.RNN_hidden_tensor = torch.autograd.Variable(
                torch.zeros(1, self.latent_channel, RNN_input_size[0], RNN_input_size[1])).to(self.device)

    def set_W_softs(self, W_softs):
        assert len(W_softs) == 5
        print("Assigning W_soft tensors...")
        self.enc_block_1.set_W_soft(W_softs[0])
        self.enc_block_2.set_W_soft(W_softs[1])
        self.enc_block_3.set_W_soft(W_softs[2])
        self.enc_block_4.set_W_soft(W_softs[3])
        self.enc_block_5.set_W_soft(W_softs[4])
        print("W_soft tensors assigned!")

    def get_W_softs(self):
        W_softs = []
        W_softs.append(self.enc_block_1.get_W_soft())
        W_softs.append(self.enc_block_2.get_W_soft())
        W_softs.append(self.enc_block_3.get_W_soft())
        W_softs.append(self.enc_block_4.get_W_soft())
        W_softs.append(self.enc_block_5.get_W_soft())
        return W_softs

    # input_tensor: shape (b, c, h, w)
    def forward(self, x, gamma=0):
        features = self.Inception(x)

        # encoding
        enc_1 = self.enc_block_1(features, gamma)
        enc_2 = self.enc_block_2(enc_1, gamma)
        enc_3 = self.enc_block_3(enc_2, gamma)
        enc_4 = self.enc_block_4(enc_3, gamma)
        latent = self.enc_block_5(enc_4, gamma)

        # decoding frame
        dec_4_fr = self.dec_b4_frame(latent)
        dec_3_fr = self.dec_b3_frame(dec_4_fr)
        dec_2_fr = self.dec_b2_frame(dec_3_fr)
        dec_1_fr = self.dec_b1_frame(dec_2_fr)
        out_frame = self.out_frame(dec_1_fr)

        # decoding optical flow
        if self.use_RNN:
            RNN_output = torch.autograd.Variable(torch.zeros(x.size(0),
                                                 self.RNN.get_attribute("hidden_channel"),
                                                 self.RNN.get_attribute("height"),
                                                 self.RNN.get_attribute("width"))).to(self.device)
            for i in range(x.size(0)):
                self.RNN_hidden_tensor = repackage_hidden(self.RNN_hidden_tensor)
                self.RNN_hidden_tensor = self.RNN(torch.unsqueeze(latent[i], 0), self.RNN_hidden_tensor)
                RNN_output[i] = self.RNN_hidden_tensor.data

            if not self.use_RNN_cat_latent:
                dec_b4_fl = self.dec_b4_optic(RNN_output)
            else:
                dec_b4_fl = self.dec_b4_optic(self.RNN_postprocess(torch.cat([latent, RNN_output], dim=1)))
        else:
            dec_b4_fl = self.dec_b4_optic(latent)
        dec_b3_fl = self.dec_b3_optic(torch.cat([dec_b4_fl, enc_4], dim=1))
        dec_b2_fl = self.dec_b2_optic(torch.cat([dec_b3_fl, enc_3], dim=1))
        dec_b1_fl = self.dec_b1_optic(torch.cat([dec_b2_fl, enc_2], dim=1))
        out_optic = self.out_optic(torch.cat([dec_b1_fl, enc_1], dim=1))

        # output
        return out_frame, out_optic
