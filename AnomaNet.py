import torch
import torch.nn as nn
import torch.nn.functional as F

#####################################

from Semantic import SemanticNet as SemaNet
from GRU import ConvGRUCell
from utils import get_img_shape


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


# https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
# Fig. 7, https://arxiv.org/pdf/1512.00567.pdf
class InceptionE(nn.Module):
    def __init__(self, in_channels, conv_block=None):
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, in_channels//2, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, in_channels, kernel_size=1)
        self.branch3x3_2a = conv_block(in_channels, in_channels,
                                       kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(in_channels, in_channels,
                                       kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, in_channels,
                                         kernel_size=1)
        self.branch3x3dbl_2 = conv_block(in_channels, in_channels,
                                         kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(in_channels, in_channels,
                                          kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(in_channels, in_channels,
                                          kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = conv_block(in_channels, in_channels//2,
                                      kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class ExpandedSE(nn.Module):
    def __init__(self, C, H, W, case="both", r=4):
        super().__init__()
        assert case in ("SE_only", "expand_only", "both")
        self.case = case
        self.r = r
        self.C, self.H, self.W = C, H, W
        # SE
        if self.case != "expand_only":
            self.SE = nn.Sequential(nn.ReLU(),
                                    nn.AvgPool2d((H, W)),
                                    nn.Conv2d(C, C//self.r, (1, 1)),
                                    nn.ReLU(),
                                    nn.Conv2d(C//self.r, C, (1, 1)),
                                    nn.Sigmoid())
        else:
            self.SE = None

        # expand
        if self.case != "SE_only":
            self.Expand = nn.Sequential(nn.ReLU(),
                                        nn.Conv2d(C, C//self.r, (1, 1)),
                                        nn.ReLU(),
                                        nn.Conv2d(C//self.r, C, (1, 1)),
                                        nn.Sigmoid())
        else:
            self.Expand = None

    def forward(self, X):
        assert len(X.shape) == 4  # (N, C, H, W)
        if not (X.shape[1] == self.C and X.shape[2] == self.H and X.shape[3] == self.W):
            print(X.shape, self.C, self.H, self.W)
        assert X.shape[1] == self.C and X.shape[2] == self.H and X.shape[3] == self.W
        X_SE = (self.SE(X) * X) if self.SE is not None else 0
        X_Expand = (self.Expand(X) * X) if self.Expand is not None else 0
        return X_SE + X_Expand


class GTNet(nn.Module):
    def __init__(self, im_size, device, prt_summary=False):
        super().__init__()
        # set device & input shape
        self.device = device
        self.IM_SIZE = get_img_shape(im_size)

        # set groundtruth model for context
        # get first 19 layers
        self.context_model = SemaNet(self.IM_SIZE, backbone="VGG19",
                                     reduction=8, n_layer_desire=19, device=self.device)
        self.context_out_shape = self.context_model.get_summary(prt_summary)[-1]
        #
        self.to(self.device)

    def get_output_size(self):
        out_size = self.context_out_shape[2:]
        return out_size

    def forward(self, x):
        context, _ = self.context_model.estimate(x)
        return context


# Block used in AnomaNet
# basic structure: (de)conv-(expandedSE)-conv-batchnorm-relu
# basic params: [in_conv1, out_conv1, out_conv2, slope]
class Block_CeCBR(nn.Module):
    def __init__(self, name, params, use_upconv=False, expandedSE=None, use_batchnorm=True):
        super().__init__()
        assert len(params) == 4
        assert name is not None
        in_conv1, out_conv1, out_conv2, slope = params
        self.use_upconv = use_upconv
        self.use_batchnorm = use_batchnorm
        self.eSE_params = expandedSE
        self.name = name
        self.network = nn.Sequential()
        # first (de)convolution
        if not self.use_upconv:
            self.network.add_module(self.name + "/conv2d_1",
                                    nn.Conv2d(in_conv1, out_conv1, kernel_size=4, stride=2, padding=1))
        else:
            self.network.add_module(self.name + "/upconv2d_1",
                                    nn.ConvTranspose2d(in_conv1, out_conv1, kernel_size=4, stride=2, padding=1))
        self.out_channels = out_conv1
        # expanded SE
        if self.eSE_params is not None:
            assert isinstance(self.eSE_params, dict)
            self.network.add_module(self.name + "/expandSE",
                                    ExpandedSE(out_conv1, self.eSE_params["H"], self.eSE_params["W"], self.eSE_params["case"]))
        # second convolution
        if out_conv2 is not None:
            self.network.add_module(self.name + "/conv2d_2",
                                    nn.Conv2d(out_conv1, out_conv2, kernel_size=3, stride=1, padding=1))
            self.out_channels = out_conv2
        # batch normalization
        if self.use_batchnorm:
            self.network.add_module(self.name + "/batchnorm2d_3",
                                    nn.BatchNorm2d(num_features=out_conv2))
        # activation
        if slope == 0:
            self.network.add_module(self.name + "/relu_4", nn.ReLU())
        elif slope is not None and slope > 0:
            self.network.add_module(self.name + "/lrelu_4", nn.LeakyReLU(negative_slope=slope))

    def forward(self, x):
        return self.network(x)

    def get_out_channels(self):
        return self.out_channels


# use_UNET: concat encoding features to prediction streams
# use_cross_pred: concat instant features to longterm stream
class AnomaNet(nn.Module):
    def __init__(self, im_size, device, use_optical_flow, use_UNET, use_cross_pred, prt_summary=True):
        super().__init__()
        assert use_UNET in ("none", "instant", "longterm", "both")
        self.use_optical_flow = use_optical_flow
        self.use_UNET = use_UNET
        self.use_cross_pred = use_cross_pred
        # set device & input shape
        self.device = device
        self.IM_SIZE = get_img_shape(im_size)

        # ================================= Context =================================
        # 256 raw channels and 64 fine channels
        # F(i) -> C(i)
        C_context = 32  # no. of channels to start
        self.context_raw_0 = nn.Conv2d(3, C_context, kernel_size=3, stride=1, padding=1)

        C_curr = C_context
        self.context_raw_1 = Block_CeCBR("ctx_raw_1", [C_curr, C_curr, 2*C_curr, 0.2])

        C_curr = self.context_raw_1.get_out_channels()
        self.context_raw_2 = Block_CeCBR("ctx_raw_2", [C_curr, C_curr, 2*C_curr, 0.2])

        C_curr = self.context_raw_2.get_out_channels()
        self.context_raw_3 = Block_CeCBR("ctx_raw_3", [C_curr, C_curr, 2*C_curr, 0.2])

        C_curr = self.context_raw_3.get_out_channels()
        self.context_fine = nn.Sequential(nn.Conv2d(C_curr, C_curr//2, kernel_size=1, stride=1, padding=0),
                                          ExpandedSE(C_curr//2, self.IM_SIZE[0]//8, self.IM_SIZE[1]//8, case="both"),
                                          nn.Conv2d(C_curr//2, C_curr//4, kernel_size=1, stride=1, padding=0),
                                          ExpandedSE(C_curr//4, self.IM_SIZE[0]//8, self.IM_SIZE[1]//8, case="both"))
        C_context_fine = C_curr//4

        # ================================= Encoder =================================
        # F(i) -> latent(i)
        C_enc = 16 * 4 // 2  # no. of channels to start
        self.frame_enc_0 = nn.Conv2d(3, C_enc, kernel_size=3, stride=1, padding=1)

        C_curr = C_enc
        self.frame_enc_1 = Block_CeCBR("frame_enc_1", [C_curr, 2*C_curr, 2*C_curr, 0.2])

        C_curr = self.frame_enc_1.get_out_channels()
        self.frame_enc_2 = Block_CeCBR("frame_enc_2", [C_curr, 2*C_curr, 2*C_curr, 0.2],
                                       expandedSE={"H": self.IM_SIZE[0]//4, "W": self.IM_SIZE[1]//4, "case": "both"})

        C_curr = self.frame_enc_2.get_out_channels()
        self.frame_enc_3 = Block_CeCBR("frame_enc_3", [C_curr, 2*C_curr, 2*C_curr, 0.2],
                                       expandedSE={"H": self.IM_SIZE[0]//8, "W": self.IM_SIZE[1]//8, "case": "both"})
        C_latent = self.frame_enc_3.get_out_channels()

        # ================================= Decoder =================================
        # latent(i) -> F^(i)
        C_curr = self.frame_enc_3.get_out_channels()
        self.frame_dec_0 = Block_CeCBR("frame_dec_0", [C_curr, C_curr, C_curr//2, 0], use_upconv=True)

        C_curr = self.frame_dec_0.get_out_channels()
        self.frame_dec_1 = Block_CeCBR("frame_dec_1", [C_curr, C_curr, C_curr//2, 0], use_upconv=True)

        C_curr = self.frame_dec_1.get_out_channels()
        self.frame_dec_2 = Block_CeCBR("frame_dec_2", [C_curr, C_curr, 3, None], use_upconv=True, use_batchnorm=False)
        self.frame_dec_3 = nn.Tanh()

        # Frame description
        # D(i) = Inception[latent(i) & C(i)]
        n_channel_in_Inception = C_latent + C_context_fine
        self.description = InceptionE(n_channel_in_Inception)
        n_channel_out_Inception = 5*n_channel_in_Inception

        # ================================= Instant prediction =================================
        # D(i) -> F'(i+1)
        # (h/8, w/8) -> (h/4, w/4)
        C_curr = n_channel_out_Inception
        self.instant_dec_0 = Block_CeCBR("instant_dec_0", [C_curr, C_curr, C_curr//2, 0], use_upconv=True)

        # (h/4, w/4) -> (h/2, w/2)
        C_curr = self.instant_dec_0.get_out_channels()
        C_addition = self.frame_enc_2.get_out_channels() if self.use_UNET in ("instant", "both") else 0
        self.instant_dec_1 = Block_CeCBR("instant_dec_1", [C_curr + C_addition, C_curr, C_curr//2, 0], use_upconv=True,
                                         expandedSE={"H": self.IM_SIZE[0]//2, "W": self.IM_SIZE[1]//2, "case": "both"})

        # (h/2, w/2) -> (h, w)
        C_curr = self.instant_dec_1.get_out_channels()
        C_addition = self.frame_enc_1.get_out_channels() if self.use_UNET in ("instant", "both") else 0
        self.instant_dec_2 = Block_CeCBR("instant_dec_2", [C_curr + C_addition, C_curr, None, None],
                                         use_upconv=True, use_batchnorm=False,
                                         expandedSE={"H": self.IM_SIZE[0], "W": self.IM_SIZE[1], "case": "both"})
        # instant prediction result
        C_curr = self.instant_dec_2.get_out_channels()
        C_addition = C_enc if self.use_UNET in ("instant", "both") else 0
        if not use_optical_flow:
            self.instant_dec_3 = nn.Sequential(nn.Conv2d(C_curr + C_addition, 3, kernel_size=3, stride=1, padding=1),
                                               nn.Tanh())   # RGB frame
        else:
            self.instant_dec_3 = nn.Conv2d(C_curr + C_addition, 2, kernel_size=3, stride=1, padding=1)  # d_x, d_y

        # ================================= Long-term prediction =================================
        # D(i) -> (RNN)
        hidden_channel = 256
        RNN_input_size = (self.IM_SIZE[0]//8, self.IM_SIZE[1]//8)
        self.RNN = ConvGRUCell(input_size=RNN_input_size,
                               input_channel=n_channel_out_Inception,
                               hidden_channel=hidden_channel,
                               kernel_size=(3, 3),
                               bias=True)
        self.RNN_hidden = torch.autograd.Variable(
            torch.zeros(1, hidden_channel, RNN_input_size[0], RNN_input_size[1])).to(self.device)

        # (RNN) -> F'(i+1)

        # (h/8, w/8) -> (h/4, w/4)
        C_curr = hidden_channel
        self.post_RNN_0 = Block_CeCBR("post_RNN_0", [C_curr, C_curr, C_curr, 0], use_upconv=True)

        # (h/4, w/4) -> (h/2, w/2)
        C_curr = self.post_RNN_0.get_out_channels()
        C_addition = (self.frame_enc_2.get_out_channels() if self.use_UNET in ("longterm", "both") else 0) + \
                     (self.instant_dec_0.get_out_channels() if self.use_cross_pred else 0)
        self.post_RNN_1 = Block_CeCBR("post_RNN_1", [C_curr + C_addition, C_curr, C_curr//2, 0], use_upconv=True)

        # (h/2, w/2) -> (h, w)
        C_curr = self.post_RNN_1.get_out_channels()
        C_addition = (self.frame_enc_1.get_out_channels() if self.use_UNET in ("longterm", "both") else 0) + \
                     (self.instant_dec_1.get_out_channels() if self.use_cross_pred else 0)
        self.post_RNN_2 = Block_CeCBR("post_RNN_2", [C_curr + C_addition, C_curr, C_curr//2, 0], use_upconv=True)

        # longterm prediction result
        C_curr = self.post_RNN_2.get_out_channels()
        C_addition = (C_enc if self.use_UNET in ("longterm", "both") else 0) + \
                     (self.instant_dec_2.get_out_channels() if self.use_cross_pred else 0)
        if not use_optical_flow:
            self.post_RNN_3 = nn.Sequential(nn.Conv2d(C_curr + C_addition, 3, kernel_size=3, stride=1, padding=1),
                                            nn.Tanh())
        else:
            self.post_RNN_3 = nn.Conv2d(C_curr + C_addition, 2, kernel_size=3, stride=1, padding=1)

        self.to(self.device)
        #
        if prt_summary:
            print("Original input size:", self.IM_SIZE)
            print("RNN input size:     ", (self.RNN.height, self.RNN.width))

    # input_tensor: shape (b, c, h, w)
    def forward(self, x):
        # context
        x_context_raw = self.context_raw_3(self.context_raw_2(self.context_raw_1(self.context_raw_0(x))))
        x_context_fine = self.context_fine(x_context_raw)
        # encoding and decoding
        x_enc_0 = self.frame_enc_0(x)
        x_enc_1 = self.frame_enc_1(x_enc_0)
        x_enc_2 = self.frame_enc_2(x_enc_1)
        x_latent = self.frame_enc_3(x_enc_2)
        x_reconst = self.frame_dec_3(self.frame_dec_2(self.frame_dec_1(self.frame_dec_0(x_latent))))
        # description from (context, encoding)
        x_description = self.description(torch.cat((x_latent, x_context_fine), dim=1))
        # instant prediction
        x_instant_0 = self.instant_dec_0(x_description)
        x_instant_1 = self.instant_dec_1(torch.cat((x_instant_0, x_enc_2), dim=1) if self.use_UNET in ("instant", "both") else x_instant_0)
        x_instant_2 = self.instant_dec_2(torch.cat((x_instant_1, x_enc_1), dim=1) if self.use_UNET in ("instant", "both") else x_instant_1)
        x_instant_pred = self.instant_dec_3(torch.cat((x_instant_2, x_enc_0), dim=1) if self. use_UNET in ("instant", "both") else x_instant_2)
        # RNN
        RNN_output = torch.autograd.Variable(torch.zeros(x.size(0), self.RNN.get_attribute("hidden_channel"),
                                             self.RNN.get_attribute("height"), self.RNN.get_attribute("width"))).to(self.device)
        for i in range(x.size(0)):
            self.RNN_hidden = repackage_hidden(self.RNN_hidden)
            self.RNN_hidden = self.RNN(torch.unsqueeze(x_description[i], 0), self.RNN_hidden)
            RNN_output[i] = self.RNN_hidden.data

        x_post_RNN_0 = self.post_RNN_0(RNN_output)

        if self.use_UNET in ("longterm", "both"):
            if self.use_cross_pred:
                x_post_RNN_1 = self.post_RNN_1(torch.cat((x_post_RNN_0, x_enc_2, x_instant_0), dim=1))
                x_post_RNN_2 = self.post_RNN_2(torch.cat((x_post_RNN_1, x_enc_1, x_instant_1), dim=1))
                x_long_term_pred = self.post_RNN_3(torch.cat((x_post_RNN_2, x_enc_0, x_instant_2), dim=1))
            else:
                x_post_RNN_1 = self.post_RNN_1(torch.cat((x_post_RNN_0, x_enc_2), dim=1))
                x_post_RNN_2 = self.post_RNN_2(torch.cat((x_post_RNN_1, x_enc_1), dim=1))
                x_long_term_pred = self.post_RNN_3(torch.cat((x_post_RNN_2, x_enc_0), dim=1))
        else:
            if self.use_cross_pred:
                x_post_RNN_1 = self.post_RNN_1(torch.cat((x_post_RNN_0, x_instant_0), dim=1))
                x_post_RNN_2 = self.post_RNN_2(torch.cat((x_post_RNN_1, x_instant_1), dim=1))
                x_long_term_pred = self.post_RNN_3(torch.cat((x_post_RNN_2, x_instant_2), dim=1))
            else:
                x_post_RNN_1 = self.post_RNN_1(x_post_RNN_0)
                x_post_RNN_2 = self.post_RNN_2(x_post_RNN_1)
                x_long_term_pred = self.post_RNN_3(x_post_RNN_2)
        #
        return x_context_raw, x_reconst, x_instant_pred, x_long_term_pred
