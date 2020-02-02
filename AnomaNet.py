import torch
import torch.nn as nn
import torch.nn.functional as F

#####################################

from Semantic import SemanticNet as SemaNet
from GRU import ConvGRU
from utils import get_img_shape


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
                                     reduction=8, n_layer_desire=19)
        self.context_out_shape = self.context_model.get_summary(prt_summary)[-1]
        #
        self.to(self.device)

    def get_output_size(self):
        out_size = self.context_out_shape[2:]
        return out_size

    def forward(self, x):
        context, _ = self.context_model.estimate(x, is_cv_img=False,
                                                 need_normalize=False)
        return context


class AnomaNet(nn.Module):
    def __init__(self, im_size, device, prt_summary=True):
        super().__init__()
        # set device & input shape
        self.device = device
        self.IM_SIZE = get_img_shape(im_size)

        # Context: 256 raw channels and 64 fine channels
        # F(i) -> C(i)
        C_context = 32  # no. of channels to start
        self.context_raw = nn.Sequential(nn.Conv2d(3, C_context, kernel_size=3, stride=1, padding=1),
                                         nn.Conv2d(C_context, C_context, kernel_size=2, stride=2, padding=0),
                                         nn.Conv2d(C_context, 2*C_context, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(num_features=2*C_context),
                                         nn.LeakyReLU(negative_slope=0.2),  # end block 1
                                         nn.Conv2d(2*C_context, 2*C_context, kernel_size=2, stride=2, padding=0),
                                         nn.Conv2d(2*C_context, 4*C_context, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(num_features=4*C_context),
                                         nn.LeakyReLU(negative_slope=0.2),  # end block 2
                                         nn.Conv2d(4*C_context, 4*C_context, kernel_size=2, stride=2, padding=0),
                                         nn.Conv2d(4*C_context, 8*C_context, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(num_features=8*C_context),
                                         nn.LeakyReLU(negative_slope=0.2))  # end block 3

        self.context_1x1conv = nn.Sequential(nn.Conv2d(8*C_context, 4*C_context, kernel_size=1, stride=1, padding=0),
                                             ExpandedSE(4*C_context, self.IM_SIZE[0]//8, self.IM_SIZE[1]//8, case="both"),
                                             nn.Conv2d(4*C_context, 2*C_context, kernel_size=1, stride=1, padding=0),
                                             ExpandedSE(2*C_context, self.IM_SIZE[0]//8, self.IM_SIZE[1]//8, case="both"))

        # Encoder
        # F(i) -> latent(i)
        C_enc = 16 * 4  # no. of channels to start
        self.frame_enc = nn.Sequential(nn.Conv2d(3, C_enc, kernel_size=3, stride=1, padding=1),  # end block 0
                                       nn.Conv2d(C_enc, 2*C_enc, kernel_size=2, stride=2, padding=0),
                                       nn.Conv2d(2*C_enc, 2*C_enc, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(num_features=2*C_enc),
                                       nn.LeakyReLU(negative_slope=0.2),  # end block 1
                                       nn.Conv2d(2*C_enc, 4*C_enc, kernel_size=2, stride=2, padding=0),
                                       ExpandedSE(4*C_enc, self.IM_SIZE[0]//4, self.IM_SIZE[1]//4, case="both"),  # expanded SE
                                       nn.Conv2d(4*C_enc, 4*C_enc, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(num_features=4*C_enc),
                                       nn.LeakyReLU(negative_slope=0.2),  # end block 2
                                       nn.Conv2d(4*C_enc, 8*C_enc, kernel_size=2, stride=2, padding=0),
                                       ExpandedSE(8*C_enc, self.IM_SIZE[0]//8, self.IM_SIZE[1]//8, case="both"),  # expanded SE
                                       nn.Conv2d(8*C_enc, 8*C_enc, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(num_features=8*C_enc),
                                       nn.LeakyReLU(negative_slope=0.2))  # end block 3

        # Decoder
        # latent(i) -> F^(i)
        self.frame_dec = nn.Sequential(nn.ConvTranspose2d(8*C_enc, 8*C_enc, kernel_size=2, stride=2, padding=0),
                                       nn.Conv2d(8*C_enc, 4*C_enc, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(num_features=4*C_enc),
                                       nn.LeakyReLU(negative_slope=0.2),  # end block 1
                                       nn.ConvTranspose2d(4*C_enc, 4*C_enc, kernel_size=2, stride=2, padding=0),
                                       nn.Conv2d(4*C_enc, 2*C_enc, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(num_features=2*C_enc),
                                       nn.LeakyReLU(negative_slope=0.2),  # end block 2
                                       nn.ConvTranspose2d(2*C_enc, 2*C_enc, kernel_size=2, stride=2, padding=0),
                                       nn.Conv2d(2*C_enc, 3, kernel_size=3, stride=1, padding=1),
                                       # nn.BatchNorm2d(num_features=C_enc),
                                       # nn.LeakyReLU(negative_slope=0.2)) #end block 3
                                       nn.Tanh())

        # Frame description
        # D(i) = Inception[latent(i) & C(i)]
        n_channel_in_Inception = 8*C_enc + 2*C_context
        self.description = InceptionE(n_channel_in_Inception)
        n_channel_out_Inception = 5*n_channel_in_Inception

        # Instant prediction
        # D(i) -> F'(i+1)
        self.instant_dec_1 = nn.Sequential(nn.ConvTranspose2d(n_channel_out_Inception, 8*C_enc, kernel_size=2, stride=2, padding=0),
                                           nn.Conv2d(8*C_enc, 4*C_enc, kernel_size=3, stride=1, padding=1),
                                           nn.BatchNorm2d(num_features=4*C_enc),
                                           nn.LeakyReLU(negative_slope=0.2),  # end block 1
                                           nn.ConvTranspose2d(4*C_enc, 4*C_enc, kernel_size=2, stride=2, padding=0),
                                           ExpandedSE(4*C_enc, self.IM_SIZE[0]//2, self.IM_SIZE[1]//2, case="both"))  # expanded SE
        self.instant_dec_2 = nn.Sequential(nn.Conv2d(4*C_enc, 2*C_enc, kernel_size=3, stride=1, padding=1),
                                           nn.BatchNorm2d(num_features=2*C_enc),
                                           nn.LeakyReLU(negative_slope=0.2),  # end block 2
                                           nn.ConvTranspose2d(2*C_enc, 2*C_enc, kernel_size=2, stride=2, padding=0),
                                           ExpandedSE(2*C_enc, self.IM_SIZE[0], self.IM_SIZE[1], case="both"))  # expanded SE
        self.instant_dec_3 = nn.Sequential(nn.Conv2d(2*C_enc, 3, kernel_size=3, stride=1, padding=1),
                                           # nn.BatchNorm2d(num_features=C_enc),
                                           # nn.LeakyReLU(negative_slope=0.2)) #end block 3
                                           nn.Tanh())

        # Long-term prediction
        # D(i) -> (RNN)
        RNN_input_size = (self.IM_SIZE[0]//8, self.IM_SIZE[1]//8)
        RNN_in_channel = n_channel_out_Inception
        RNN_hidden_channel = [256, 256, 256]
        RNN_kernel_size = (3, 3)
        RNN_num_layer = len(RNN_hidden_channel)
        self.RNN = ConvGRU(input_size=RNN_input_size,
                           input_channel=RNN_in_channel,
                           hidden_channel=RNN_hidden_channel,
                           kernel_size=RNN_kernel_size,
                           num_layers=RNN_num_layer,
                           device=self.device,
                           batch_first=True,
                           bias=True,
                           return_all_layers=False,
                           preprocess=self.frame_to_RNN,
                           postprocess=self.RNN_to_frame)

        # (RNN) -> F'(i+1)
        C_out = RNN_hidden_channel[-1]
        self.RNN_out = nn.Sequential(nn.ConvTranspose2d(C_out, C_out, kernel_size=2, stride=2, padding=0),
                                     nn.Conv2d(C_out, C_out//2, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(num_features=C_out//2),
                                     nn.LeakyReLU(negative_slope=0.2),  # end block 1
                                     nn.ConvTranspose2d(C_out//2, C_out//2, kernel_size=2, stride=2, padding=0),
                                     nn.Conv2d(C_out//2, C_out//4, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(num_features=C_out//4),
                                     nn.LeakyReLU(negative_slope=0.2),  # end block 2
                                     nn.ConvTranspose2d(C_out//4, C_out//4, kernel_size=2, stride=2, padding=0),
                                     nn.Conv2d(C_out//4, 3, kernel_size=3, stride=1, padding=1),
                                     # nn.BatchNorm2d(num_features=C_enc),
                                     # nn.LeakyReLU(negative_slope=0.2)) #end block 3
                                     nn.Tanh())
        self.to(self.device)
        #
        if prt_summary:
            print("Original input size:", self.IM_SIZE)
            print("RNN input size:     ", RNN_input_size)

    def frame_to_RNN(self, X):
        # context
        X_context_raw = self.context_raw(X)
        X_context_fine = self.context_1x1conv(X_context_raw)
        # encoding & decoding
        X_latent = self.frame_enc(X)
        X_hat = self.frame_dec(X_latent)
        # description from (context, encoding)
        X_description = self.description(torch.cat((X_latent, X_context_fine), dim=1))
        # instant prediction
        X_expandSE_1 = self.instant_dec_1(X_description)
        X_expandSE_2 = self.instant_dec_2(X_expandSE_1)
        X_instant_pred = self.instant_dec_3(X_expandSE_2)
        return X_description, X_context_raw, X_hat, X_instant_pred

    def RNN_to_frame(self, X):
        return self.RNN_out(X)

    def forward(self, X):
        X_layer_output_RNN, _, X_description, X_context, X_reconstruction, X_instant_pred, X_long_term_pred = self.RNN(X)
        X_out = X_layer_output_RNN[-1]
        return X_out, X_description, X_context, X_reconstruction, X_instant_pred, X_long_term_pred


# config0 = {"instance_pred": 1, "long_term_pred": 1,
#            "instance_concat_before_RNN": 1, "instance_concat_after_RNN": 0, "residual_before_after_RNN": 0}

# class NetCreator(nn.Module):
#     def __init__(self, config):
#         pass
#
#     def forward(self, X):
#         pass

#################################################################
