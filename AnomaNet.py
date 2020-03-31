import torch
import torch.nn as nn
# import torch.nn.functional as F

#####################################

# from Semantic import SemanticNet as SemaNet
# from GRU import ConvGRUCell
from utils import get_img_shape


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


class AnomaNet(nn.Module):
    def __init__(self, im_size, device, keep_prob, prt_summary=True):
        super().__init__()
        # set device & input shape
        self.device = device
        self.IM_SIZE = get_img_shape(im_size)

        n_filter = 64
        kernel_size = 5
        padding = kernel_size // 2

        # Inception
        self.Inception = adapted_Inception(3, n_filter)

        # encoding
        in_channel = self.Inception.get_out_channels()
        self.enc_block_1 = nn.Sequential(nn.Conv2d(in_channel, n_filter, kernel_size, stride=1, padding=padding),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.enc_block_2 = nn.Sequential(nn.Conv2d(n_filter, n_filter*2, kernel_size, stride=2, padding=padding),
                                         nn.BatchNorm2d(n_filter*2),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.enc_block_3 = nn.Sequential(nn.Conv2d(n_filter*2, n_filter*4, kernel_size, stride=2, padding=padding),
                                         nn.BatchNorm2d(n_filter*4),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.enc_block_4 = nn.Sequential(nn.Conv2d(n_filter*4, n_filter*8, kernel_size, stride=2, padding=padding),
                                         nn.BatchNorm2d(n_filter*8),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.enc_block_5 = nn.Sequential(nn.Conv2d(n_filter*8, n_filter*8, kernel_size, stride=2, padding=padding),
                                         nn.BatchNorm2d(n_filter*8),
                                         nn.LeakyReLU(negative_slope=0.2))

        # decoding frame
        self.dec_b4_frame = nn.Sequential(nn.ConvTranspose2d(n_filter*8, n_filter*4, kernel_size, stride=2,
                                                             padding=padding, output_padding=1),
                                          nn.BatchNorm2d(n_filter*4),
                                          nn.Dropout2d(p=keep_prob),
                                          nn.ReLU())
        self.dec_b3_frame = nn.Sequential(nn.ConvTranspose2d(n_filter*4, n_filter*4, kernel_size, stride=2,
                                                             padding=padding, output_padding=1),
                                          nn.BatchNorm2d(n_filter*4),
                                          nn.Dropout2d(p=keep_prob),
                                          nn.ReLU())
        self.dec_b2_frame = nn.Sequential(nn.ConvTranspose2d(n_filter*4, n_filter*2, kernel_size, stride=2,
                                                             padding=padding, output_padding=1),
                                          nn.BatchNorm2d(n_filter*2),
                                          nn.Dropout2d(p=keep_prob),
                                          nn.ReLU())
        self.dec_b1_frame = nn.Sequential(nn.ConvTranspose2d(n_filter*2, n_filter, kernel_size, stride=2,
                                                             padding=padding, output_padding=1),
                                          nn.BatchNorm2d(n_filter),
                                          nn.Dropout2d(p=keep_prob),
                                          nn.ReLU())
        self.out_frame = nn.Conv2d(n_filter, 3, kernel_size, stride=1, padding=padding)

        # decoding optical flow
        self.dec_b4_optic = nn.Sequential(nn.ConvTranspose2d(n_filter*8, n_filter*4, kernel_size, stride=2,
                                                             padding=padding, output_padding=1),
                                          nn.BatchNorm2d(n_filter*4),
                                          nn.Dropout2d(p=keep_prob),
                                          nn.ReLU())
        self.dec_b3_optic = nn.Sequential(nn.ConvTranspose2d(n_filter*12, n_filter*4, kernel_size, stride=2,
                                                             padding=padding, output_padding=1),
                                          nn.BatchNorm2d(n_filter*4),
                                          nn.Dropout2d(p=keep_prob),
                                          nn.ReLU())
        self.dec_b2_optic = nn.Sequential(nn.ConvTranspose2d(n_filter*8, n_filter*2, kernel_size, stride=2,
                                                             padding=padding, output_padding=1),
                                          nn.BatchNorm2d(n_filter*2),
                                          nn.Dropout2d(p=keep_prob),
                                          nn.ReLU())
        self.dec_b1_optic = nn.Sequential(nn.ConvTranspose2d(n_filter*4, n_filter, kernel_size, stride=2,
                                                             padding=padding, output_padding=1),
                                          nn.BatchNorm2d(n_filter),
                                          nn.Dropout2d(p=keep_prob),
                                          nn.ReLU())
        self.out_optic = nn.Conv2d(n_filter*2, 3, kernel_size, stride=1, padding=padding)

        self.to(self.device)
        #
        if prt_summary:
            print("Original input size:", self.IM_SIZE)

    # input_tensor: shape (b, c, h, w)
    def forward(self, x):
        features = self.Inception(x)

        # encoding
        enc_1 = self.enc_block_1(features)
        enc_2 = self.enc_block_2(enc_1)
        enc_3 = self.enc_block_3(enc_2)
        enc_4 = self.enc_block_4(enc_3)
        latent = self.enc_block_5(enc_4)

        # decoding frame
        dec_4_fr = self.dec_b4_frame(latent)
        dec_3_fr = self.dec_b3_frame(dec_4_fr)
        dec_2_fr = self.dec_b2_frame(dec_3_fr)
        dec_1_fr = self.dec_b1_frame(dec_2_fr)
        out_frame = self.out_frame(dec_1_fr)

        # decoding optical flow
        dec_b4_fl = self.dec_b4_optic(latent)
        dec_b3_fl = self.dec_b3_optic(torch.cat([dec_b4_fl, enc_4], dim=1))
        dec_b2_fl = self.dec_b2_optic(torch.cat([dec_b3_fl, enc_3], dim=1))
        dec_b1_fl = self.dec_b1_optic(torch.cat([dec_b2_fl, enc_2], dim=1))
        out_optic = self.out_optic(torch.cat([dec_b1_fl, enc_1], dim=1))

        # output
        return out_frame, out_optic
