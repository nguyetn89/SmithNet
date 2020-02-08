import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torch.autograd import Variable
from torchvision import utils
from torch.utils.tensorboard import SummaryWriter

from utils import get_img_shape, image_gradient, tensor_restore, ProgressBar, DatasetDefiner
from AnomaNet import ExpandedSE, GTNet
from AnomaNet import AnomaNet as Generator
from CONFIG import loss_weights

LEN_ZFILL = 5


class Discriminator(nn.Module):
    def __init__(self, im_size, device):
        super().__init__()
        # set device & input shape
        self.device = device
        self.IM_SIZE = get_img_shape(im_size)

        # architecture
        # pair(F[i], F[i+1]) or pair(F[i], F_hat[i+1])
        n_base_channel = 16 * 4  # no. of channels to start
        self.network = nn.Sequential(nn.Conv2d(6, n_base_channel, kernel_size=3, stride=1, padding=1),  # end block 0
                                     nn.Conv2d(n_base_channel, 2*n_base_channel, kernel_size=2, stride=2, padding=0),
                                     nn.Conv2d(2*n_base_channel, 2*n_base_channel, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True),  # end block 1
                                     nn.Conv2d(2*n_base_channel, 4*n_base_channel, kernel_size=2, stride=2, padding=0),
                                     ExpandedSE(4*n_base_channel, self.IM_SIZE[0]//4, self.IM_SIZE[1]//4, case="both"),  # expanded SE
                                     nn.Conv2d(4*n_base_channel, 4*n_base_channel, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(num_features=4*n_base_channel),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True),  # end block 2
                                     nn.Conv2d(4*n_base_channel, 8*n_base_channel, kernel_size=2, stride=2, padding=0),
                                     ExpandedSE(8*n_base_channel, self.IM_SIZE[0]//8, self.IM_SIZE[1]//8, case="both"),  # expanded SE
                                     nn.Conv2d(8*n_base_channel, 8*n_base_channel, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(num_features=8*n_base_channel),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True),  # end block 3
                                     nn.Conv2d(8*n_base_channel, 16*n_base_channel, kernel_size=2, stride=2, padding=0),
                                     ExpandedSE(16*n_base_channel, self.IM_SIZE[0]//16, self.IM_SIZE[1]//16, case="both"),  # expanded SE
                                     nn.Conv2d(16*n_base_channel, 16*n_base_channel, kernel_size=3, stride=1, padding=1),
                                     nn.Conv2d(16*n_base_channel, n_base_channel//2, kernel_size=3, stride=1, padding=1))  # end block 4
        self.output = nn.Sigmoid()
        self.to(self.device)

    # X must have shape (B, 9, H, W)
    # 9 channels: F[i], F_instant[i+1], F_longterm[i+1]
    def forward(self, X):
        frame, pred_instant, pred_longterm = torch.split(X, 3, dim=1)
        prob_instant = self.output(self.network(torch.cat([frame, pred_instant], dim=1)))
        prob_longterm = self.output(self.network(torch.cat([frame, pred_longterm], dim=1)))
        return prob_instant * prob_longterm


# name: dataset's name
class DCGAN(object):
    def __init__(self, name, im_size, store_path, device_str=None):
        self.name = name
        self.im_size = im_size
        # paths
        self.store_path = os.path.join(store_path, self.name)
        self.input_store_path = self.store_path + "/inputs"             # data for training and evaluation
        self.training_data_file = os.path.join(self.input_store_path, self.name + "_training.pt")
        self.evaluation_data_file = os.path.join(self.input_store_path, self.name + "_evaluation.pt")
        self.model_store_path = self.store_path + "/models"             # trained models
        self.gen_image_store_path = self.store_path + "/gen_images"     # generated images (for visual checking)
        self.output_store_path = self.store_path + "/outputs"           # outputs for evaluation
        self.log_path = self.store_path + "/log"                        # tensorboard log
        self._create_all_paths()
        # device
        if device_str is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                       else "cpu")
        else:
            assert isinstance(device_str, str)
            self.device = torch.device(device_str)

        print("ContextNet init...")
        self.ContextNet = GTNet(self.im_size, self.device)
        self.ContextNet.eval()

        print("DCGAN init...")
        self.G = Generator(self.im_size, self.device)
        self.D = Discriminator(self.im_size, self.device)
        self.loss = nn.BCELoss()

        # ADAM optimizers
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=0.00002, betas=(0.5, 0.9))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.9))

        # Set the logger
        self.logger = SummaryWriter(self.log_path)
        self.logger.flush()

    # create necessary directories
    def _create_all_paths(self):
        self._create_path(self.input_store_path)
        self._create_path(self.model_store_path)
        self._create_path(self.gen_image_store_path)
        self._create_path(self.output_store_path)
        self._create_path(self.log_path)

    def _create_path(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    # load pretrained models and optimizers
    def _load_model(self, G_model_filename, D_model_filename=None, G_optim_filename=None, D_optim_filename=None):
        self.G.load_state_dict(torch.load(os.path.join(self.model_store_path, G_model_filename)))
        print("Generator loaded from %s" % G_model_filename)
        if D_model_filename is not None:
            self.D.load_state_dict(torch.load(os.path.join(self.model_store_path, D_model_filename)))
            print("Discriminator loaded from %s" % D_model_filename)
        if G_optim_filename is not None:
            self.g_optimizer.load_state_dict(torch.load(os.path.join(self.model_store_path, G_optim_filename)))
            print("G_optimizer loaded from %s" % G_optim_filename)
        if D_optim_filename is not None:
            self.d_optimizer.load_state_dict(torch.load(os.path.join(self.model_store_path, D_optim_filename)))
            print("D_optimizer loaded from %s" % D_optim_filename)

    # save pretrained models and optimizers
    def _save_model(self, G_model_filename, D_model_filename=None, G_optim_filename=None, D_optim_filename=None):
        torch.save(self.G.state_dict(), os.path.join(self.model_store_path, G_model_filename))
        print("Generator saved to %s" % G_model_filename)
        if D_model_filename is not None:
            torch.save(self.D.state_dict(), os.path.join(self.model_store_path, D_model_filename))
            print("Discriminator saved to %s" % D_model_filename)
        if D_optim_filename is not None:
            torch.save(self.d_optimizer.state_dict(), os.path.join(self.model_store_path, D_optim_filename))
            print("D_optimizer saved to %s" % D_optim_filename)
        if G_optim_filename is not None:
            torch.save(self.g_optimizer.state_dict(), os.path.join(self.model_store_path, G_optim_filename))
            print("G_optimizer saved to %s" % G_optim_filename)

    def train(self, epoch_start, epoch_end, batch_size=16, save_every_x_epochs=5):
        # set mode for networks
        self.G.train()
        self.D.train()
        self.ContextNet.eval()
        if epoch_start > 0:
            self._load_model("G_model_epoch_%s.pkl" % str(epoch_start).zfill(LEN_ZFILL),
                             "D_model_epoch_%s.pkl" % str(epoch_start).zfill(LEN_ZFILL),
                             "G_optim_epoch_%s.pkl" % str(epoch_start).zfill(LEN_ZFILL),
                             "D_optim_epoch_%s.pkl" % str(epoch_start).zfill(LEN_ZFILL))

        # turn on debugging related to gradient
        torch.autograd.set_detect_anomaly(True)

        # create data loader for yielding batches
        dataset = DatasetDefiner(self.name, self.im_size, batch_size)
        dataset.load_training_data(out_file=self.training_data_file)
        dataset = dataset.get_attribute("training_data")
        dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=True)  # batchsize = 1
        total_batch_num = dataset.get_num_of_batchs()

        # variables
        imgs, out_reconstruction, out_instant_pred, out_longterm_pred = None, None, None, None

        # progress bar
        progress = ProgressBar(total_batch_num * (epoch_end - epoch_start), fmt=ProgressBar.FULL)
        print("Started time:", datetime.datetime.now())

        # loop over epoch
        for epoch in range(epoch_start, epoch_end):
            iter_count = 0

            # get data info for each whole video
            for video_idx, clip_indices in dataloader:

                # zero grad for generator due to RNN
                self.G.zero_grad()

                #
                d_loss_real, d_loss_fake = 0, 0
                g_loss, d_loss = 0, 0
                g_loss_total = 0
                instant_loss, longterm_loss, reconst_loss = 0, 0, 0

                # process short clip
                for clip_index in clip_indices:

                    # load batch data
                    assert len(clip_index) == 2
                    imgs = dataset.data[video_idx][clip_index[0]:clip_index[1]].to(self.device)

                    # ============================== Discriminator optimizing ==============================

                    # discriminator loss with real images
                    real_D_input = torch.cat([imgs[:-1], imgs[1:], imgs[1:]], dim=1)
                    real_D_output = self.D(real_D_input)
                    d_loss_real = self.loss(real_D_output, torch.ones_like(real_D_output).to(self.device))

                    # get fake outputs from Generator
                    in_context, out_reconstruction, out_instant_pred, out_longterm_pred = self.G(imgs)

                    # discriminator loss with fake images
                    fake_D_input = torch.cat([imgs[:-1], out_instant_pred[:-1], out_longterm_pred[:-1]], dim=1)
                    fake_D_output = self.D(fake_D_input)
                    d_loss_fake = self.loss(fake_D_output, torch.zeros_like(fake_D_output).to(self.device))

                    # optimize discriminator
                    d_loss = 0.5*d_loss_fake + 0.5*d_loss_real
                    self.D.zero_grad()
                    d_loss.backward(retain_graph=True)
                    self.d_optimizer.step()

                    # ============================== Generator optimizing ==============================

                    fake_D_output = self.D(fake_D_input)
                    g_loss = self.loss(fake_D_output, torch.ones_like(fake_D_output).to(self.device))

                    # groundtruth context
                    gt_context = self.ContextNet(imgs)

                    # define loss functions, may be different for partial losses
                    L2_loss, L1_loss = nn.MSELoss(), nn.L1Loss()

                    # context loss
                    context_loss = L2_loss(in_context, gt_context)

                    # prediction losses
                    dx_instant_pred, dy_instant_pred = image_gradient(out_instant_pred[:-1], out_abs=True)
                    dx_longterm_pred, dy_longterm_pred = image_gradient(out_longterm_pred[:-1], out_abs=True)
                    dx_input, dy_input = image_gradient(imgs[1:], out_abs=True)
                    instant_loss = L2_loss(out_instant_pred[:-1], imgs[1:]) + \
                        L1_loss(dx_instant_pred, dx_input) + L1_loss(dy_instant_pred, dy_input)
                    longterm_loss = L2_loss(out_longterm_pred[:-1], imgs[1:]) + \
                        L1_loss(dx_longterm_pred, dx_input) + L1_loss(dy_longterm_pred, dy_input)

                    # reconstruction loss
                    dx_recons_pred, dy_recons_pred = image_gradient(out_reconstruction[:-1], out_abs=True)
                    dx_input, dy_input = image_gradient(imgs[:-1], out_abs=True)
                    reconst_loss = L2_loss(out_reconstruction[:-1], imgs[:-1]) + \
                        L1_loss(dx_recons_pred, dx_input) + L1_loss(dy_recons_pred, dy_input)

                    # total loss
                    g_loss_total = loss_weights["g_loss"]*g_loss + loss_weights["context"]*context_loss + \
                        loss_weights["reconst"]*reconst_loss + loss_weights["instant"]*instant_loss + loss_weights["longterm"]*longterm_loss

                    self.D.zero_grad()
                    g_loss_total.backward()
                    self.g_optimizer.step()

                    # emit losses for visualization
                    msg = " [(ctx: %.2f, rec: %.2f, ins: %.2f, ltm: %.2f), G_total: %.2f, G: %.2f, D: %.2f]" \
                          % (context_loss.data.item(), reconst_loss.data.item(), instant_loss.data.item(), longterm_loss.data.item(),
                             g_loss_total.data.item(), g_loss.data.item(), d_loss.data.item())
                    progress.current += 1
                    progress(msg)

                    # ============ TensorBoard logging ============#
                    # Log the scalar values
                    info = {
                       'Loss D Real': d_loss_real.data.item(),
                       'Loss D Fake': d_loss_fake.data.item(),
                       'Loss D': d_loss.data.item(),
                       'Loss G total': g_loss_total.data.item(),
                       'Loss G': g_loss.data.item(),
                       'Loss instant': instant_loss.data.item(),
                       'Loss longterm': longterm_loss.data.item(),
                       'Loss reconst': reconst_loss.data.item(),
                    }
                    for tag, value in info.items():
                        self.logger.add_scalar(tag, value, epoch * total_batch_num + iter_count)

                    iter_count += 1

            self.logger.flush()

            # Saving model and sampling images every X epochs
            if (epoch + 1) % save_every_x_epochs == 0:
                self._save_model("G_model_epoch_%s.pkl" % str(epoch + 1).zfill(LEN_ZFILL),
                                 "D_model_epoch_%s.pkl" % str(epoch + 1).zfill(LEN_ZFILL),
                                 "G_optim_epoch_%s.pkl" % str(epoch + 1).zfill(LEN_ZFILL),
                                 "D_optim_epoch_%s.pkl" % str(epoch + 1).zfill(LEN_ZFILL))

                # Denormalize images and save them in grid 8x8
                images_to_save = [[tensor_restore(imgs.data.cpu()[0]),
                                   tensor_restore(out_reconstruction.data.cpu()[0]),
                                   tensor_restore(imgs.data.cpu()[1]),
                                   tensor_restore(out_instant_pred.data.cpu()[0]),
                                   tensor_restore(out_longterm_pred.data.cpu()[0])],
                                  [tensor_restore(imgs.data.cpu()[-2]),
                                   tensor_restore(out_reconstruction.data.cpu()[-2]),
                                   tensor_restore(imgs.data.cpu()[-1]),
                                   tensor_restore(out_instant_pred.data.cpu()[-2]),
                                   tensor_restore(out_longterm_pred.data.cpu()[-2])]]
                print([x.shape for x in images_to_save[0]], [x.shape for x in images_to_save[1]])
                images_to_save = [utils.make_grid(images, nrow=1) for images in images_to_save]
                grid = utils.make_grid(images_to_save, nrow=2)
                utils.save_image(grid, "%s/gen_epoch_%s.png" % (self.gen_image_store_path, str(epoch + 1).zfill(LEN_ZFILL)))

        # finish iteration
        progress.done()
        print("Finished time:", datetime.datetime.now())

        # Save the trained parameters
        if (epoch + 1) % save_every_x_epochs != 0:  # not already saved inside loop
            self._save_model("G_model_epoch_%s.pkl" % str(epoch + 1).zfill(LEN_ZFILL),
                             "D_model_epoch_%s.pkl" % str(epoch + 1).zfill(LEN_ZFILL),
                             "G_optim_epoch_%s.pkl" % str(epoch + 1).zfill(LEN_ZFILL),
                             "D_optim_epoch_%s.pkl" % str(epoch + 1).zfill(LEN_ZFILL))

    # calculate output from pretrained model and store them to files
    # may feed training data to get losses as weights in evaluation
    def infer(self, epoch, batch_size=16, data_set="test_set"):
        assert data_set in ("test_set", "training_set")
        # load pretrained model and set to eval() mode
        self._load_model("G_model_epoch_%s.pkl" % str(epoch).zfill(LEN_ZFILL))
        self.G.eval()

        # dataloader for yielding batches
        dataset = DatasetDefiner(self.name, self.im_size, batch_size)
        if data_set == "test_set":
            dataset.load_evaluation_data(out_file=self.evaluation_data_file)
            dataset = dataset.get_attribute("evaluation_data")
        else:
            dataset.load_training_data(out_file=self.training_data_file)
            dataset = dataset.get_attribute("training_data")

        dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False)  # batchsize = 1
        total_batch_num = dataset.get_num_of_batchs()

        # init variables for batch evaluation
        results_reconst, results_instant, results_longterm = [], [], []

        # progress bar
        progress = ProgressBar(total_batch_num, fmt=ProgressBar.FULL)
        print("Started time:", datetime.datetime.now())

        with torch.no_grad():
            # get data info for each whole video
            for video_idx, clip_indices in dataloader:
                output_reconst, output_instant, output_longterm = [], [], []

                # evaluate a batch
                for clip_idx in clip_indices:
                    assert len(clip_idx) == 2
                    imgs = dataset.data[video_idx][clip_idx[0]:clip_idx[1]].to(self.device)
                    _, out_reconstruction, out_instant_pred, out_longterm_pred = self.G(imgs)

                    # store results
                    output_reconst.append(out_reconstruction)
                    output_instant.append(out_instant_pred)
                    output_longterm.append(out_longterm_pred)

                    progress.current += 1
                    progress()

                results_reconst.append(torch.cat(output_reconst, dim=0))
                results_instant.append(torch.cat(output_instant, dim=0))
                results_longterm.append(torch.cat(output_longterm, dim=0))

        progress.done()
        print("Finished time:", datetime.datetime.now())

        # store data to file
        data = {"reconst": results_reconst,
                "instant": results_instant,
                "longterm": results_longterm}
        out_file = self.output_store_path + '/out_epoch_%s_data_%s.pt' % (str(epoch).zfill(LEN_ZFILL), data_set)
        torch.save(data, out_file)
        print("Data saved to %s" % out_file)

    # evaluation from frame-level groundtruth and (real eval data, output eval data)
    def evaluate(self, epoch):

        # define function for computing anomaly score
        # input tensor shape: (n, C, H, W)
        # power: used for combining channels (1=abs, 2=square)
        def calc_score(tensor, power=1, patch_size=5):
            assert power in (1, 2) and patch_size % 2
            # combine channels
            tensor2 = torch.sum(torch.abs(tensor) if power == 1 else tensor**2, dim=1)
            tensor2.unsqueeze_(1)
            # convolution for most salient patch
            weight = torch.ones(1, 1, patch_size, patch_size)
            padding = patch_size // 2
            # heatmaps = [F.conv2d(item, weight, stride=1, padding=padding).numpy() for item in tensor2]
            heatmaps = F.conv2d(tensor2, weight, stride=1, padding=padding).numpy()
            # get sum value and position of the patch
            scores = [np.max(heatmap) for heatmap in heatmaps]
            positions = [np.where(heatmap == np.max(heatmap)) for heatmap in heatmaps]
            positions = [(position[0][0], position[1][0]) for position in positions]
            # return scores and positions
            return {"score": scores, "position": positions}

        # load real data
        eval_data_list = torch.load(self.evaluation_data_file)
        assert isinstance(eval_data_list, (list, tuple))
        print("Eval data shape:", [video.shape for video in eval_data_list])

        # load outputted data
        output_file = self.output_store_path + '/out_epoch_%s_data_test_set.pt' % str(epoch).zfill(LEN_ZFILL)
        outputs = torch.load(output_file)
        assert isinstance(outputs, dict)
        reconst_list, instant_list, longterm_list = outputs["reconst"], outputs["instant"], outputs["longterm"]
        assert isinstance(reconst_list, (list, tuple))
        assert isinstance(instant_list, (list, tuple))
        assert isinstance(longterm_list, (list, tuple))

        # evaluation
        assert len(eval_data_list) == len(reconst_list) == len(instant_list) == len(longterm_list)
        # torch.tensor([torch.max(score) for score in torch.abs(out_reconstruction[0] - imgs[0, :-1])])
        reconst_diff = [reconst_list[i][:-1].cpu() - eval_data_list[i][:-1].cpu() for i in range(len(eval_data_list))]
        instant_diff = [instant_list[i][:-1].cpu() - eval_data_list[i][1:].cpu() for i in range(len(eval_data_list))]
        longterm_diff = [longterm_list[i][:-1].cpu() - eval_data_list[i][1:].cpu() for i in range(len(eval_data_list))]

        # compute patch scores and localize positions -> list of dicts
        # temporary: get only scores
        reconst_patches = [calc_score(tensor)["score"] for tensor in reconst_diff]
        instant_patches = [calc_score(tensor)["score"] for tensor in instant_diff]
        longterm_patches = [calc_score(tensor)["score"] for tensor in longterm_diff]

        # return auc(s)
        dataset = DatasetDefiner(self.name, self.im_size, -1)
        return dataset.evaluate(reconst_patches), dataset.evaluate(instant_patches), dataset.evaluate(longterm_patches)
