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

from utils import get_img_shape, image_gradient, images_restore, ProgressBar, DatasetDefiner
from AnomaNet import ExpandedSE, GTNet
from AnomaNet import AnomaNet as Generator
from CONFIG import loss_weights

LEN_ZFILL = 5


class Discriminator(nn.Module):
    def __init__(self, im_size, device, use_optical_flow=True):
        super().__init__()
        self.use_optical_flow = use_optical_flow
        # set device & input shape
        self.device = device
        self.IM_SIZE = get_img_shape(im_size)

        # architecture
        # pair(F[i], F[i+1]) or pair(F[i], F_hat[i+1])
        in_channel = 5 if self.use_optical_flow else 6
        n_base_channel = 16 * 4  # no. of channels to start
        self.network = nn.Sequential(nn.Conv2d(in_channel, n_base_channel, kernel_size=3, stride=1, padding=1),  # end block 0
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

    # X must have shape (B, 6, H, W)
    # 6 channels: F[i], F_instant[i+1] or F_longterm[i+1]
    def forward(self, X):
        logit = self.network(X)
        prob = self.output(logit)
        return logit, prob


# name: dataset's name
class DCGAN(object):
    def __init__(self, name, im_size, store_path, use_optical_flow=True, device_str=None, use_progress_bar=True):
        self.use_optical_flow = use_optical_flow
        self.use_progress_bar = use_progress_bar
        #
        self.name = name
        self.im_size = im_size
        # paths
        self.store_path = os.path.join(store_path, self.name)
        self.input_store_path = self.store_path + "/input_data_%s_%s" \
            % (str(self.im_size[0]).zfill(3), str(self.im_size[1]).zfill(3))  # data for training and evaluation
        self.training_store_path = self.input_store_path + "/training"
        self.evaluation_store_path = self.input_store_path + "/evaluation"
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
        self.G = Generator(self.im_size, self.device, self.use_optical_flow)
        self.D = Discriminator(self.im_size, self.device, self.use_optical_flow)
        self.loss = nn.BCEWithLogitsLoss()  # BCEWithLogitsLoss replacing BCELoss

        # ADAM optimizers
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=0.00002, betas=(0.5, 0.9))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.9))

        # Set the logger
        self.logger = SummaryWriter(self.log_path)
        self.logger.flush()

    def _create_path(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    # create necessary directories
    def _create_all_paths(self):
        # create_path(self.input_store_path)
        self._create_path(self.training_store_path)
        self._create_path(self.evaluation_store_path)
        self._create_path(self.model_store_path)
        self._create_path(self.gen_image_store_path)
        self._create_path(self.output_store_path)
        self._create_path(self.log_path)

    # load pretrained models and optimizers
    def _load_model(self, G_model_filename, D_model_filename=None, G_optim_filename=None, D_optim_filename=None, silence=True):
        loaded_data = torch.load(os.path.join(self.model_store_path, G_model_filename))
        self.G.load_state_dict(loaded_data['G'])
        if not silence:
            print("Generator loaded from %s" % G_model_filename)
        if D_model_filename is not None:
            self.D.load_state_dict(torch.load(os.path.join(self.model_store_path, D_model_filename)))
            if not silence:
                print("Discriminator loaded from %s" % D_model_filename)
        if G_optim_filename is not None:
            self.g_optimizer.load_state_dict(torch.load(os.path.join(self.model_store_path, G_optim_filename)))
            if not silence:
                print("G_optimizer loaded from %s" % G_optim_filename)
        if D_optim_filename is not None:
            self.d_optimizer.load_state_dict(torch.load(os.path.join(self.model_store_path, D_optim_filename)))
            if not silence:
                print("D_optimizer loaded from %s" % D_optim_filename)
        return loaded_data['iter']

    # save pretrained models and optimizers
    def _save_model(self, G_model_filename, D_model_filename=None, G_optim_filename=None, D_optim_filename=None, iter_count=None, silence=True):
        torch.save({'G': self.G.state_dict(), 'iter': iter_count}, os.path.join(self.model_store_path, G_model_filename))
        if not silence:
            print("Generator saved to %s" % G_model_filename)
        if D_model_filename is not None:
            torch.save(self.D.state_dict(), os.path.join(self.model_store_path, D_model_filename))
            if not silence:
                print("Discriminator saved to %s" % D_model_filename)
        if D_optim_filename is not None:
            torch.save(self.d_optimizer.state_dict(), os.path.join(self.model_store_path, D_optim_filename))
            if not silence:
                print("D_optimizer saved to %s" % D_optim_filename)
        if G_optim_filename is not None:
            torch.save(self.g_optimizer.state_dict(), os.path.join(self.model_store_path, G_optim_filename))
            if not silence:
                print("G_optimizer saved to %s" % G_optim_filename)

    def train(self, epoch_start, epoch_end, batch_size=16, save_every_x_epochs=5):
        # set mode for networks
        self.G.train()
        self.D.train()
        self.ContextNet.eval()
        if epoch_start > 0:
            iter_count = self._load_model("G_model_epoch_%s.pkl" % str(epoch_start).zfill(LEN_ZFILL),
                                          "D_model_epoch_%s.pkl" % str(epoch_start).zfill(LEN_ZFILL),
                                          "G_optim_epoch_%s.pkl" % str(epoch_start).zfill(LEN_ZFILL),
                                          "D_optim_epoch_%s.pkl" % str(epoch_start).zfill(LEN_ZFILL))
            assert isinstance(iter_count, int)
        else:
            iter_count = 0

        # turn on debugging related to gradient
        torch.autograd.set_detect_anomaly(True)

        # create data loader
        dataset = DatasetDefiner(self.name, self.im_size, self.training_store_path, mode="train")

        # variables
        n_clip = dataset.get_info("n_clip_train")
        imgs, out_reconstruction, out_instant_pred, out_longterm_pred = None, None, None, None

        # progress bar
        progress = None
        if self.use_progress_bar:
            progress = ProgressBar(n_clip * (epoch_end - epoch_start), fmt=ProgressBar.FULL)
        print("Started time:", datetime.datetime.now())

        # loop over epoch
        for epoch in range(epoch_start, epoch_end):
            np.random.seed(epoch)   # to make sure getting similar results when training from pretrained models
            torch.manual_seed(epoch)
            clip_order = np.random.permutation(n_clip)

            # process each clip
            for clip_idx in clip_order:
                dataset.load_data(clip_idx)
                # >>>>>>>> TODO: check whether it is better if shuffle is True <<<<<<<<<
                dataloader = torch.utils.data.DataLoader(dataset.training_data, batch_size, shuffle=False)
                #
                d_loss_real, d_loss_fake = 0, 0
                g_loss, d_loss = 0, 0
                g_loss_total = 0
                instant_loss, longterm_loss, reconst_loss = 0, 0, 0

                # process batch
                msg = ""
                for data_batch in dataloader:
                    # skip last batch with very few samples
                    if len(data_batch) < 3:
                        continue
                    # normalize data to range [0, 1] and then [-1, 1]
                    imgs = data_batch[:, :3, :, :].to(self.device) / 255.
                    imgs *= 2.
                    imgs -= 1.
                    #
                    if self.use_optical_flow:
                        flows = data_batch[:, 3:, :, :].to(self.device)
                        assert len(imgs) == len(flows)
                        if torch.sum(torch.abs(flows[-1])) == 0.0:
                            imgs, flows = imgs[:-1], flows[:-1]

                    # ============================== Discriminator optimizing ==============================

                    # discriminator loss with real data
                    if not self.use_optical_flow:
                        real_D_input = torch.cat([imgs[:-1], imgs[1:]], dim=1)
                    else:
                        real_D_input = torch.cat([imgs, flows], dim=1)
                    real_D_output_logit, _ = self.D(real_D_input)
                    d_loss_real = self.loss(real_D_output_logit, torch.ones_like(real_D_output_logit).to(self.device))

                    # get fake outputs from Generator
                    in_context, out_reconstruction, out_instant_pred, out_longterm_pred = self.G(imgs)

                    # discriminator loss with fake data
                    last_idx = len(imgs) if self.use_optical_flow else -1   # remove last frame if using frame prediction
                    fake_D_input_instant = torch.cat([imgs[:last_idx], out_instant_pred[:last_idx]], dim=1)
                    fake_D_input_longterm = torch.cat([imgs[:last_idx], out_longterm_pred[:last_idx]], dim=1)

                    fake_D_output_logit_instant, _ = self.D(fake_D_input_instant)
                    fake_D_output_logit_longterm, _ = self.D(fake_D_input_longterm)
                    d_loss_fake_instant = self.loss(fake_D_output_logit_instant,
                                                    torch.zeros_like(fake_D_output_logit_instant).to(self.device))
                    d_loss_fake_longterm = self.loss(fake_D_output_logit_longterm,
                                                     torch.zeros_like(fake_D_output_logit_longterm).to(self.device))
                    d_loss_fake = 0.5 * (d_loss_fake_instant + d_loss_fake_longterm)

                    # optimize discriminator
                    d_loss = 0.5*d_loss_fake + 0.5*d_loss_real
                    self.D.zero_grad()
                    d_loss.backward(retain_graph=True)
                    self.d_optimizer.step()

                    # ============================== Generator optimizing ==============================

                    fake_D_output_logit_instant, _ = self.D(fake_D_input_instant)
                    fake_D_output_logit_longterm, _ = self.D(fake_D_input_longterm)
                    g_loss_instant = self.loss(fake_D_output_logit_instant,
                                               torch.ones_like(fake_D_output_logit_instant).to(self.device))
                    g_loss_longterm = self.loss(fake_D_output_logit_longterm,
                                                torch.ones_like(fake_D_output_logit_longterm).to(self.device))

                    g_loss = g_loss_instant + g_loss_longterm

                    # groundtruth context
                    gt_context = self.ContextNet((imgs + 1.) * 0.5)  # recover original inputs because of the pretrained model

                    # define loss functions, may be different for partial losses
                    L2_loss, L1_loss = nn.MSELoss(), nn.L1Loss()

                    # context loss
                    context_loss = L2_loss(in_context, gt_context)

                    # prediction losses
                    if not self.use_optical_flow:
                        dx_instant_pred, dy_instant_pred = image_gradient(out_instant_pred[:-1], out_abs=True)
                        dx_longterm_pred, dy_longterm_pred = image_gradient(out_longterm_pred[:-1], out_abs=True)
                        dx_input, dy_input = image_gradient(imgs[1:], out_abs=True)
                        instant_loss = L2_loss(out_instant_pred[:-1], imgs[1:]) + \
                            L1_loss(dx_instant_pred, dx_input) + L1_loss(dy_instant_pred, dy_input)
                        longterm_loss = L2_loss(out_longterm_pred[:-1], imgs[1:]) + \
                            L1_loss(dx_longterm_pred, dx_input) + L1_loss(dy_longterm_pred, dy_input)
                    else:
                        instant_loss = L1_loss(out_instant_pred, flows)
                        longterm_loss = L1_loss(out_longterm_pred, flows)

                    # reconstruction loss
                    last_idx = len(imgs) if self.use_optical_flow else -1
                    dx_recons_pred, dy_recons_pred = image_gradient(out_reconstruction[:last_idx], out_abs=True)
                    dx_input, dy_input = image_gradient(imgs[:last_idx], out_abs=True)
                    reconst_loss = L2_loss(out_reconstruction[:last_idx], imgs[:last_idx]) + \
                        L1_loss(dx_recons_pred, dx_input) + L1_loss(dy_recons_pred, dy_input)

                    # total loss
                    g_loss_total = loss_weights["g_loss"]*g_loss + loss_weights["context"]*context_loss + \
                        loss_weights["reconst"]*reconst_loss + loss_weights["instant"]*instant_loss + loss_weights["longterm"]*longterm_loss

                    self.G.zero_grad()
                    g_loss_total.backward()
                    self.g_optimizer.step()

                    # ============ TensorBoard logging ============#
                    # Log the scalar values
                    info = {
                       'Loss D Real': d_loss_real.data.item(),
                       'Loss D Fake': d_loss_fake.data.item(),
                       'Loss D': d_loss.data.item(),
                       'Loss G total': g_loss_total.data.item(),
                       'Loss G': g_loss.data.item(),
                       'Loss context': context_loss.data.item(),
                       'Loss instant': instant_loss.data.item(),
                       'Loss longterm': longterm_loss.data.item(),
                       'Loss reconst': reconst_loss.data.item(),
                    }
                    for tag, value in info.items():
                        self.logger.add_scalar(tag, value, iter_count)

                    iter_count += 1

                    # emit losses for visualization
                    msg = " [(ctx: %.2f, rec: %.2f, ins: %.2f, ltm: %.2f), G_total: %.2f, G: %.2f, D: %.2f]" \
                          % (context_loss.data.item(), reconst_loss.data.item(), instant_loss.data.item(), longterm_loss.data.item(),
                             g_loss_total.data.item(), g_loss.data.item(), d_loss.data.item())

                if progress is not None:
                    progress.current += 1
                    progress(msg)

            self.logger.flush()

            # Saving model and sampling images every X epochs
            if (epoch + 1) % save_every_x_epochs == 0:
                self._save_model("G_model_epoch_%s.pkl" % str(epoch + 1).zfill(LEN_ZFILL),
                                 "D_model_epoch_%s.pkl" % str(epoch + 1).zfill(LEN_ZFILL),
                                 "G_optim_epoch_%s.pkl" % str(epoch + 1).zfill(LEN_ZFILL),
                                 "D_optim_epoch_%s.pkl" % str(epoch + 1).zfill(LEN_ZFILL),
                                 iter_count=iter_count)

                # Denormalize images and save them in grid
                images_to_save = [images_restore(imgs.data[0].cpu()),
                                  images_restore(out_reconstruction.data[0].cpu()),
                                  images_restore(imgs.data[1].cpu()),
                                  images_restore(out_instant_pred.data[0].cpu(), is_optical_flow=self.use_optical_flow),
                                  images_restore(out_longterm_pred.data[0].cpu(), is_optical_flow=self.use_optical_flow)]
                grid = utils.make_grid(images_to_save, nrow=1)
                utils.save_image(grid, "%s/gen_epoch_%s.png" % (self.gen_image_store_path, str(epoch + 1).zfill(LEN_ZFILL)))

        # finish iteration
        if progress is not None:
            progress.done()
        print("Finished time:", datetime.datetime.now())

        # Save the trained parameters
        if (epoch + 1) % save_every_x_epochs != 0:  # not already saved inside loop
            self._save_model("G_model_epoch_%s.pkl" % str(epoch + 1).zfill(LEN_ZFILL),
                             "D_model_epoch_%s.pkl" % str(epoch + 1).zfill(LEN_ZFILL),
                             "G_optim_epoch_%s.pkl" % str(epoch + 1).zfill(LEN_ZFILL),
                             "D_optim_epoch_%s.pkl" % str(epoch + 1).zfill(LEN_ZFILL),
                             iter_count=iter_count)

    # calculate output from pretrained model and store them to files
    # may feed training data to get losses as weights in evaluation
    def infer(self, epoch, batch_size=16, data_set="test_set"):
        assert data_set in ("test_set", "training_set")
        # load pretrained model and set to eval() mode
        self._load_model("G_model_epoch_%s.pkl" % str(epoch).zfill(LEN_ZFILL))
        self.G.eval()

        # dataloader for yielding batches
        if data_set == "test_set":
            dataset = DatasetDefiner(self.name, self.im_size, self.evaluation_store_path, mode="eval")
            n_clip = dataset.get_info("n_clip_test")
        else:
            dataset = DatasetDefiner(self.name, self.im_size, self.training_store_path, mode="train")
            n_clip = dataset.get_info("n_clip_train")

        # init variables for batch evaluation
        # results_reconst, results_instant, results_longterm = [], [], []

        # progress bar
        progress = None
        if self.use_progress_bar:
            progress = ProgressBar(n_clip, fmt=ProgressBar.FULL)
        print("Started time:", datetime.datetime.now())

        with torch.no_grad():
            # process each clip
            for clip_idx in range(n_clip):
                dataset.load_data(clip_idx)
                if data_set == "test_set":
                    dataloader = torch.utils.data.DataLoader(dataset.evaluation_data, batch_size, shuffle=False)
                else:
                    dataloader = torch.utils.data.DataLoader(dataset.training_data, batch_size, shuffle=False)

                output_reconst, output_instant, output_longterm = [], [], []

                # evaluate a batch
                imgs = None
                for data_batch in dataloader:
                    imgs = data_batch[:, :3, :, :].to(self.device) / 255.  # eval ALL video frames
                    imgs *= 2.
                    imgs -= 1.
                    _, batch_reconst, batch_instant_pred, batch_longterm_pred = self.G(imgs)

                    # store results
                    output_reconst.append(batch_reconst.cpu().numpy())
                    output_instant.append(batch_instant_pred.cpu().numpy())
                    output_longterm.append(batch_longterm_pred.cpu().numpy())

                # store data to file
                data = {"reconst": np.concatenate(output_reconst, axis=0),
                        "instant": np.concatenate(output_instant, axis=0),
                        "longterm": np.concatenate(output_longterm, axis=0)}
                out_path = self.output_store_path + '/out_epoch_%s/%s' % (str(epoch).zfill(LEN_ZFILL), data_set[:-4])
                self._create_path(out_path)
                out_file = os.path.join(out_path, '%s.npy' % str(clip_idx + 1).zfill(len(str(n_clip))))
                np.save(out_file, data)
                if not self.use_progress_bar:
                    print("Data saved to %s" % out_file)

                # save example image
                if len(imgs) > 1:
                    images_to_save = [images_restore(imgs.data[-2].cpu().numpy()),
                                      images_restore(data["reconst"][-2]),
                                      images_restore(imgs.data[-1].cpu().numpy()),
                                      images_restore(data["instant"][-2], is_optical_flow=self.use_optical_flow),
                                      images_restore(data["longterm"][-2], is_optical_flow=self.use_optical_flow)]
                else:
                    images_to_save = [images_restore(imgs.data[0].cpu().numpy()),
                                      images_restore(data["reconst"][0]),
                                      images_restore(data["instant"][0], is_optical_flow=self.use_optical_flow),
                                      images_restore(data["longterm"][0], is_optical_flow=self.use_optical_flow)]
                grid = utils.make_grid([torch.tensor(image) for image in images_to_save], nrow=1)
                out_file = os.path.join(out_path, '%s.png' % str(clip_idx + 1).zfill(len(str(n_clip))))
                utils.save_image(grid, out_file)

                if progress is not None:
                    progress.current += 1
                    progress()

        if progress is not None:
            progress.done()
        print("Finished time:", datetime.datetime.now())

    # function for computing anomaly score
    # input tensor shape: (n, C, H, W)
    # power: used for combining channels (1=abs, 2=square)
    def _calc_score(self, tensor, power, patch_size, stride):
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
        assert power in (1, 2) and patch_size % 2
        # combine channels
        tensor2 = torch.sum(torch.abs(tensor) if power == 1 else tensor**2, dim=1)
        tensor2.unsqueeze_(1)
        # convolution for most salient patch
        weight = torch.ones(1, 1, patch_size, patch_size)
        padding = patch_size // 2
        heatmaps = F.conv2d(tensor2, weight, stride=stride, padding=padding).numpy()
        # get sum value and position of the patch
        scores = [np.max(heatmap) for heatmap in heatmaps]
        positions = [np.where(heatmap == np.max(heatmap)) for heatmap in heatmaps]
        positions = [(position[0][0], position[1][0]) for position in positions]
        # return scores and positions
        return {"score": scores, "position": positions}

    # evaluation from frame-level groundtruth and (real eval data, output eval data)
    def evaluate(self, epoch, power=1, patch_size=5, stride=1):
        dataset = DatasetDefiner(self.name, self.im_size, self.evaluation_store_path, mode="eval")
        n_clip = dataset.get_info("n_clip_test")
        reconst_scores, instant_scores, longterm_scores = [], [], []

        for clip_idx in range(n_clip):
            dataset.load_data(clip_idx)

            # get input data
            imgs = dataset.evaluation_data[:][:, :3, :, :]
            if self.use_optical_flow:
                flows = dataset.evaluation_data[:][:, 3:, :, :]

            # get output results
            output_path = self.output_store_path + '/out_epoch_%s/test' % str(epoch).zfill(LEN_ZFILL)
            output_file = os.path.join(output_path, '%s.npy' % str(clip_idx + 1).zfill(len(str(n_clip))))
            output_data = np.load(output_file, allow_pickle=True).item()

            # calc difference tensor and patch scores
            reconst_scores.append(self._calc_score(output_data["reconst"][:-1] - imgs[:-1],
                                                   power, patch_size, stride)["score"])
            if not self.use_optical_flow:
                instant_scores.append(self._calc_score(output_data["instant"][:-1] - imgs[1:],
                                                       power, patch_size, stride)["score"])
                longterm_scores.append(self._calc_score(output_data["longterm"][:-1] - imgs[1:],
                                                        power, patch_size, stride)["score"])
            else:
                instant_scores.append(self._calc_score(output_data["instant"][:-1] - flows[:-1],
                                                       power, patch_size, stride)["score"])
                longterm_scores.append(self._calc_score(output_data["longterm"][:-1] - flows[:-1],
                                                        power, patch_size, stride)["score"])

        # return auc(s)
        auc_reconst_norm = dataset.evaluate(reconst_scores, normalize_each_clip=True)
        auc_reconst = dataset.evaluate(reconst_scores, normalize_each_clip=False)
        auc_instant_norm = dataset.evaluate(instant_scores, normalize_each_clip=True)
        auc_instant = dataset.evaluate(instant_scores, normalize_each_clip=False)
        auc_longterm_norm = dataset.evaluate(longterm_scores, normalize_each_clip=True)
        auc_longterm = dataset.evaluate(longterm_scores, normalize_each_clip=False)
        return auc_reconst_norm, auc_reconst, auc_instant_norm, auc_instant, auc_longterm_norm, auc_longterm
