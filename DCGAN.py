import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import utils
from torch.utils.tensorboard import SummaryWriter
from skimage.measure import compare_ssim as ssim

from utils import get_img_shape, image_gradient, images_restore, ProgressBar, DatasetDefiner, extend_flow_channel_in_batch, visualize_error_map
from AnomaNet import AnomaNet as Generator
from CONFIG import loss_weights

LEN_ZFILL = 5


class Discriminator(nn.Module):
    def __init__(self, im_size, device):
        super().__init__()
        # set device & input shape
        self.device = device
        self.IM_SIZE = get_img_shape(im_size)

        n_filter = 64
        kernel_size = (4, 4)

        # architecture
        self.disc_block_1 = nn.Sequential(nn.Conv2d(6, n_filter, kernel_size, stride=2),
                                          nn.LeakyReLU(negative_slope=0.2))
        self.disc_block_2 = nn.Sequential(nn.Conv2d(n_filter, n_filter*2, kernel_size, stride=2),
                                          nn.BatchNorm2d(n_filter*2),
                                          nn.LeakyReLU(negative_slope=0.2))
        self.disc_block_3 = nn.Sequential(nn.Conv2d(n_filter*2, n_filter*4, kernel_size, stride=2),
                                          nn.BatchNorm2d(n_filter*4),
                                          nn.LeakyReLU(negative_slope=0.2))
        self.disc_block_4 = nn.Sequential(nn.Conv2d(n_filter*4, n_filter*8, kernel_size, stride=2),
                                          nn.BatchNorm2d(n_filter*8))

        self.output = nn.Sigmoid()
        self.to(self.device)

    # data must have shape (B, 6, H, W)
    # 6 channels: true frame & true/fake flow
    def forward(self, data):
        logit = self.disc_block_4(self.disc_block_3(self.disc_block_2(self.disc_block_1(data))))
        prob = self.output(logit)
        return logit, prob


# name: dataset's name
class DCGAN(object):
    def __init__(self, name, im_size, store_path, extension_params, training_gamma=0.9, drop_prob=0.3, device_str=None,
                 use_progress_bar=True, prt_summary=False):
        self.drop_prob = drop_prob
        self.use_progress_bar = use_progress_bar
        #
        self.name = name
        self.im_size = im_size
        self.training_gamma = training_gamma
        # paths
        str_extension = "RNN_%d_cat_%d_elenorm_%d_sigmoid_%d_gamma_%s_chanorm_%d" % (int("RNN" in extension_params),
                                                                                     int("cat_latent" in extension_params),
                                                                                     int("element_norm" in extension_params),
                                                                                     int("sigmoid_instead_tanh" in extension_params),
                                                                                     "auto" if training_gamma < 0 else "%.2f" % training_gamma,
                                                                                     int("channel_norm" in extension_params))
        self.store_path = os.path.join(store_path, self.name, str_extension)
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
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            assert isinstance(device_str, str)
            self.device = torch.device(device_str)

        print("DCGAN init...")
        self.G = Generator(self.im_size, self.device, self.drop_prob, extension_params, prt_summary=prt_summary)
        self.D = Discriminator(self.im_size, self.device)
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
        self.G.set_W_softs(loaded_data['W_softs'])
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
        torch.save({'G': self.G.state_dict(), 'iter': iter_count, 'W_softs': self.G.get_W_softs()},
                   os.path.join(self.model_store_path, G_model_filename))
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

    def train(self, epoch_start, epoch_end, batch_size=16, save_every_x_epochs=None):
        # set mode for networks
        self.G.train()
        self.D.train()
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
        n_clip = dataset.get_n_clip("train")
        frames, flows, frames_hat, flows_hat = None, None, None, None

        # progress bar
        progress = None
        if self.use_progress_bar:
            progress = ProgressBar(n_clip * (epoch_end - epoch_start), use_ETA=True)
        print("Started time:", datetime.datetime.now())

        # define loss functions, may be different for partial losses
        L2_loss, L1_loss = nn.MSELoss(), nn.L1Loss()

        # loop over epoch
        msg = ""
        for epoch in range(epoch_start, epoch_end):
            np.random.seed(epoch)   # to make sure getting similar results when training from pretrained models
            torch.manual_seed(epoch)
            clip_order = np.random.permutation(n_clip)

            # process each clip
            for clip_idx in clip_order:
                self.G.reset_hidden_tensor()
                dataset.load_data(clip_idx)
                # >>>>>>>> TODO: check whether it is better if shuffle is True <<<<<<<<<
                dataloader = torch.utils.data.DataLoader(dataset.data["train"], batch_size, shuffle=False)

                # process batch
                for data_batch in dataloader:
                    # skip last batch with very few samples
                    if len(data_batch) < 2:
                        continue
                    # normalize data to range [0, 1] and then [-1, 1]
                    frames = data_batch[:, :3, :, :].to(self.device) / 255.
                    frames *= 2.
                    frames -= 1.
                    #
                    flows = extend_flow_channel_in_batch(data_batch[:, 3:, :, :]).to(self.device)
                    assert len(frames) == len(flows)
                    if torch.sum(torch.abs(flows[-1])) == 0.0:
                        frames, flows = frames[:-1], flows[:-1]

                    # ============================== Discriminator optimizing ==============================
                    self.D.zero_grad()
                    # discriminator loss with real data
                    real_D_input = torch.cat([frames, flows], dim=1)
                    real_D_output_logit, _ = self.D(real_D_input)
                    d_loss_real = self.loss(real_D_output_logit,
                                            torch.ones_like(real_D_output_logit).to(self.device))

                    # get fake outputs from Generator
                    gamma = iter_count / (iter_count + 1) if self.training_gamma < 0 else self.training_gamma
                    frames_hat, flows_hat = self.G(frames, gamma)   # default gamma is 0.9

                    # discriminator loss with fake data
                    fake_D_input = torch.cat([frames, flows_hat], dim=1)
                    fake_D_output_logit, _ = self.D(fake_D_input)
                    d_loss_fake = self.loss(fake_D_output_logit,
                                            torch.zeros_like(fake_D_output_logit).to(self.device))

                    # optimize discriminator
                    d_loss = 0.5*d_loss_fake + 0.5*d_loss_real
                    d_loss.backward(retain_graph=True)
                    self.d_optimizer.step()

                    # ============================== Generator optimizing ==============================

                    fake_D_output_logit, _ = self.D(fake_D_input)
                    g_loss = self.loss(fake_D_output_logit,
                                       torch.ones_like(fake_D_output_logit).to(self.device))

                    # frame loss
                    dx_frame_in, dy_frame_in = image_gradient(frames, out_abs=True)
                    dx_frame_out, dy_frame_out = image_gradient(frames_hat, out_abs=True)
                    frame_loss = L2_loss(frames, frames_hat) + \
                        torch.mean(torch.abs(dx_frame_in - dx_frame_out) + torch.abs(dy_frame_in - dy_frame_out))

                    # flow loss
                    flow_loss = L1_loss(flows, flows_hat)

                    # total loss
                    g_loss_total = loss_weights["g_loss"]*g_loss + loss_weights["frame"]*frame_loss + loss_weights["flow"]*flow_loss

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
                       'Loss frame': frame_loss.data.item(),
                       'Loss flow': flow_loss.data.item(),
                    }
                    for tag, value in info.items():
                        self.logger.add_scalar(tag, value, iter_count)

                    iter_count += 1

                    # emit losses for visualization
                    msg = " [(frame: %.2f, flow: %.2f, G_loss: %.2f), G_total: %.2f, D: %.2f]" \
                          % (frame_loss.data.item(), flow_loss.data.item(), g_loss.data.item(),
                             g_loss_total.data.item(), d_loss.data.item())

                if progress is not None:
                    progress.current += 1
                    progress(msg)

            self.logger.flush()

            # Saving model and sampling images every X epochs
            if save_every_x_epochs is not None and (epoch + 1) % save_every_x_epochs == 0:
                self._save_model("G_model_epoch_%s.pkl" % str(epoch + 1).zfill(LEN_ZFILL),
                                 "D_model_epoch_%s.pkl" % str(epoch + 1).zfill(LEN_ZFILL),
                                 "G_optim_epoch_%s.pkl" % str(epoch + 1).zfill(LEN_ZFILL),
                                 "D_optim_epoch_%s.pkl" % str(epoch + 1).zfill(LEN_ZFILL),
                                 iter_count=iter_count)

                # Denormalize images and save them in grid
                images_to_save = [images_restore(frames.data[0].cpu()),
                                  images_restore(frames_hat.data[0].cpu()),
                                  visualize_error_map((images_restore(frames.data[0].cpu()) - images_restore(frames_hat.data[0].cpu()))**2),
                                  images_restore(flows.data[0].cpu(), is_optical_flow=True),
                                  images_restore(flows_hat.data[0].cpu(), is_optical_flow=True),
                                  visualize_error_map((images_restore(flows.data[0].cpu(), is_optical_flow=True) -
                                                       images_restore(flows_hat.data[0].cpu(), is_optical_flow=True))**2)]
                grid = utils.make_grid(images_to_save, nrow=1)
                utils.save_image(grid, "%s/gen_epoch_%s.png" % (self.gen_image_store_path, str(epoch + 1).zfill(LEN_ZFILL)))

        # finish iteration
        if progress is not None:
            progress.done()
        print(msg)
        print("Finished time:", datetime.datetime.now())

        # Save the trained parameters
        if save_every_x_epochs is None or epoch_end % save_every_x_epochs != 0:  # not already saved inside loop
            self._save_model("G_model_epoch_%s.pkl" % str(epoch_end).zfill(LEN_ZFILL),
                             "D_model_epoch_%s.pkl" % str(epoch_end).zfill(LEN_ZFILL),
                             "G_optim_epoch_%s.pkl" % str(epoch_end).zfill(LEN_ZFILL),
                             "D_optim_epoch_%s.pkl" % str(epoch_end).zfill(LEN_ZFILL),
                             iter_count=iter_count)
            # Denormalize images and save them in grid
            images_to_save = [images_restore(frames.data[0].cpu()),
                              images_restore(frames_hat.data[0].cpu()),
                              visualize_error_map((images_restore(frames.data[0].cpu()) - images_restore(frames_hat.data[0].cpu()))**2),
                              images_restore(flows.data[0].cpu(), is_optical_flow=True),
                              images_restore(flows_hat.data[0].cpu(), is_optical_flow=True),
                              visualize_error_map((images_restore(flows.data[0].cpu(), is_optical_flow=True) -
                                                   images_restore(flows_hat.data[0].cpu(), is_optical_flow=True))**2)]
            grid = utils.make_grid(images_to_save, nrow=1)
            utils.save_image(grid, "%s/gen_epoch_%s.png" % (self.gen_image_store_path, str(epoch_end).zfill(LEN_ZFILL)))

    # calculate output from pretrained model and store them to files
    # may feed training data to get losses as weights in evaluation
    def infer(self, epoch, batch_size=16, part="test"):
        assert part in ("train", "test")
        # load pretrained model and set to eval() mode
        self._load_model("G_model_epoch_%s.pkl" % str(epoch).zfill(LEN_ZFILL))
        self.G.eval()

        # dataloader for yielding batches
        dataset = DatasetDefiner(self.name, self.im_size,
                                 self.evaluation_store_path if part == "test" else self.training_store_path,
                                 mode=part)
        n_clip = dataset.get_n_clip(part)

        # progress bar
        progress = None
        if self.use_progress_bar:
            progress = ProgressBar(n_clip, use_ETA=True)
        print("Started time:", datetime.datetime.now())

        with torch.no_grad():
            frames, flows, frames_hat, flows_hat = None, None, None, None
            # process each clip
            for clip_idx in range(n_clip):
                self.G.reset_hidden_tensor()
                dataset.load_data(clip_idx)
                dataloader = torch.utils.data.DataLoader(dataset.data[part], batch_size, shuffle=False)

                output_frames, output_flows = [], []

                # evaluate a batch
                gamma = 1.
                for data_batch in dataloader:
                    frames = data_batch[:, :3, :, :].to(self.device) / 255.  # eval ALL video frames
                    frames *= 2.
                    frames -= 1.
                    flows = extend_flow_channel_in_batch(data_batch[:, 3:, :, :])
                    frames_hat, flows_hat = self.G(frames, gamma)

                    # store results
                    output_frames.append(frames_hat.cpu().numpy())
                    output_flows.append(flows_hat.cpu().numpy())

                # store data to file
                data = {"frames_hat": np.concatenate(output_frames, axis=0),
                        "flows_hat": np.concatenate(output_flows, axis=0)}
                out_path = self.output_store_path + '/out_epoch_%s/%s' % (str(epoch).zfill(LEN_ZFILL), part)
                self._create_path(out_path)
                out_file = os.path.join(out_path, 'clip_%s.npy' % str(clip_idx + 1).zfill(len(str(n_clip))))
                np.save(out_file, data)
                if not self.use_progress_bar:
                    print("Data saved to %s" % out_file)

                # save example image
                images_to_save = [images_restore(frames.data[0].cpu().numpy()),
                                  images_restore(frames_hat.data[0].cpu().numpy()),
                                  visualize_error_map((images_restore(frames.data[0].cpu().numpy()) -
                                                       images_restore(frames_hat.data[0].cpu().numpy()))**2),
                                  images_restore(flows.data[0].cpu().numpy(), is_optical_flow=True),
                                  images_restore(flows_hat.data[0].cpu().numpy(), is_optical_flow=True),
                                  visualize_error_map((images_restore(flows.data[0].cpu().numpy(), is_optical_flow=True) -
                                                       images_restore(flows_hat.data[0].cpu().numpy(), is_optical_flow=True))**2)]
                grid = utils.make_grid([torch.tensor(image) for image in images_to_save], nrow=1)
                out_file = os.path.join(out_path, '%s.png' % str(clip_idx + 1).zfill(len(str(n_clip))))
                utils.save_image(grid, out_file)

                if progress is not None:
                    progress.current += 1
                    progress()

        if progress is not None:
            progress.done()
        print("Finished time:", datetime.datetime.now())

    # SSIM on input and reconstructed frames
    def _calc_score_SSIM(self, data):
        # extract data
        frames, frames_hat, _, _ = data
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()
        if isinstance(frames_hat, torch.Tensor):
            frames_hat = frames_hat.cpu().numpy()
        SSIM_scores = [ssim(np.transpose(frame, (1, 2, 0)), np.transpose(frame_hat, (1, 2, 0)),
                            data_range=np.max([frame, frame_hat]) - np.min([frame, frame_hat]), multichannel=True)
                       for (frame, frame_hat) in zip(frames, frames_hat)]
        return np.array(SSIM_scores)

    # function for computing anomaly score from [frames, frames_hat, flows, flows_hat]
    # each input tensor shape: (n, C, H, W)
    # power: used for combining channels (1=abs, 2=square)
    def _calc_score(self, data, patch_size, stride, power=2):
        assert power in (1, 2)

        # extract data
        frames, frames_hat, flows, flows_hat = data
        if not isinstance(frames, torch.Tensor):
            frames = torch.tensor(frames)
        if not isinstance(frames_hat, torch.Tensor):
            frames_hat = torch.tensor(frames_hat)
        if not isinstance(flows, torch.Tensor):
            flows = torch.tensor(flows)
        if not isinstance(flows_hat, torch.Tensor):
            flows_hat = torch.tensor(flows_hat)

        flows = extend_flow_channel_in_batch(flows)

        # find max patch of optical flow
        kernel = torch.ones(1, 1, patch_size, patch_size)
        padding = patch_size // 2
        flows_diff = torch.sum(torch.abs(flows - flows_hat) if power == 1 else (flows - flows_hat)**2, dim=1, keepdim=True)
        flows_heatmaps = F.conv2d(flows_diff, kernel, stride=stride, padding=padding).numpy()
        flows_scores = np.array([np.max(heatmap) for heatmap in flows_heatmaps])

        # frame-scores according to max patches
        frames_diff = torch.sum(torch.abs(frames - frames_hat) if power == 1 else (frames - frames_hat)**2, dim=1, keepdim=True)
        frames_heatmaps = F.conv2d(frames_diff, kernel, stride=stride, padding=padding).numpy()
        frames_scores = np.array([np.max(frame_heatmap[flow_heatmap == np.max(flow_heatmap)])
                                  for (frame_heatmap, flow_heatmap) in zip(frames_heatmaps, flows_heatmaps)])

        return frames_scores, flows_scores

    # frame-level calculation for training/test sets
    def calc_raw_scores(self, epoch, part, patch_size, stride, power, force_calc=False):
        assert part in ("train", "test")
        out_path = self.output_store_path + '/out_epoch_%s/%s' % (str(epoch).zfill(LEN_ZFILL), part)
        self._create_path(out_path)
        out_file = os.path.join(out_path, 'scores.npy')
        # check whether file existed
        if os.path.exists(out_file) and not force_calc:
            scores = np.load(out_file, allow_pickle=True).item()
        else:
            scores = {}
        # check whether results already calculated
        key = "%d_%d_%d" % (patch_size, stride, power)
        if key in scores and not force_calc:
            return scores[key]
        # calculate new results
        dataset = DatasetDefiner(self.name, self.im_size, self.evaluation_store_path, mode=part)
        n_clip = dataset.get_n_clip(part)
        frames_scores, flows_scores, SSIM_scores = [], [], []

        for clip_idx in range(n_clip):
            dataset.load_data(clip_idx)

            # get input data
            frames = dataset.data[part][:][:, :3, :, :] / 127.5 - 1.
            flows = extend_flow_channel_in_batch(dataset.data[part][:][:, 3:, :, :])

            # get output results
            output_file = os.path.join(out_path, 'clip_%s.npy' % str(clip_idx + 1).zfill(len(str(n_clip))))
            output_data = np.load(output_file, allow_pickle=True).item()

            data = [frames, output_data["frames_hat"], flows, output_data["flows_hat"]]
            tmp_frames_scores, tmp_flows_scores = self._calc_score(data, patch_size, stride, power)
            frames_scores.append(tmp_frames_scores)
            flows_scores.append(tmp_flows_scores)
            if self.name in ("Belleview", "Train"):
                tmp_SSIM_scores = self._calc_score_SSIM(data)
                SSIM_scores.append(tmp_SSIM_scores)

        scores[key] = {"frame": frames_scores, "flow": flows_scores}
        if len(SSIM_scores) > 0:
            scores[key]["SSIM"] = SSIM_scores
        np.save(out_file, scores)
        return scores[key]

    # evaluation from frame-level groundtruth and (real eval data, output eval data)
    def evaluate(self, epoch, patch_size, stride, power, use_weight=True, force_calc=False):
        # load weights for summation of frame and flow scores
        if use_weight:
            training_scores = self.calc_raw_scores(epoch, "train", patch_size, stride, power, force_calc=force_calc)
            weights = (1./np.mean(np.concatenate(training_scores["frame"])),
                       1./np.mean(np.concatenate(training_scores["flow"])))
            print("Loaded weights:", weights)
        else:
            weights = (1., 1.)
        const_lambda = 0.2   # lambda in ICCV paper

        # load scores of test set
        test_scores = self.calc_raw_scores(epoch, "test", patch_size, stride, power, force_calc=force_calc)
        frames_scores, flows_scores = test_scores["frame"], test_scores["flow"]
        sum_scores = [const_lambda*np.log(weights[0]*frame_scores) + np.log(weights[1]*flow_scores)
                      for (frame_scores, flow_scores) in zip(frames_scores, flows_scores)]

        # return auc(s)
        dataset = DatasetDefiner(self.name, self.im_size, self.evaluation_store_path, mode="test")
        auc_frames_norm, aPR_frames_norm = dataset.evaluate(frames_scores, normalize_each_clip=True)
        auc_frames, aPR_frames = dataset.evaluate(frames_scores, normalize_each_clip=False)
        auc_flows_norm, aPR_flows_norm = dataset.evaluate(flows_scores, normalize_each_clip=True)
        auc_flows, aPR_flows = dataset.evaluate(flows_scores, normalize_each_clip=False)
        auc_sum_norm, aPR_sum_norm = dataset.evaluate(sum_scores, normalize_each_clip=True)
        auc_sum, aPR_sum = dataset.evaluate(sum_scores, normalize_each_clip=False)
        AUCs = [auc_frames_norm, auc_frames, auc_flows_norm, auc_flows, auc_sum_norm, auc_sum]
        aPRs = [aPR_frames_norm, aPR_frames, aPR_flows_norm, aPR_flows, aPR_sum_norm, aPR_sum]
        if "SSIM" in test_scores:
            auc_SSIM_norm, aPR_SSIM_norm = dataset.evaluate(test_scores["SSIM"], normalize_each_clip=True)
            auc_SSIM, aPR_SSIM = dataset.evaluate(test_scores["SSIM"], normalize_each_clip=False)
            print("SSIM: AUC = %.4f (norm), %.4f | aPR = %.4f (norm), %.4f" % (auc_SSIM_norm, auc_SSIM, aPR_SSIM_norm, aPR_SSIM))
        return AUCs, aPRs
