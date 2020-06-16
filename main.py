import os
import argparse
import torch
import numpy as np

# from WGAN import WGAN_GP
# from NoGan import NoGAN
from DCGAN import DCGAN
from CONFIG import data_info

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# set init seed values
torch.manual_seed(2020)
np.random.seed(2020)


def print_params(params):
    print("dataset: %s | task: %s | epoch: %s" % (params["dataset"], params["task"], params["epoch"]))
    print("method: %s | workspace: %s" % (params["method"], params["workspace"]))
    print("height: %d | width: %d | batch: %d" % (params["height"], params["width"], params["batch"]))
    print("use RNN: %d (cat latent: %d) | use element norm: %d (gamma %.1f, sigmoid %d)| use channel norm: %d" %
          (params["RNN"], params["cat_latent"], params["elenorm"], params["training_gamma"], params["sigmoid_instead_tanh"], params["chanorm"]))
    print("skip extension for blocks: %s" % params["skip_blocks"])
    if params["task"] == "eval":
        print("power: %d | patch: %d | stride: %d" % (params["power"], params["patch"], params["stride"]))


def get_epoch_info(epoch_str):
    if '-' in epoch_str:
        vals = epoch_str.split('-')
        assert len(vals) == 2
        return int(vals[0]), int(vals[1])
    return 0, int(epoch_str)


# main execution
def run(params):
    # parse parameters
    dataset = params["dataset"]
    task = params["task"]
    method = params["method"]
    im_size = (params["height"], params["width"])
    store_path = params["workspace"]
    training_gamma = params["training_gamma"]
    use_progress_bar = params["progressbar"] != 0
    print_summary = params["prt_summary"] != 0
    # extension parameters
    extension_params = ["skip:%s" % params["skip_blocks"]]
    if params["RNN"] != 0:
        extension_params.append("RNN")
    if params["cat_latent"] != 0:
        extension_params.append("cat_latent")
    if params["elenorm"] != 0:
        extension_params.append("element_norm")
    if params["sigmoid_instead_tanh"] != 0:
        extension_params.append("sigmoid_instead_tanh")
    if params["chanorm"] != 0:
        extension_params.append("channel_norm")
    if params["relu_chanorm"] != 0:
        extension_params.append("relu_chanorm")
    # init model controller
    if method == "WGAN":
        # controller = WGAN_GP(dataset, im_size, store_path)
        print("To be implemented...")
        return
    elif method == "NoGAN":
        # controller = NoGAN(dataset, im_size, store_path)
        print("To be implemented...")
        return
    else:
        controller = DCGAN(dataset, im_size, store_path,
                           extension_params,
                           training_gamma=training_gamma,
                           use_progress_bar=use_progress_bar,
                           prt_summary=print_summary)
    #
    if task == "train":
        print("========== Mode: Training ==========")
        epoch_start, epoch_end = get_epoch_info(params["epoch"])
        print("Epoch %d to %d" % (epoch_start, epoch_end))
        batch_size = params["batch"]
        n_epoch_to_save = params["every"]
        controller.train(epoch_start, epoch_end, batch_size, n_epoch_to_save)
    elif task == "infer":
        print("========== Mode: Inference ==========")
        _, epoch_eval = get_epoch_info(params["epoch"])
        print("Epoch %d" % epoch_eval)
        batch_size = params["batch"]
        controller.infer(epoch_eval, batch_size, part=params["subset"])
    elif task == "eval":
        print("========== Mode: Evaluation ==========")
        _, epoch_eval = get_epoch_info(params["epoch"])
        power = params["power"]
        patch_size = params["patch"]
        stride = params["stride"]
        print("Epoch %d" % epoch_eval)
        AUCs, aPRs = controller.evaluate(epoch_eval, patch_size, stride, power)
        print("Epoch", epoch_eval)
        print("AUCs:", AUCs)
        print("aPRs:", aPRs)
    else:
        print("Unknown task", task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--RNN", type=int, default=1)
    parser.add_argument("--cat_latent", type=int, default=1)
    parser.add_argument("--elenorm", type=int, default=1)
    parser.add_argument("--sigmoid_instead_tanh", type=int, default=0)
    parser.add_argument("--training_gamma", type=float, default=-1)
    parser.add_argument("--chanorm", type=int, default=1)
    parser.add_argument("--relu_chanorm", type=int, default=1)
    parser.add_argument("--skip_blocks", type=str, default="none")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--subset", type=str, default="test")
    parser.add_argument("--epoch", type=str, default=None)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--progressbar", type=int, default=0)
    parser.add_argument("--method", type=str, default="DCGAN")
    parser.add_argument("--workspace", type=str, default="./workspace_ICCV_extend")
    parser.add_argument("--power", type=int, default=2)
    parser.add_argument("--patch", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--print", type=int, default=1)
    parser.add_argument("--prt_summary", type=int, default=0)
    args = vars(parser.parse_args())
    # validate arguments
    assert args["dataset"] in data_info
    assert args["task"] in ["train", "infer", "eval"]
    assert args["subset"] in ["train", "test"]
    assert args["batch"] > 1
    assert args["method"] in ["DCGAN", "WGAN", "NoGAN"]
    #
    if args["print"] != 0:
        print_params(args)
    run(args)
