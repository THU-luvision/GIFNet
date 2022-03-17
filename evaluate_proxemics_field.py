import argparse
import logging
import os
import random
import numpy as np
import torch
from data.loader import data_loader_panda
from models import ProxemicsFieldGenerator
from utils import (
    displacement_error,
    final_displacement_error,
    int_tuple,
    relative_to_abs,
)

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--obs_len", default=9, type=int)
parser.add_argument("--pred_len", default=9, type=int)
parser.add_argument("--batch_size", default=1024, type=int)
parser.add_argument("--num_samples", default=20, type=int)
parser.add_argument("--loader_num_workers", default=0, type=int)
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--noise_dim", default=(16,), type=int_tuple)
parser.add_argument("--noise_type", default="gaussian")
parser.add_argument("--traj_lstm_input_size", type=int, default=2, help="traj_lstm_input_size")
parser.add_argument("--traj_lstm_hidden_size", default=32, type=int)
parser.add_argument("--heads", type=str, default="4,1", help="Heads in each layer, splitted with comma")
parser.add_argument("--hidden-units", type=str, default="16", help="Hidden units in each hidden layer, ',' split")
parser.add_argument("--graph_network_out_dims", type=int, default=32, help="dims of every node after GAT module")
parser.add_argument("--graph_lstm_hidden_size", default=32, type=int)
parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (1 - keep probability).")
parser.add_argument("--alpha", type=float, default=0.2, help="Alpha for the leaky_relu.")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
parser.add_argument("--print_every", default=10, type=int)
parser.add_argument("--use_gpu", default=1, type=int)
parser.add_argument("--gpu_num", default="0", type=str)
parser.add_argument("--neib_dist_thres", default=10, type=float)
parser.add_argument("--only_neib_in_state", type=str, default="inte")
parser.add_argument("--graph_mode", type=str, default="gat")
parser.add_argument('--use_face', action='store_true', default=True)
parser.add_argument('--half_data', action='store_true', default=False)
parser.add_argument("--dataset_mode", type=str, default="")
parser.add_argument("--feat_coef", default=1, type=float)

parser.add_argument(
    "--model-load-path",
    default="./dataset/GIF_Dataset/checkpoints/proxemics/GIFNet_proxemics.pth.tar",
    type=str,
    help="path to latest checkpoint",
)

parser.add_argument(
    "--save-path",
    default="./dataset/GIF_Dataset/predictions/proxemics/pred.npy",
    type=str,
    help="path to save output prediction",
)


def evaluate_helper(error, mode='avg'):
    error = torch.stack(error, dim=0)
    if mode == 'avg':
        error = torch.sum(error, dim=0) / error.shape[0]
    elif mode == 'min':
        error = torch.min(error, dim=0).values
    return error


def cal_ade_fde(pred_traj_gt, pred_traj_fake, mode='sum'):
    ade = displacement_error(pred_traj_fake, pred_traj_gt, mode=mode)
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode=mode)
    return ade, fde


def main(args):
    # fix all seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    logging.info("Initializing val dataset")
    _, val_loader = data_loader_panda(args,
                                      mode='test',
                                      specific_angle=0,
                                      only_neib_in_state=None if args.only_neib_in_state == ""
                                      else args.only_neib_in_state)

    # [32, 16, 32]
    n_units = (
            [args.traj_lstm_hidden_size]
            + [int(x) for x in args.hidden_units.strip().split(",")]
            + [args.graph_lstm_hidden_size]
    )
    # [4, 1]
    n_heads = [int(x) for x in args.heads.strip().split(",")]

    model = ProxemicsFieldGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        traj_lstm_input_size=args.traj_lstm_input_size,
        traj_lstm_hidden_size=args.traj_lstm_hidden_size,
        n_units=n_units,
        graph_network_out_dims=args.graph_network_out_dims,
        dropout=args.dropout,
        graph_lstm_hidden_size=args.graph_lstm_hidden_size,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        n_heads=n_heads,
        alpha=args.alpha,
        use_face=args.use_face,
        graph_mode=args.graph_mode,
        feat_concat_samp_coef=args.feat_coef
    )

    checkpoint = torch.load(args.model_load_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda()

    ade, fde, all_pred = validate(args, model, val_loader)
    print("ADE: {:.6f}, FDE: {:.6f}".format(ade, fde))
    all_pred = all_pred.transpose((2, 0, 1, 3))  # (n_sample, n_exp, len_pred, feat_dim)
    print(all_pred.shape)
    np.save(args.save_path, all_pred)


def validate(args, model, val_loader):
    all_z = np.load('./dataset/GIF_Dataset/evaluate_z_20.npy')
    ade_outer, fde_outer = [], []
    total_traj = 0
    model.eval()

    all_prediction = []
    all_ade_raw, all_fde_raw = [], []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            print('evaluating:\t{} / {}'.format(i + 1, len(val_loader)))
            batch = [[t.cuda() for t in tensor] if isinstance(tensor, list) else tensor.cuda() for tensor in batch]
            (
                obs_traj,
                pred_traj_gt,
                obs_traj_rel,
                pred_traj_gt_rel,
                obs_face,
                pred_face,
                obs_face_rel,
                pred_face_rel,
                loss_mask,
                neib_seq_list_rel,
                neib_seq_list_self,
                neib_face_list_abs,
                neib_face_list_rel,
                group_states,
                inte_states
            ) = batch

            ade, fde = [], []
            ade_raw, fde_raw = [], []
            total_traj += pred_traj_gt.size(1)

            cur_batch_pred = []
            for exp_time in range(args.num_samples):
                pred_traj_fake_rel = model(obs_traj_rel,
                                           obs_traj,
                                           obs_face,
                                           neib_seq_list_rel,
                                           neib_seq_list_self,
                                           neib_face_list_abs,
                                           neib_face_list_rel,
                                           training_step=3,
                                           decoder_z=torch.from_numpy(all_z[exp_time]))

                pred_traj_fake_rel_predpart = pred_traj_fake_rel[-args.pred_len:]
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel_predpart, obs_traj[-1])
                cur_batch_pred.append(pred_traj_fake.cpu().numpy())
                ade_, fde_ = cal_ade_fde(pred_traj_gt, pred_traj_fake)
                ade.append(ade_)
                fde.append(fde_)
                raw_ade_, raw_fde_ = cal_ade_fde(pred_traj_gt, pred_traj_fake, mode='raw')
                ade_raw.append(raw_ade_.cpu().numpy())
                fde_raw.append(raw_fde_.cpu().numpy())

            ade_sum = evaluate_helper(ade)
            fde_sum = evaluate_helper(fde)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)

            all_ade_raw.append(ade_raw)
            all_fde_raw.append(fde_raw)

            all_prediction.append(np.stack(cur_batch_pred, axis=0))

    print('===============================================')
    ade = sum(ade_outer) / (total_traj * args.pred_len)
    fde = sum(fde_outer) / total_traj
    all_prediction = np.concatenate(all_prediction, axis=2)
    print('total traj num: ', total_traj)
    return ade, fde, all_prediction


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
