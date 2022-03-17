import argparse
import logging
import os
import random
import numpy as np
import torch
from data.loader import data_loader_panda
from models import AttentionFieldGenerator
from utils import (
    int_tuple,
    angle_difference,
    final_angle_difference,
    relative_to_abs
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
parser.add_argument("--face_lstm_input_size", type=int, default=2)
parser.add_argument("--face_lstm_hidden_size", default=32, type=int)
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
parser.add_argument('--use_traj', action='store_true', default=True)
parser.add_argument('--half_data', action='store_true', default=False)
parser.add_argument("--dataset_mode", type=str, default="")
parser.add_argument("--feat_coef", default=1, type=float)
parser.add_argument("--lstm_layers", default=1, type=int)

parser.add_argument(
    "--model-load-path",
    default="./dataset/GIF_Dataset/checkpoints/attention/GIFNet_attention.pth.tar",
    type=str,
    help="path to latest checkpoint",
)

parser.add_argument(
    "--save-path",
    default="./dataset/GIF_Dataset/predictions/attention/pred.npy",
    type=str,
    help="path to save output prediction",
)


def cal_angle_diff(pred_face_gt, pred_face_fake, mode='sum'):
    angle_diff = angle_difference(pred_face_fake, pred_face_gt, mode=mode)
    final_angle_diff = final_angle_difference(pred_face_fake[-1], pred_face_gt[-1], mode=mode)
    return angle_diff, final_angle_diff


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
            [args.face_lstm_hidden_size]
            + [int(x) for x in args.hidden_units.strip().split(",")]
            + [args.graph_lstm_hidden_size]
    )
    # [4, 1]
    n_heads = [int(x) for x in args.heads.strip().split(",")]

    model = AttentionFieldGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        face_lstm_input_size=args.face_lstm_input_size,
        face_lstm_hidden_size=args.face_lstm_hidden_size,
        n_units=n_units,
        graph_network_out_dims=args.graph_network_out_dims,
        dropout=args.dropout,
        graph_lstm_hidden_size=args.graph_lstm_hidden_size,
        graph_mode=args.graph_mode,
        use_traj=args.use_traj,
        n_heads=n_heads,
        alpha=args.alpha,
        feat_concat_samp_coef=args.feat_coef,
        lstm_layers=args.lstm_layers
    )

    checkpoint = torch.load(args.model_load_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda()

    ade, fde, all_pred = validate(args, model, val_loader)  # (n_sample, feat_dim)
    print("ADE: {:.6f}, FDE: {:.6f}".format(ade, fde), 'result shape: {}'.format(all_pred.shape))
    print(all_pred.shape)
    np.save(args.save_path, all_pred)


def validate(args, model, val_loader):
    total_angle_diff, total_final_angle_diff = 0, 0
    total_face = 0
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

            total_face += pred_traj_gt.size(1)

            pred_ori_fake_rel = model(obs_traj_rel,
                                      obs_face_rel,
                                      neib_seq_list_rel,
                                      neib_seq_list_self,
                                      neib_face_list_abs,
                                      neib_face_list_rel,
                                      training_step=3)

            pred_ori_fake_rel_predpart = pred_ori_fake_rel[-args.pred_len:]
            pred_ori_fake = relative_to_abs(pred_ori_fake_rel_predpart, obs_face[-1])
            ad_, fd_ = cal_angle_diff(pred_face, pred_ori_fake)
            raw_ad_, raw_fd_ = cal_angle_diff(pred_face, pred_ori_fake, mode='raw')

            all_ade_raw.append(raw_ad_.cpu().numpy())
            all_fde_raw.append(raw_fd_.cpu().numpy())

            total_angle_diff += ad_
            total_final_angle_diff += fd_
            all_prediction.append(pred_ori_fake.cpu().numpy().transpose(1, 0, 2))  # (1024, 9, 2)

    print('===============================================')
    angle_diff = total_angle_diff / total_face / args.pred_len
    final_angle_diff = total_final_angle_diff / total_face
    all_prediction = np.concatenate(all_prediction, axis=0)
    print('total face num: ', total_face)
    return angle_diff, final_angle_diff, all_prediction


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
