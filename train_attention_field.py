import argparse
import logging
import os
import random
import shutil
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import sys

sys.path.append('..')
import utils as utils
from data.loader import data_loader_panda
from models import AttentionFieldGenerator
from utils import (
    l2_loss,
    angle_difference,
    relative_to_abs
)

# Argument parsing
parser = argparse.ArgumentParser()

# frequently modified
parser.add_argument("--obs_len", default=9, type=int)
parser.add_argument("--pred_len", default=9, type=int)
parser.add_argument("--batch_size", default=1024, type=int)
parser.add_argument("--num_epochs", default=400, type=int)
parser.add_argument("--lr", default=1e-3, type=float, metavar="LR", help="initial learning rate", dest="lr")
parser.add_argument("--neib_dist_thres", default=10, type=float)
parser.add_argument("--only_neib_in_state", type=str, default="inte")
parser.add_argument("--graph_mode", type=str, default="gat")
parser.add_argument('--use_traj', action='store_true', default=True)
parser.add_argument('--half_data', action='store_true', default=False)
parser.add_argument("--dataset_mode", type=str, default="")
parser.add_argument("--feat_coef", default=1, type=float)
parser.add_argument("--lstm_layers", default=1, type=int)

# other settings
parser.add_argument("--log_dir", default="./", help="Directory containing logging file")
parser.add_argument("--loader_num_workers", default=0, type=int)
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
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
parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
parser.add_argument("--model_save_name", default="best_attention", type=str)

best_ade = 100
debug_mode = False


def main(args):
    # fix all seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    logging.info("Initializing train dataset")
    train_dset, train_loader = data_loader_panda(args,
                                                 mode='{}_train'.format(args.dataset_mode) if
                                                 args.dataset_mode else 'train',
                                                 specific_angle=0,
                                                 only_neib_in_state=None if args.only_neib_in_state == ""
                                                 else args.only_neib_in_state)
    logging.info("Initializing val dataset")
    _, val_loader = data_loader_panda(args,
                                      mode='{}_val'.format(args.dataset_mode) if
                                      args.dataset_mode else 'val',
                                      specific_angle=0,
                                      only_neib_in_state=None if args.only_neib_in_state == ""
                                      else args.only_neib_in_state)

    if not debug_mode:
        writer = SummaryWriter()
    else:
        writer = None

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
    model.cuda()

    optimizer = optim.Adam([{"params": model.parameters()}], lr=args.lr)

    global best_ade

    for epoch in range(args.start_epoch, args.num_epochs + 1):
        if epoch < 100:
            training_step = 1
        elif epoch < 200:
            training_step = 2
        else:
            if epoch == 200:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = 2e-4
            if epoch == 300:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = 1e-4
            training_step = 3
        train(args, model, train_loader, optimizer, epoch, training_step, writer)
        validate(model, train_loader, epoch, writer, training_step=training_step, loader='train')
        ade = validate(model, val_loader, epoch, writer, training_step=training_step)
        if training_step == 3 and not debug_mode:
            is_best = ade < best_ade
            best_ade = min(ade, best_ade)
            save_checkpoint(args,
                            {
                                "epoch": epoch + 1,
                                "state_dict": model.state_dict(),
                                "best_ade": best_ade,
                                "optimizer": optimizer.state_dict(),
                            },
                            is_best,
                            f"./checkpoint/checkpoint{epoch}.pth.tar",
                            )
    if not debug_mode:
        writer.close()


def train(args, model, train_loader, optimizer, epoch, training_step, writer):
    losses_face = utils.AverageMeter("Loss_face", ":.6f")
    progress = utils.ProgressMeter(
        len(train_loader), [losses_face], prefix="Epoch: [{}]".format(epoch)
    )

    model.train()
    for batch_idx, batch in enumerate(train_loader):
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

        optimizer.zero_grad()
        loss = torch.zeros(1).to(pred_traj_gt)
        loss_mask = loss_mask[:, args.obs_len:]

        pred_ori_fake = model(obs_traj_rel,
                              obs_face_rel,
                              neib_seq_list_rel,
                              neib_seq_list_self,
                              neib_face_list_abs,
                              neib_face_list_rel,
                              training_step)

        l2_loss_face = l2_loss(pred_ori_fake, pred_face_rel, loss_mask, mode="raw")

        l2_loss_sum_face = torch.zeros(1).to(pred_face_rel)
        _l2_loss_face = torch.sum(l2_loss_face)
        l2_loss_sum_face += _l2_loss_face
        loss += l2_loss_sum_face[0]
        losses_face.update(l2_loss_sum_face[0], obs_face.shape[1])  # batch_size

        loss.backward()
        optimizer.step()
        if batch_idx % args.print_every == 0:
            progress.display(batch_idx)
    if not debug_mode:
        writer.add_scalar("train_loss_face", losses_face.avg, epoch)


def validate(model, val_loader, epoch, writer, training_step, loader='val'):
    angle_diff_outer = []
    total_face = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
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

            total_face += pred_face.size(1)

            pred_ori_fake_rel = model(obs_traj_rel,
                                      obs_face_rel,
                                      neib_seq_list_rel,
                                      neib_seq_list_self,
                                      neib_face_list_abs,
                                      neib_face_list_rel,
                                      training_step)

            pred_ori_fake_rel_predpart = pred_ori_fake_rel[-args.pred_len:]
            pred_ori_fake = relative_to_abs(pred_ori_fake_rel_predpart, obs_face[-1])

            angle_diff_sum = cal_angle_diff(pred_face, pred_ori_fake)
            angle_diff_outer.append(angle_diff_sum)

        angle_diff = sum(angle_diff_outer) / (total_face * args.pred_len)
        logging.info(" * {loader} Angle diff  {angle_diff:.3f}".format(loader=loader, angle_diff=angle_diff))
        if not debug_mode:
            writer.add_scalar("{}_angle_diff".format(loader), angle_diff, epoch)
    return angle_diff


def cal_angle_diff(pred_face_gt, pred_face_fake):
    angle_diff = angle_difference(pred_face_fake, pred_face_gt)
    return angle_diff


def save_checkpoint(args, state, is_best, filename="checkpoint.pth.tar"):
    if is_best:
        torch.save(state, filename)
        logging.info("-------------- lower ade ----------------")
        shutil.copyfile(filename, "{}.pth.tar".format(args.model_save_name))


if __name__ == "__main__":
    args = parser.parse_args()
    utils.set_logger(os.path.join(args.log_dir, "train.log"))
    checkpoint_dir = "./checkpoint"
    if os.path.exists(checkpoint_dir) is False:
        os.mkdir(checkpoint_dir)
    main(args)
