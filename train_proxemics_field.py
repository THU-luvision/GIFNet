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
from models import ProxemicsFieldGenerator
from utils import (
    displacement_error,
    final_displacement_error,
    int_tuple,
    l2_loss,
    relative_to_abs,
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
parser.add_argument('--use_face', action='store_true', default=True)
parser.add_argument('--half_data', action='store_true', default=False)
parser.add_argument("--dataset_mode", type=str, default="")
parser.add_argument("--feat_coef", default=1, type=float)

# other settings
parser.add_argument("--log_dir", default="./", help="Directory containing logging file")
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
parser.add_argument("--best_k", default=5, type=int)
parser.add_argument("--eval_num_samples", default=5, type=int)
parser.add_argument("--print_every", default=10, type=int)
parser.add_argument("--use_gpu", default=1, type=int)
parser.add_argument("--gpu_num", default="0", type=str)
parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
parser.add_argument("--model_save_name", default="best_proxemics", type=str)

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
        ade = validate(args, model, val_loader, epoch, writer, training_step=training_step)
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
    losses_traj = utils.AverageMeter("Loss_traj", ":.6f")
    progress = utils.ProgressMeter(
        len(train_loader), [losses_traj], prefix="Epoch: [{}]".format(epoch)
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
        l2_loss_rel = []
        loss_mask = loss_mask[:, args.obs_len:]

        if training_step == 1 or training_step == 2:
            pred_traj_fake_rel = model(obs_traj_rel,
                                       obs_traj,
                                       obs_face,
                                       neib_seq_list_rel,
                                       neib_seq_list_self,
                                       neib_face_list_abs,
                                       neib_face_list_rel,
                                       training_step)
            l2_loss_rel.append(
                l2_loss(pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode="raw")
            )
        else:
            model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)
            for _ in range(args.best_k):
                pred_traj_fake_rel = model(model_input,
                                           obs_traj,
                                           obs_face,
                                           neib_seq_list_rel,
                                           neib_seq_list_self,
                                           neib_face_list_abs,
                                           neib_face_list_rel,
                                           training_step,
                                           teacher_forcing_ratio=0.5)
                l2_loss_rel.append(l2_loss(pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode="raw"))

        l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
        _l2_loss_rel = torch.stack(l2_loss_rel, dim=1)
        _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)  # [20]
        _l2_loss_rel = torch.min(_l2_loss_rel) / (pred_traj_fake_rel.shape[0] * pred_traj_fake_rel.shape[1])
        l2_loss_sum_rel += _l2_loss_rel
        loss += l2_loss_sum_rel
        losses_traj.update(l2_loss_sum_rel[0], obs_traj.shape[1])  # batch_size

        loss.backward()
        optimizer.step()
        if batch_idx % args.print_every == 0:
            progress.display(batch_idx)
    if not debug_mode:
        writer.add_scalar("train_loss_traj", losses_traj.avg, epoch)


def validate(args, model, val_loader, epoch, writer, training_step):
    ade_outer, fde_outer = [], []
    total_traj = 0
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

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            for exp_time in range(args.eval_num_samples):
                pred_traj_fake_rel = model(obs_traj_rel,
                                           obs_traj,
                                           obs_face,
                                           neib_seq_list_rel,
                                           neib_seq_list_self,
                                           neib_face_list_abs,
                                           neib_face_list_rel,
                                           training_step)

                pred_traj_fake_rel_predpart = pred_traj_fake_rel[-args.pred_len:]
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel_predpart, obs_traj[-1])
                ade_, fde_ = cal_ade_fde(pred_traj_gt, pred_traj_fake)
                ade.append(ade_)
                fde.append(fde_)
            ade_sum = evaluate_helper(ade)
            fde_sum = evaluate_helper(fde)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)

        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / total_traj
        logging.info(" * ADE  {ade:.3f} FDE  {fde:.3f}".format(ade=ade, fde=fde))
        if not debug_mode:
            writer.add_scalar("val_ade", ade, epoch)
            writer.add_scalar("val_fde", fde, epoch)
    return ade


def evaluate_helper(error, mode='avg'):
    error = torch.stack(error, dim=0)
    if mode == 'avg':
        error = torch.sum(error, dim=0) / error.shape[0]
    elif mode == 'min':
        error = torch.min(error, dim=0).values
    return error


def cal_ade_fde(pred_traj_gt, pred_traj_fake):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    return ade, fde


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
