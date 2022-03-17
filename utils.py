import os
import logging
import numpy as np
import torch
import cmath
from sklearn import preprocessing


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)


def get_dset_path(dset_name, dset_type):
    _dir = os.path.dirname(__file__)
    # _dir = _dir.split("/")[:-1]
    # _dir = "/".join(_dir)
    return os.path.join(_dir, "datasets", dset_name, dset_type)


def int_tuple(s):
    return tuple(int(i) for i in s.split(","))


def l2_loss(pred_traj, pred_traj_gt, loss_mask, mode="average", norm_to_1=False):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - loss_mask: Tensor of shape (batch, seq_len)
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    seq_len, batch, _ = pred_traj.size()
    # equation below , the first part do noing, can be delete

    # calibrate face output: only keep array orientation, normalize length
    if norm_to_1:
        temp = torch.reshape(pred_traj, (-1, 2))
        temp = preprocessing.normalize(temp, norm='l2')
        pred_traj = torch.reshape(temp, (seq_len, batch, 2))

    loss = (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)) ** 2
    if mode == "sum":
        return torch.sum(loss)
    elif mode == "average":
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == "raw":
        return loss.sum(dim=2).sum(dim=1)


def l2_face_loss(pred_face, pred_face_gt, mode="average"):
    """
    Input:
    - pred_face: Tensor of shape (batch, 2). Predicted trajectory.
    - pred_face_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - loss_mask: Tensor of shape (batch, seq_len)
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    seq_len, batch, _ = pred_face_gt.size()
    # equation below , the first part do noing, can be delete

    # calibrate face output: only keep array orientation, normalize length
    avg_pred_face_gt = []
    for i in range(batch):
        angles = []
        for j in range(seq_len):
            cn = complex(pred_face_gt[j][i][0], pred_face_gt[j][i][1])
            _, angle = cmath.polar(cn)
            angles.append(angle)
        mean_angle = np.mean(angles)
        mean_cn = cmath.rect(1, mean_angle)
        x, y = round(mean_cn.real, 5), round(mean_cn.imag, 5)
        avg_pred_face_gt.append((x, y))
    avg_pred_face_gt = torch.tensor(avg_pred_face_gt).to(pred_face)
    loss = (avg_pred_face_gt - pred_face) ** 2
    if mode == "sum":
        return torch.sum(loss)
    elif mode == "raw":
        return loss.sum(dim=1)


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode="sum"):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory. [12, person_num, 2]
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """

    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)

    loss = loss ** 2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == "sum":
        return torch.sum(loss)
    elif mode == "mean":
        return torch.mean(loss)
    elif mode == "raw":
        return loss


def final_displacement_error(pred_pos, pred_pos_gt, consider_ped=None, mode="sum"):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (batch, 2). Groud truth
    last pos
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the eculidian displacement error
    """

    loss = pred_pos_gt - pred_pos
    loss = loss ** 2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == "raw":
        return loss
    else:
        return torch.sum(loss)


def calc_vector_inner_angle(v1, v2, trans_180=False):
    v1, v2 = np.array(v1, dtype=np.float), np.array(v2, dtype=np.float)
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    cos_ = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    sin_ = np.cross(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    arctan2_ = np.arctan2(sin_, cos_)
    return arctan2_ if not trans_180 else arctan2_ / np.pi * 180


def angle_difference(pred_face, pred_face_gt, trans_180=True, mode='sum'):
    seq_len, batch_size, _ = pred_face_gt.size()
    pred_face, pred_face_gt = pred_face.cpu(), pred_face_gt.cpu()
    angle_error = torch.zeros((batch_size, seq_len))
    for i in range(batch_size):
        for j in range(seq_len):
            # cn1 = complex(pred_face_gt[j][i][0], pred_face_gt[j][i][1])
            # _, angle_gt = cmath.polar(cn1)
            # cn2 = complex(pred_face[j][i][0], pred_face[j][i][1])
            # _, angle_pred = cmath.polar(cn2)

            # angle_error[i][j] = abs(angle_gt - angle_pred) / cmath.pi * 180 if trans_180 \
            #     else abs(angle_gt - angle_pred)
            inner_angle = calc_vector_inner_angle(pred_face_gt[j][i], pred_face[j][i], trans_180=trans_180)
            if not np.isnan(inner_angle):
                angle_error[i][j] = abs(inner_angle)

    if mode == 'sum':
        return torch.sum(angle_error.sum(dim=1))
    elif mode == 'raw':
        return angle_error.sum(dim=1)


def final_angle_difference(pred_face, pred_face_gt, trans_180=True, mode='sum'):
    batch_size, _ = pred_face_gt.size()
    pred_face, pred_face_gt = pred_face.cpu(), pred_face_gt.cpu()
    angle_error = torch.zeros((batch_size,))
    for i in range(batch_size):
        # cn1 = complex(pred_face_gt[i][0], pred_face_gt[i][1])
        # _, angle_gt = cmath.polar(cn1)
        # cn2 = complex(pred_face[i][0], pred_face[i][1])
        # _, angle_pred = cmath.polar(cn2)
        # angle_error[i] = abs(angle_gt - angle_pred) / cmath.pi * 180 if trans_180 \
        #     else abs(angle_gt - angle_pred)
        inner_angle = calc_vector_inner_angle(pred_face_gt[i], pred_face[i], trans_180=trans_180)
        if not np.isnan(inner_angle):
            angle_error[i] = abs(inner_angle)

    if mode == 'sum':
        return torch.sum(angle_error)
    elif mode == 'raw':
        return angle_error
