import logging
import os
import math
import json
import numpy as np

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def calc_seq_dist(anchor_seq, comp_seq):
    """
    seq: (seq_len, 2)
    """
    pt1 = torch.mean(anchor_seq, dim=0)
    pt2 = torch.mean(comp_seq, dim=0)
    return torch.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def make_init_adj(obs_traj, seq_start_end, keep=25):
    init_adjs = []
    for start, end in seq_start_end:
        adj = np.eye(end - start)
        trajs_temp = obs_traj[:, start:end, :].permute(1, 0, 2)
        for i in range(end - start):
            dists = []
            for j in range(end - start):
                dists.append(calc_seq_dist(trajs_temp[i], trajs_temp[j]))
            sort_index = sorted(range(len(dists)), key=lambda k: dists[k])
            if len(sort_index) > keep:
                adj[i][sort_index[:keep]] = 1
            else:
                adj[i][sort_index] = 1
        init_adjs.append(adj)
    return init_adjs


def seq_collate(data):
    (
        pids_list,
        obs_fids_list,
        pred_fids_list,
        obs_seq_list,
        pred_seq_list,
        obs_seq_rel_list,
        pred_seq_rel_list,
        non_linear_ped_list,
        loss_mask_list,
    ) = zip(*data)
    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [
        [start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])
    ]
    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    pids = torch.cat(pids_list, dim=0)
    obs_fids = torch.cat(obs_fids_list, dim=0).permute(1, 0)
    pred_fids = torch.cat(pred_fids_list, dim=0).permute(1, 0)
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        pids,
        obs_fids,
        pred_fids,
        obs_traj,
        pred_traj,
        obs_traj_rel,
        pred_traj_rel,
        non_linear_ped,
        loss_mask,
        seq_start_end,
    ]

    return tuple(out)


def read_file(_path, delim="\t"):
    data = []
    if delim == "tab":
        delim = "\t"
    elif delim == "space":
        delim = " "
    with open(_path, "r") as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
            self,
            data_dir,
            obs_len=8,
            pred_len=12,
            skip=1,
            threshold=0.002,
            min_ped=1,
            delim="\t",
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a sequence
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        pid_list = []
        fid_list = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            if 'txt' not in path:
                continue
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            # make 1 sequence at each time stamp
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                # curr_seq_data is a 20 length sequence
                # put 20 length sequence together, each frame and each person
                curr_seq_data = np.concatenate(
                    frame_data[idx: idx + self.seq_len], axis=0
                )

                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])  # all person_id in current sequence
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))  # shape: (person_num, 2, seq_length)
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_person_ids = np.zeros(len(peds_in_curr_seq))
                curr_frame_ids = np.zeros((len(peds_in_curr_seq), self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []

                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]  # shape: (seq_length, 4)
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len or curr_ped_seq.shape[0] != self.seq_len:
                        continue
                    _idx = num_peds_considered
                    curr_frame_ids[_idx] = curr_ped_seq[:, 0]
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])  # shape: (2, seq_length)
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    # relative displacement: t+1 - t
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]

                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    curr_person_ids[_idx] = ped_id
                    # Linear vs Non-Linear Trajectory
                    # _non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                # at least #min_ped person
                if num_peds_considered > min_ped:
                    # non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])  # each ele append: ((n_person, 2, seq_len)
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    pid_list.append(curr_person_ids[:num_peds_considered])
                    fid_list.append(curr_frame_ids[:num_peds_considered])

        self.num_seq = len(seq_list)
        pid_list = np.concatenate(pid_list, axis=0)
        fid_list = np.concatenate(fid_list, axis=0)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.pids = torch.from_numpy(pid_list).type(
            torch.int
        )
        self.obs_fids = torch.from_numpy(fid_list[:, : self.obs_len]).type(
            torch.int
        )
        self.pred_fids = torch.from_numpy(fid_list[:, self.obs_len:]).type(
            torch.int
        )
        self.obs_traj = torch.from_numpy(seq_list[:, :, : self.obs_len]).type(
            torch.float
        )
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(
            torch.float
        )
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, : self.obs_len]).type(
            torch.float
        )
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len:]).type(
            torch.float
        )
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.pids[start:end],  # shape: (n_persons)
            self.obs_fids[start:end, :],  # shape: (n_persons, seq_len_obs)
            self.pred_fids[start:end, :],  # shape: (n_persons, seq_len_obs)
            self.obs_traj[start:end, :],  # shape: (n_persons, 2, seq_len_obs)
            self.pred_traj[start:end, :],  # shape: (n_persons, 2, seq_len_pred)
            self.obs_traj_rel[start:end, :],  # shape: (n_persons, 2, seq_len_obs)
            self.pred_traj_rel[start:end, :],  # shape: (n_persons, 2, seq_len_pred)
            self.non_linear_ped[start:end],  # shape: ()
            self.loss_mask[start:end, :],  # shape: (n_persons, 2, seq_len)
        ]
        return out


def seq_collate_panda(data):
    (
        obs_seq_list,
        pred_seq_list,
        obs_seq_rel_list,
        pred_seq_rel_list,
        obs_face_list,
        pred_face_list,
        obs_face_rel_list,
        pred_face_rel_list,
        loss_mask_list,
        neib_seq_list_rel_list,
        neib_seq_list_self_list,
        neib_face_list_abs_list,
        neib_face_list_rel_list,
        all_main_group_states,
        all_main_inte_states
    ) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.stack(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.stack(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.stack(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.stack(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    obs_face = torch.stack(obs_face_list, dim=0).permute(2, 0, 1)
    pred_face = torch.stack(pred_face_list, dim=0).permute(2, 0, 1)
    obs_face_rel = torch.stack(obs_face_rel_list, dim=0).permute(2, 0, 1)
    pred_face_rel = torch.stack(pred_face_rel_list, dim=0).permute(2, 0, 1)
    loss_mask = torch.stack(loss_mask_list, dim=0)
    out = [
        obs_traj,
        pred_traj,
        obs_traj_rel,
        pred_traj_rel,
        obs_face,
        pred_face,
        obs_face_rel,
        pred_face_rel,
        loss_mask,
        neib_seq_list_rel_list,
        neib_seq_list_self_list,
        neib_face_list_abs_list,
        neib_face_list_rel_list,
        all_main_group_states,
        all_main_inte_states
    ]
    return tuple(out)


class PANDADataset(Dataset):
    """Dataloader for the Trajectory datasets"""

    def __init__(
            self,
            mode,
            obs_len=9,
            pred_len=9,
            neib_dist_thres=10,
            specific_scene=None,
            specific_angle=None,
            half_data=False,
            only_neib_in_state=None
    ):
        """
        Args:
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - neib_dist_thres: Maximum distance to be considered as a neighbour
        - specific_scene: only use specific scene
        """
        super(PANDADataset, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len

        root_dir = './dataset/GIF_Dataset'
        main_traj = np.load(os.path.join(root_dir, 'poi_trajectory.npy'))
        neib_traj = np.load(os.path.join(root_dir, 'neighbour_trajectory.npy'))
        main_face = np.load(os.path.join(root_dir, 'poi_viusal_orientation.npy'))
        neib_face = np.load(os.path.join(root_dir, 'neighbour_visual_orientation.npy'))

        if mode in ['train', 'val', 'test']:
            with open(os.path.join(root_dir, 'info_dicts', 'info_dicts_{}.json'.format(mode)), 'r') as load_f:
                data_info = json.load(load_f)

        if specific_scene is None:
            all_idxes = sorted([int(t) for t in data_info.keys()])
        else:
            all_idxes = sorted([int(t) for t in data_info.keys() if specific_scene in data_info[t]['scene']])

        if specific_angle is not None:
            all_idxes = sorted([t for t in all_idxes if 'rotate_{}'.format(specific_angle)
                                in data_info[str(t)]['scene']])

        if half_data:
            all_idxes = all_idxes[:int(len(all_idxes) / 2)]

        seq_num = len(all_idxes)

        all_main_traj = main_traj[all_idxes]
        all_main_face = main_face[all_idxes]

        all_neib_traj = []
        all_neib_face = []
        all_main_group_states = []
        all_main_inte_states = []

        buff_traj_abs = np.zeros((seq_num, 2, self.seq_len))  # shape: (person_num, 2, seq_length)
        buff_traj_rel = np.zeros((seq_num, 2, self.seq_len))
        buff_face_abs = np.zeros((seq_num, 2, self.seq_len))
        buff_face_rel = np.zeros((seq_num, 2, self.seq_len))
        buff_loss_mask = np.zeros((seq_num, self.seq_len))
        buff_neib_traj_self = []  # calculate updated node feature
        buff_neib_traj_rel = []  # calculate attention
        buff_neib_face_abs = []  # observed absolute angle sequence
        buff_neib_face_rel = []  # angle sequence relative to main person

        # processing main seqs
        for ped_id in range(seq_num):
            curr_ped_traj_abs = all_main_traj[ped_id]  # shape: (seq_length, 2)
            curr_ped_traj_abs = np.around(curr_ped_traj_abs, decimals=4)
            curr_ped_traj_abs = np.transpose(curr_ped_traj_abs)  # shape: (2, seq_length)

            curr_ped_face_abs = all_main_face[ped_id]
            curr_ped_face_abs = np.around(curr_ped_face_abs, decimals=4)
            curr_ped_face_abs = np.transpose(curr_ped_face_abs)  # shape: (2, seq_length)

            # Make coordinates relative
            curr_ped_traj_rel = np.zeros(curr_ped_traj_abs.shape)
            # relative displacement: relative to the last temporal_point, t+1 - t
            curr_ped_traj_rel[:, 1:] = curr_ped_traj_abs[:, 1:] - curr_ped_traj_abs[:, :-1]

            curr_ped_face_rel = np.zeros(curr_ped_face_abs.shape)
            # relative displacement: relative to the last temporal_point, t+1 - t
            curr_ped_face_rel[:, 1:] = curr_ped_face_abs[:, 1:] - curr_ped_face_abs[:, :-1]

            buff_traj_abs[ped_id] = curr_ped_traj_abs
            buff_face_abs[ped_id] = curr_ped_face_abs
            buff_traj_rel[ped_id] = curr_ped_traj_rel
            buff_face_rel[ped_id] = curr_ped_face_rel
            buff_loss_mask[ped_id] = 1

        # processing group and interaction states
        for seq_idx in [str(t) for t in all_idxes]:
            valid_neib_ids = [int(t.split('-')[0]) for t in data_info[seq_idx]['neibs'].keys() if
                              data_info[seq_idx]['neibs'][t] < neib_dist_thres]
            curr_ped_neib_seqs = neib_traj[valid_neib_ids]
            curr_ped_neib_oris = neib_face[valid_neib_ids]
            all_neib_traj.append(curr_ped_neib_seqs)
            all_neib_face.append(curr_ped_neib_oris)

            # make in-same-group and in-interaction sequences
            in_group_states = np.zeros((self.obs_len, len(valid_neib_ids) + 1))
            in_interaction_states = np.zeros((self.obs_len, len(valid_neib_ids) + 1))
            in_group_states[:, 0] = 1
            in_interaction_states[:, 0] = 1

            for point_id, cur_point_neib_ids in enumerate(data_info[seq_idx]['group_neibs'][:self.obs_len]):
                for neib_id in cur_point_neib_ids:
                    if neib_id in valid_neib_ids:
                        in_group_states[point_id][valid_neib_ids.index(neib_id) + 1] = 1

            for point_id, cur_point_neib_ids in enumerate(data_info[seq_idx]['inte_neibs'][:self.obs_len]):
                for neib_id in cur_point_neib_ids:
                    if neib_id in valid_neib_ids:
                        in_interaction_states[point_id][valid_neib_ids.index(neib_id) + 1] = 1

            all_main_group_states.append(torch.from_numpy(in_group_states).type(torch.int))
            all_main_inte_states.append(torch.from_numpy(in_interaction_states).type(torch.int))

        # processing neib seqs
        for ped_id in range(seq_num):
            curr_ped_neib_traj_abs = all_neib_traj[ped_id]  # shape: (X, 18, 2)
            curr_ped_neib_face_abs = all_neib_face[ped_id]  # shape: (X, 18, 2)
            curr_ped_neib_traj_abs = np.around(curr_ped_neib_traj_abs, decimals=4)
            curr_ped_neib_face_abs = np.around(curr_ped_neib_face_abs, decimals=4)

            # relative location to main person
            curr_ped_neib_traj_rel = curr_ped_neib_traj_abs - all_main_traj[ped_id]
            curr_ped_neib_traj_rel = curr_ped_neib_traj_rel[:, :9]
            buff_neib_traj_rel.append(torch.from_numpy(curr_ped_neib_traj_rel).permute(1, 0, 2).type(torch.float))

            # relative location to themselves
            curr_ped_neib_traj_self = np.zeros(curr_ped_neib_traj_abs.shape)
            curr_ped_neib_traj_self[:, 1:] = curr_ped_neib_traj_abs[:, 1:] - curr_ped_neib_traj_abs[:, :-1]
            curr_ped_neib_traj_self = curr_ped_neib_traj_self[:, :9]
            buff_neib_traj_self.append(torch.from_numpy(curr_ped_neib_traj_self).permute(1, 0, 2).type(torch.float))

            # absolute face orientation
            buff_neib_face_abs.append(torch.from_numpy(curr_ped_neib_face_abs[:, :9]).permute(1, 0, 2).type(torch.float))

            # relative face orientation to main person (directly array subtraction)
            curr_ped_neib_face_rel = curr_ped_neib_face_abs - all_main_face[ped_id]
            curr_ped_neib_face_rel = curr_ped_neib_face_rel[:, :9]
            buff_neib_face_rel.append(torch.from_numpy(curr_ped_neib_face_rel).permute(1, 0, 2).type(torch.float))

        if only_neib_in_state is not None:
            # Load only neib in state {only_neib_in_state}
            if only_neib_in_state == 'group':
                states = all_main_group_states
            elif only_neib_in_state == 'inte':
                states = all_main_inte_states

            neib_traj_self_state = []
            neib_traj_rel_state = []
            neib_face_abs_state = []
            neib_face_rel_state = []
            for seq_id in range(seq_num):
                cur_state = states[seq_id][:, 1:]
                all_neib_id_of_interest = []
                for point_id in range(cur_state.shape[0]):
                    neib_id_of_interest = [i for i, t in enumerate(cur_state[point_id]) if t]
                    all_neib_id_of_interest.extend(neib_id_of_interest)
                all_neib_id_of_interest = list(set(all_neib_id_of_interest))
                neib_traj_self_state.append(buff_neib_traj_self[seq_id][:, all_neib_id_of_interest])
                neib_traj_rel_state.append(buff_neib_traj_rel[seq_id][:, all_neib_id_of_interest])
                neib_face_abs_state.append(buff_neib_face_abs[seq_id][:, all_neib_id_of_interest])
                neib_face_rel_state.append(buff_neib_face_rel[seq_id][:, all_neib_id_of_interest])
            self.neib_seq_list_rel = neib_traj_rel_state
            self.neib_seq_list_self = neib_traj_self_state
            self.neib_face_list_abs = neib_face_abs_state
            self.neib_face_list_rel = neib_face_rel_state
        else:
            self.neib_seq_list_rel = buff_neib_traj_rel
            self.neib_seq_list_self = buff_neib_traj_self
            self.neib_face_list_abs = buff_neib_face_abs
            self.neib_face_list_rel = buff_neib_face_rel

        self.seq_num = seq_num
        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(buff_traj_abs[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(buff_traj_abs[:, :, self.obs_len:]).type(torch.float)
        self.obs_face = torch.from_numpy(buff_face_abs[:, :, :self.obs_len]).type(torch.float)
        self.pred_face = torch.from_numpy(buff_face_abs[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(buff_traj_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(buff_traj_rel[:, :, self.obs_len:]).type(torch.float)
        self.obs_face_rel = torch.from_numpy(buff_face_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_face_rel = torch.from_numpy(buff_face_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(buff_loss_mask).type(torch.float)

        self.all_main_group_states = all_main_group_states
        self.all_main_inte_states = all_main_inte_states

    def __len__(self):
        return self.seq_num

    def __getitem__(self, index):
        out = [
            self.obs_traj[index],  # shape: (2, seq_len_obs)
            self.pred_traj[index],  # shape: (2, seq_len_pred)
            self.obs_traj_rel[index],  # shape: (2, seq_len_obs)
            self.pred_traj_rel[index],  # shape: (2, seq_len_pred)
            self.obs_face[index],  # shape: (2, seq_len_obs)
            self.pred_face[index],  # shape: (2, seq_len_pred)
            self.obs_face_rel[index],  # shape: (2, seq_len_obs)
            self.pred_face_rel[index],  # shape: (2, seq_len_pred)
            self.loss_mask[index],  # shape: (2, seq_len)
            self.neib_seq_list_rel[index],  # shape: (X, seq_len, 2)
            self.neib_seq_list_self[index],  # shape: (X, seq_len, 2)
            self.neib_face_list_abs[index],  # shape: (X, seq_len, 2)
            self.neib_face_list_rel[index],  # shape: (X, seq_len, 2)
            self.all_main_group_states[index],  # shape: (seq_len, X + 1)
            self.all_main_inte_states[index]  # shape: (seq_len, X + 1)
        ]
        return out
