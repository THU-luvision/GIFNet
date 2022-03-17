from torch.utils.data import DataLoader
from data.trajectories import TrajectoryDataset, PANDADataset
from data.trajectories import seq_collate, seq_collate_panda


def data_loader(args, path):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate,
        pin_memory=True)
    return dset, loader


def data_loader_panda(args, mode, specific_scene=None, specific_angle=None, only_neib_in_state=None):
    dset = PANDADataset(
        mode=mode,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        specific_scene=specific_scene,
        specific_angle=specific_angle,
        only_neib_in_state=only_neib_in_state,
        half_data=args.half_data,
        neib_dist_thres=args.neib_dist_thres)

    print('Loaded {} dataset length: '.format(mode), len(dset))

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate_panda,
        pin_memory=True)
    return dset, loader
