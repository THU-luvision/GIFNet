import cmath
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import seaborn as sn


def draw_proxemics_field(gt_traj, pred_traj, save_name=None, save_root='./'):
    """
        gt_traj: (18, 2) or (N, 18, 2)
        pred_traj: (20, 9, 2) or (N, 20, 9, 2)
    """
    if len(gt_traj.shape) == 2:
        gt_traj = gt_traj[np.newaxis, :]
        pred_traj = pred_traj[np.newaxis, :]

    min_x = min(np.min(gt_traj[:, :, 0]), np.min(pred_traj[:, :, :, 0]))
    max_x = max(np.max(gt_traj[:, :, 0]), np.max(pred_traj[:, :, :, 0]))
    min_y = min(np.min(gt_traj[:, :, 1]), np.min(pred_traj[:, :, :, 1]))
    max_y = max(np.max(gt_traj[:, :, 1]), np.max(pred_traj[:, :, :, 1]))
    if max_x - min_x > max_y - min_y:
        side_length = max_x - min_x
        fig_min_x = min_x - 0.5
        fig_max_x = max_x + 0.5
        fig_min_y = (min_y + max_y) / 2 - side_length / 2 - 0.5
        fig_max_y = (min_y + max_y) / 2 + side_length / 2 + 0.5
    else:
        side_length = max_y - min_y
        fig_min_y = min_y - 0.5
        fig_max_y = max_y + 0.5
        fig_min_x = (min_x + max_x) / 2 - side_length / 2 - 0.5
        fig_max_x = (min_x + max_x) / 2 + side_length / 2 + 0.5

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.grid(False)
    num_p, _, seq_len, _ = pred_traj.shape
    for pid in range(num_p):
        ax.plot(gt_traj[pid, :9, 0],
                gt_traj[pid, :9, 1],
                marker='.',
                linestyle='None',
                color='navy',
                markersize=4)
        colors = ['#ffb549'] * 9
        for tid in reversed(list(range(seq_len))):
            x = pred_traj[pid, :, tid, 0].ravel()
            y = pred_traj[pid, :, tid, 1].ravel()
            sn.kdeplot(x=x,
                       y=y,
                       shade=True,
                       ax=ax,
                       levels=20,
                       cmap=sn.light_palette(colors[tid], as_cmap=True))

    plt.xlim(fig_min_x, fig_max_x)
    plt.ylim(fig_min_y, fig_max_y)
    if save_name:
        fig.savefig('{}/{}.png'.format(save_root, save_name), dpi=300)
    else:
        plt.show()
    plt.close()


def draw_attn_field_with_traj(gt_traj, face_gt, face_pred, radius=0.5, angle_range=30,
                              show_gt=True, save_name=None, save_root='./'):
    """
    traj_gt: [N, 18, 2]
    face_gt: [N, 18, 2]
    face_pred: [N, 9, 2]
    """
    if len(gt_traj.shape) == 2:
        gt_traj = gt_traj[np.newaxis, :]
        face_gt = face_gt[np.newaxis, :]
        face_pred = face_pred[np.newaxis, :]

    min_x = np.min(gt_traj[:, :, 0])
    max_x = np.max(gt_traj[:, :, 0])
    min_y = np.min(gt_traj[:, :, 1])
    max_y = np.max(gt_traj[:, :, 1])
    if max_x - min_x > max_y - min_y:
        side_length = max_x - min_x
        fig_min_x = min_x - 1
        fig_max_x = max_x + 1
        fig_min_y = (min_y + max_y) / 2 - side_length / 2 - 1
        fig_max_y = (min_y + max_y) / 2 + side_length / 2 + 1
    else:
        side_length = max_y - min_y
        fig_min_y = min_y - 1
        fig_max_y = max_y + 1
        fig_min_x = (min_x + max_x) / 2 - side_length / 2 - 1
        fig_max_x = (min_x + max_x) / 2 + side_length / 2 + 1

    fig, ax = plt.subplots(nrows=1,
                           ncols=1,
                           figsize=(8, 8))

    ax.grid(False)
    num_p, seq_len, _ = gt_traj.shape
    polygons = []
    for pid in range(num_p):
        ax.plot(gt_traj[pid, :9, 0],
                gt_traj[pid, :9, 1],
                marker='.',
                linestyle='None',
                color='navy',
                markersize=6)
        ax.plot(gt_traj[pid, 9:, 0],
                gt_traj[pid, 9:, 1],
                marker='.',
                linestyle='None',
                color='orange',
                markersize=6)

        for tid in range(9):
            ori = face_gt[pid, tid]
            cn = complex(ori[0], ori[1])
            _, angle = cmath.polar(cn)

            base_x, base_y = gt_traj[pid, tid, 0], gt_traj[pid, tid, 1]
            polygons.append(Arc(xy=(base_x, base_y),  # 椭圆中心，（圆弧是椭圆的一部分而已）
                                width=radius,  # 长半轴
                                height=radius,  # 短半轴
                                theta1=angle / cmath.pi * 180 - angle_range / 2,  # 圆弧起点处角度
                                theta2=angle / cmath.pi * 180 + angle_range / 2,  # 圆弧终点处角度
                                fc='w',  # 填充色
                                ec='navy'  # 边框颜色
                                ))
            for cur_angle in [angle - angle_range / 360 * cmath.pi, angle + angle_range / 360 * cmath.pi]:
                cn = cmath.rect(radius / 2, cur_angle)
                edge_x = cn.real + base_x
                edge_y = cn.imag + base_y
                ax.plot([base_x, edge_x], [base_y, edge_y], c='navy')

        for tid in range(9):
            base_x, base_y = gt_traj[pid, tid + 9, 0], gt_traj[pid, tid + 9, 1]
            if show_gt:
                ori = face_gt[pid, tid + 9]
            else:
                ori = face_pred[pid, tid]

            cn = complex(ori[0], ori[1])
            _, angle = cmath.polar(cn)
            polygons.append(Arc(xy=(base_x, base_y),  # 椭圆中心，（圆弧是椭圆的一部分而已）
                                width=radius,  # 长半轴
                                height=radius,  # 短半轴
                                theta1=angle / cmath.pi * 180 - angle_range / 2,  # 圆弧起点处角度
                                theta2=angle / cmath.pi * 180 + angle_range / 2,  # 圆弧终点处角度
                                fc='w',  # 填充色
                                ec='orange'  # 边框颜色
                                ))
            for cur_angle in [angle - angle_range / 360 * cmath.pi, angle + angle_range / 360 * cmath.pi]:
                cn = cmath.rect(radius / 2, cur_angle)
                edge_x = cn.real + base_x
                edge_y = cn.imag + base_y
                ax.plot([base_x, edge_x], [base_y, edge_y], c='orange')

        for pln in polygons:
            ax.add_patch(pln)
    plt.xlim(fig_min_x, fig_max_x)
    plt.ylim(fig_min_y, fig_max_y)

    if save_name:
        fig.savefig('{}/{}.png'.format(save_root, save_name), dpi=300)
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    gt_traj = np.load('dataset/GIF_Dataset/predictions/proxemics/proxemics_gt.npy')
    pred_traj = np.load('dataset/GIF_Dataset/predictions/proxemics/proxemics_prediction.npy')
    gt_face = np.load('dataset/GIF_Dataset/predictions/attention/attention_gt.npy')
    pred_face = np.load('dataset/GIF_Dataset/predictions/attention/attention_prediction.npy')
    print(gt_traj.shape, pred_traj.shape, gt_face.shape, pred_face.shape)
    draw_proxemics_field(gt_traj[:2], pred_traj[:2])
    draw_attn_field_with_traj(gt_traj[:2], gt_face[:2], pred_face[:2])
