import json
import matplotlib.pyplot as plt
import numpy as np
import torch

from smplx import SMPL
import torch
from utils import *
import tensorboardX
import os
import drawsvg as draw
import cv2

def get_data(path = 'graphics.txt'):
    with open(path, 'r') as f:
        data = f.readlines()

    data = [json.loads(i) for i in data]
    # data = [elm for elm in data if elm['stroke_order'][0] is not None]
    data = {data[i]['character']: data[i] for i in range(len(data))}

    return data

def sample_points_on_lines(lines, alpha=0.5):
    points = []
    for line in lines:
        start = line[0]
        end = line[1]
        norm = torch.norm(end - start)
        num_points = int(norm  * alpha)

        vector = (end - start)[:,None]
        steps = torch.linspace(0, 1, num_points + 2)[1:-1][None,:].to(vector.device)
        _points = (start[:,None] + vector * steps).T
        if len(_points) > 0:
            points.append(_points)
    
    if len(points) == 0:
        return torch.tensor([]).to(lines.device)
    return torch.cat(points, dim=0)

class SMPL_Wrapper(object):
    def __init__(self, model_path, device):
        self.device = device
        self.model_path = model_path
        self.body_model = SMPL(model_path=model_path, device=device, use_hands=False, use_feet_keypoints=False).to(device)
        self.zero_pose = self.body_model.body_pose.clone()
        self.zero_pose[0, 13 * 3 + 2] = -torch.pi / 3
        self.zero_pose[0, 12 * 3 + 2] = torch.pi / 3
        self.zero_rot = self.body_model.global_orient.clone()
        self.transl = self.body_model.transl.clone()
        self.transl[0,0] = -2 # Bias by default to be further away
        self.HEAD_IDX = 12
        self.parents = self.body_model.parents.clone().to(self.device)
        # self.parents = torch.cat([self.parents, torch.tensor([self.HEAD_IDX]).to(self.device)])
        self.faces = torch.from_numpy(self.body_model.faces[None].astype(int)).to(self.device)

    def __call__(self, camera, body_pose=None, global_orient=None, transl=None, alpha=0.1):
        if body_pose is None:
            body_pose = self.zero_pose
        if global_orient is None:
            global_orient = self.zero_rot
        if transl is None:
            transl = self.transl
        out = self.body_model(body_pose=body_pose, global_orient=global_orient, transl=transl)

        vertices = out.vertices[None]
        joints = out.joints[0, :24]

        points = (camera.transform_points_screen(joints)[:, :2] - 0.5)
        lines = torch.stack([points[self.parents[1:]], points[1:]], dim=-2)
        extra_points = sample_points_on_lines(lines, alpha=alpha)

        return vertices, self.faces, points, extra_points, lines, joints


    def get_chamfer_loss(self, pcd1, pcd2):
        xyz1 = pcd1[:, None, :]
        xyz2 = pcd2[None, :, :]
        D_ij = ((xyz1 - xyz2) ** 2).sum(-1)

        nn_i1 = D_ij.argmin(dim=1)
        nn_distance1 = ((pcd1[:, :] - pcd2[nn_i1, :])**2).sum(-1)
        nn_i2 = D_ij.argmin(dim=0)
        nn_distance2 = ((pcd2[:, :] - pcd1[nn_i2, :])**2).sum(-1)
        loss = nn_distance1.mean() + nn_distance2.mean()

        return loss


    def optimize_pose(self, camera, all_stroke_points, lr=2e-2, weights=[1, 2e+1, 1e+2], closest=0.8, alpha=0.1, iters=1000, log_iter=10, renderer=None, tensorboard_path=None, verbose=True):
        pose_opt = torch.nn.Parameter(self.zero_pose.clone())
        rot_opt = torch.nn.Parameter(self.zero_rot.clone())
        transl_opt = torch.nn.Parameter(self.transl.clone())

        with torch.no_grad():
            transl_opt += torch.randn_like(transl_opt).mul_(0.3)
            # rot_opt += torch.randn_like(rot_opt).mul_(0.05)

        zero_pose = self.zero_pose.clone().detach()

        optimizer = torch.optim.Adam([pose_opt, rot_opt, transl_opt], lr=lr)

        if tensorboard_path is not None:
            writer = tensorboardX.SummaryWriter(tensorboard_path)
            os.makedirs(tensorboard_path, exist_ok=True)
            os.chmod(tensorboard_path, 0o755)

        for iter in range(iters):
            vertices, _, points, extra_points, lines, joints = self(camera, pose_opt, rot_opt, transl_opt, alpha=alpha)
            vertices.detach()
            all_points = torch.cat([points, extra_points], dim=0)

            chamfer = weights[0] * self.get_chamfer_loss(all_points, all_stroke_points)
            l1 = weights[1] * (zero_pose - pose_opt).pow(2).mean()
            dist = (joints[:,0] - closest).clip(0)
            closeness = weights[2] * (dist.sum() + dist.pow(2).sum()) / 2

            loss = chamfer + l1 + closeness
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (iter % log_iter) == 0 and verbose:
                print(iter, l1.item(), closeness.item(), chamfer.item())
                if tensorboard_path is not None:
                    with torch.no_grad():
                        # if iter % (plot_iter * 10) == 0:
                        #     plot_img = renderer.render_experiment_image(vertices, faces, camera)[0,0]
                        # else:
                        plot_img = torch.ones(*img_size, 4).cpu().numpy()
                        plot_img = renderer.draw_lines_on_img(plot_img, lines, color=[1, 0, 0, 1])
                        plot_img = renderer.draw_points_on_img(plot_img, points, color=[1, 0, 0, 1], radius=3)
                        plot_img = renderer.draw_points_on_img(plot_img, extra_points, color=[1, 0, 0, 1], radius=1)

                        plot_img = renderer.draw_lines_on_img(plot_img, stroke_lines, color=[0, 1, 0, 1])
                        plot_img = renderer.draw_points_on_img(plot_img, stroke_points, color=[0, 1, 0, 1], radius=3)
                        plot_img = renderer.draw_points_on_img(plot_img, extra_stroke_points, color=[0, 1, 0, 1], radius=1)
                        
                        plot_img = torch.tensor(plot_img).permute(2, 0, 1)
                        writer.add_image('fitting', plot_img, iter)
        return pose_opt, rot_opt, transl_opt

class Stroke_Wrapper(object):
    def __init__(self, graphics_path):
        self.data = get_data(graphics_path)
        _min = np.array([[1000, 1000]])
        _max = np.array([[-1000, -1000]])
        for k, v in self.data.items():
            for stroke in v['medians']:
                stroke = np.array(stroke)
                _min = np.min(np.concatenate([_min, np.min(stroke, axis=0)[None]], axis=0), axis=0)[None]
                _max = np.max(np.concatenate([_max, np.max(stroke, axis=0)[None]], axis=0), axis=0)[None]
        self.min = _min
        self.max = _max

    def get_img(self, character, img_size=[512, 512]):
        sample = self.data[character]
        svg_path = sample['strokes']

        size = self.max - self.min
        d = draw.Drawing(
            size[0,0], size[0,1], 
            origin=(self.min[0,0],self.min[0,1]))

        for path in svg_path:
            d.append(draw.Path(path, fill='black', stroke='black'))

        d.save_png('temp_stroke.png')
        img = cv2.imread('temp_stroke.png', cv2.IMREAD_UNCHANGED)
        img = cv2.flip(img, 0)
        # resize image
        img = cv2.resize(img, tuple(img_size))

        os.remove('temp_stroke.png')

        return img


    def __call__(self, character, normalize=True, img_size=[512, 512], alpha=0.1):
        medians = self.data[character]['medians']
        num_strokes = len(medians)
        count = 0
        for stroke in medians:
            stroke = np.array(stroke)
            if normalize:
                stroke_points = (stroke - self.min) / (self.max - self.min)
                # flip y
                stroke_points[:, 1] = 1 - stroke_points[:, 1]
                stroke_points *= np.array(img_size)[None]
            stroke_points = torch.tensor(stroke_points).float()
            stroke_lines = torch.stack([stroke_points[:-1], stroke_points[1:]], dim=-2)
            extra_stroke_points = sample_points_on_lines(stroke_lines, alpha=alpha)
            count += 1
            yield (stroke_points, extra_stroke_points, stroke_lines, f'{count}/{num_strokes}')


def draw_stroke(stroke, title=None, new_figure=True):
    if new_figure:
        plt.figure()
    for i in range(len(stroke)-1):
        plt.plot([stroke[i][0], stroke[i+1][0]], [stroke[i][1], stroke[i+1][1]], 'k-')

def rotate(stroke, alpha):
    rotation_matrix = np.array([
        [np.cos(alpha), -np.sin(alpha)], 
        [np.sin(alpha), np.cos(alpha)]
    ])
    return np.dot(stroke, rotation_matrix)

def normalize_stroke_new(stroke, eps=1e-8, size=1):
    offset = stroke[0]
    stroke = stroke - offset

    end_point = stroke[-1]
    alpha = np.arctan(end_point[1] / (end_point[0] + eps))

    if end_point[0] < 0: # rotate such that end point is always right of start point
        alpha -= np.pi

    alpha -= np.pi/2
    stroke = rotate(stroke, alpha)

    # min max normalization
    _min = np.min(stroke)
    _max = np.max(stroke)

    stroke = (stroke - _min) / (_max - _min)

    stroke = stroke * size

    # center stroke
    x_min, y_min = np.min(stroke, axis=0)
    x_max, y_max = np.max(stroke, axis=0)

    center = np.array([(size - (x_max - x_min)) / 2, (size - (y_max - y_min)) / 2 ])
    stroke = stroke - stroke[0] + center

    return stroke, {'offset': offset, 'alpha': alpha, 'length': len(stroke)}

def normalize_stroke(stroke, eps=1e-8, do_rotation=True, do_scale=True):
    # translate such that first point is (0, 0)
    offset = stroke[0]
    stroke = stroke - offset

    # rotate such that last point has y = 0
    end_point = stroke[-1]
    alpha = np.arctan(end_point[1] / (end_point[0] + eps))
    if do_rotation:
        if end_point[0] < 0: # rotate such that end point is always right of start point
            alpha -= np.pi
        stroke = rotate(stroke, alpha)
    else:
        alpha = 0

    # scale such that stroke x is between 0 and 1
    scale = np.linalg.norm(stroke[-1])
    if do_scale:
        stroke = stroke / scale

    return stroke, {'offset': offset, 'alpha': alpha, 'scale': scale, 'length': len(stroke)}