#!/usr/bin/env python
# coding: utf-8

import torch
import os
import numpy as np
import trimesh
from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.mesh.sphere import points_to_spheres
from body_visualizer.tools.vis_tools import show_image

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from os import path as osp

support_dir = '../support_data/'

# Choose the device to run the body model on.
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

amass_npz_fname = osp.join(support_dir, 'github_data/dmpl_sample.npz')
bdata = np.load(amass_npz_fname)

# Decode the subject_gender from bytes to string
subject_gender = bdata['gender'].decode('utf-8')

print('Data keys available:%s' % list(bdata.keys()))
print('The subject of the mocap sequence is  {}.'.format(subject_gender))

bm_fname = osp.join(support_dir, 'body_models/smplh/{}/model.npz'.format(subject_gender))
dmpl_fname = osp.join(support_dir, 'body_models/dmpls/{}/model.npz'.format(subject_gender))

num_betas = 16
num_dmpls = 8

bm = BodyModel(bm_fname, num_betas, num_dmpls, dmpl_fname).to(comp_device)
faces = c2c(bm.f)

time_length = len(bdata['trans'])

body_parms = {
    'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(comp_device),
    'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(comp_device),
    'pose_hand': torch.Tensor(bdata['poses'][:, 66:]).to(comp_device),
    'trans': torch.Tensor(bdata['trans']).to(comp_device),
    'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(comp_device),
    'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]).to(comp_device)
}

imw, imh = 1600, 1600
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)


def vis_body_pose_beta(fId=0):
    body_pose_beta = bm(**{k: v for k, v in body_parms.items() if k in ['pose_body', 'betas']})
    body_mesh = trimesh.Trimesh(vertices=c2c(body_pose_beta.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)


def vis_body_pose_hand(fId=0):
    body_pose_hand = bm(**{k: v for k, v in body_parms.items() if k in ['pose_body', 'betas', 'pose_hand']})
    body_mesh = trimesh.Trimesh(vertices=c2c(body_pose_hand.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)


def vis_body_joints(fId=0):
    joints = c2c(body_pose_hand.Jtr[fId])
    joints_mesh = points_to_spheres(joints, point_color=colors['red'], radius=0.005)
    mv.set_static_meshes([joints_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)


def vis_body_dmpls(fId=0):
    body_dmpls = bm(**{k: v for k, v in body_parms.items() if k in ['pose_body', 'betas', 'pose_hand', 'dmpls']})
    body_mesh = trimesh.Trimesh(vertices=c2c(body_dmpls.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)


def vis_body_trans_root(fId=0):
    body_trans_root = bm(**{k: v for k, v in body_parms.items() if k in ['pose_body', 'betas', 'pose_hand', 'dmpls',
                                                                       'trans', 'root_orient']})
    body_mesh = trimesh.Trimesh(vertices=c2c(body_trans_root.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)


def vis_body_transformed(fId=0):
    body_trans_root = bm(**{k: v for k, v in body_parms.items() if k in ['pose_body', 'betas', 'pose_hand', 'dmpls',
                                                                       'trans', 'root_orient']})
    body_mesh = trimesh.Trimesh(vertices=c2c(body_trans_root.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    body_mesh.apply_transform(trimesh.transformations.rotation_matrix(-90, (0, 0, 1)))
    body_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (1, 0, 0)))

    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)


if __name__ == "__main__":
    vis_body_pose_beta(fId=0)
    vis_body_pose_hand(fId=0)
    vis_body_joints(fId=0)
    vis_body_dmpls(fId=0)
    vis_body_trans_root(fId=0)
    vis_body_transformed(fId=0)
