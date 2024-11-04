import os
import os.path as osp

import numpy as np
import torch
import pickle as pkl

from manopth.manolayer import ManoLayer
from chamferdist import ChamferDistance
from scipy.spatial.distance import cdist

from torch.optim.adam import Adam
from torch.optim import SGD
from tqdm import tqdm, trange

import plotly.graph_objects as go
import trimesh as tm
from utils.utils_plotly import plot_mesh, plot_point_cloud, plot_point_cloud_cmap, get_color_levels
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from itertools import combinations, chain
import argparse
import pickle
import fnmatch


# global paras
device = torch.device("cuda:0")

# load
user_id = 'a'
user_dir = f"data/scans/user_{user_id}"
num_scan_points = 16384 # how many points are sampled from original scan mesh
if_load_poses = False
init_poses_id = "01"
init_poses_dir = f'data/mano/mano_poses_v1_0/handsOnly_REGISTRATIONS_r_lm___POSES'

# mano
if_use_pca = False
num_comps = 36
num_reg_points = 778
mano_layer = ManoLayer(mano_root='data/mano/mano_v1_2/models', use_pca=if_use_pca, ncomps=num_comps).to(device)
comps = mano_layer.smpl_data['hands_components']

# optimize
num_steps = 30000
lr_beta = 1e-3
lr_poses = 1e-3
lr_trans = 1e-3
chamfer_dist = ChamferDistance()
i_chamfer_thresh = 5000
k_loss_thresh = 5000
s_loss_thresh = 0
i_chamfer_weight = 1
k_loss_weight = 2
s_loss_weight = 1
finger_indices = torch.tensor([4,8,12,16,20])

# visualize
visualize_step = 30000
write_mesh_step = 500
# required_scans_ids = {"01r","02r"}
required_scans_ids = {"01r","02r","ktest12","ktest13","ktest14","ktest15","r0"}
# visualized_scans_ids = {"ktest12","ktest13","ktest14","ktest15"}
visualized_scans_ids = required_scans_ids
log_dir = "log"
# run_time = datetime.now().strftime("%Y%m%d%H%M%S")
run_time = "20240717234347"
log_run_dir = f"{log_dir}/{run_time}"
writer = SummaryWriter(log_dir=log_run_dir)
if_write_to_results = True
results_file = "results/cross_validation.txt"
if_save_results = True
root_dir = f"results/saved_registered_poses_and_meshes/{user_id}"
dirs = [f"{root_dir}/state_of_art_beta/02r_ktest12_ktest13_ktest14_ktest15;01r",
        f"{root_dir}/registration_1104/02r;",
        f"{root_dir}/beta_regulation/02r;",
        f"{root_dir}/ncomps36/02r;",
        f"{root_dir}/ncomps12/02r;",
        f"{root_dir}/k12_both_regulation/ktest12;",
        f"{root_dir}/k12_beta_regulation/ktest12;",
        f"{root_dir}/with_glove",
        f"{root_dir}/glove_regulation"]
# dirs = [f"{root_dir}/registration_1104/02r;"]
faces = mano_layer.th_faces


class registered_hand:
    def __init__(self):
        self.beta = torch.zeros(10,dtype=torch.float32,device=device)
        self.pose = torch.zeros(48,dtype=torch.float32,device=device)
        self.tran = torch.zeros(3,dtype=torch.float32,device=device)
        self.mesh = 0
        self.scan = 0
        self.id = 0
    def __init__(self, beta, pose, tran, mesh, scan, id):
        self.beta = beta
        self.pose = pose
        self.tran = tran
        self.mesh = mesh
        self.scan = scan
        self.id = id

    def show_poses(self):
        print(f"Beta is {self.beta}.")
        print(f"Tran is {self.tran}.")
        # print(f"Pose is {self.pose}.")
        print(f"Thetas are {pose_to_thetas(self.pose)}")
        
        betas = torch.stack([self.beta],dim=0)
        poses = torch.stack([self.pose],dim=0)
        verts, joints = mano_layer(poses, betas)
        verts = verts.cpu().numpy()
        faces = mano_layer.th_faces.cpu().numpy()
        joints = joints.cpu().numpy()
        mesh_from_mano = tm.Trimesh(verts[0], faces=faces)
        go.Figure([plot_mesh(mesh_from_mano, color='lightpink', opacity=0.6, name=f"registered hand")]).show()


    def show_mesh(self):
        show_in_go(self.mesh,self.scan,self.id)


def load_hands_from_dir(dir,scans,scans_ids)->list[registered_hand]:
    hands = []
    beta = load_beta(f"{dir}/beta.pkl")
    for file_name in os.listdir(dir):
        scan_id = file_name.split('_')[0]
        if scan_id in scans_ids:
            mesh = tm.load(f"{dir}/{file_name}/mesh.stl")
            pose, tran = load_poses(f"{dir}/{file_name}/poses.pkl")
            i_scan = scans_ids.index(scan_id)
            hands.append(registered_hand(beta,pose,tran,mesh,scans[i_scan],scan_id))
    return hands


def load_scans():
    user_scans_files = os.listdir(user_dir)
    user_scans_files = [ p for p in user_scans_files if p.endswith("_preproc.stl") ]
    scans_ids = [ p.split("_")[1] for p in user_scans_files if p.split("_")[1] in required_scans_ids]
    num_scans = len(scans_ids)
    print(f"{num_scans} hand poses of user {user_id} are scanned.")
    print(f"They are {scans_ids}.")

    user_scans_meshes = [ tm.load(osp.join(user_dir, f)) for f in user_scans_files if f.split("_")[1] in scans_ids ]
    user_scans = torch.stack([ torch.tensor(m.sample(num_scan_points), dtype=torch.float32, device=device) for m in user_scans_meshes ], dim=0)
    return user_scans, num_scans, scans_ids
def load_beta(beta_dir):
    loaded = pkl.load(open(beta_dir,"rb"))
    beta = torch.tensor(loaded['beta'],dtype=torch.float32,device=device)
    return beta

def load_poses(poses_dir):
    loaded = pkl.load(open(poses_dir,"rb"))
    pose = torch.tensor(loaded['pose'],dtype=torch.float32,device=device)
    tran = torch.tensor(loaded['tran'],dtype=torch.float32,device=device)
    return pose, tran

def pose_to_thetas(pose):
    return pose.reshape(-1,3).norm(dim=1)

def show_in_go(mesh, scan_points, scan_id):
    fig = go.Figure([
        plot_mesh(mesh, color='lightpink', opacity=0.6, name=f"{scan_id}"),
        plot_point_cloud(scan_points, marker={"color": "red", "opacity": 0.8,"size": 1}, name=f"{scan_id}")
    ])
    fig.update_layout(scene_camera=dict(
        eye=dict(x=0,y=-1.5,z=0),
        center=dict(x=0,y=0,z=0),
        up=dict(x=0,y=0,z=1)
    ))
    invis = dict(
            showticklabels=False, 
            showgrid=False,       
            showline=False,        
            visible=False         
        )
    fig.update_layout(scene=dict(
        xaxis=invis,
        yaxis=invis,
        zaxis=invis
    ))
    fig.show()

if __name__ == "__main__":
    scans, num_scans, scans_ids = load_scans()
    scans = scans.cpu().detach().numpy()
    hands = []
    hands.append(load_hands_from_dir(dirs[0],scans,scans_ids)[1])
    for i in range(1,5):
        hand = load_hands_from_dir(dirs[i],scans,scans_ids)[0]
        hands.append(hand)
    hands[0].id = "old"
    hands[1].id = "both_regulated"
    hands[2].id = "beta_regulated"
    hands[3].id = "pca_36"
    hands[4].id = "pca_12"
    for i in range(len(hands)):
        hands[i].show_mesh()
    input()

    hands_k12 = []
    hands_k12.append(load_hands_from_dir(dirs[0],scans,scans_ids)[2])
    hands_k12.append(load_hands_from_dir(dirs[5],scans,scans_ids)[0])
    hands_k12.append(load_hands_from_dir(dirs[6],scans,scans_ids)[0])
    hands_k12[0].id = "old"
    hands_k12[1].id = "both_regulated"
    hands_k12[2].id = "beta_regulated"
    for i in range(len(hands_k12)):
        hands_k12[i].show_mesh()
    input()

    hands_glove = []
    hands_glove.append(load_hands_from_dir(dirs[7],scans,scans_ids)[0])
    hands_glove.append(load_hands_from_dir(dirs[8],scans,scans_ids)[0])
    hands_glove[0].id = "old"
    hands_glove[1].id = "regulated"
    for i in range(len(hands_glove)):
        hands_glove[i].show_mesh()