import os
import os.path as osp

import numpy as np
import torch
import pickle as pkl

from manopth.manolayer import ManoLayer
from chamferdist import ChamferDistance
from scipy.spatial.distance import cdist

from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
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
import math


import matplotlib.pyplot as plt

# global paras
device = torch.device("cuda:0")
if_fixed_beta = False
seed = 100000
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# load
user_id = 'a'
user_dir = f"data/scans/user_{user_id}"
num_scan_points = 16384 # how many points are sampled from original scan mesh
if_load_poses = False
init_poses_id = "01"
init_poses_dir = f'data/mano/mano_poses_v1_0/handsOnly_REGISTRATIONS_r_lm___POSES'
beta_id = 'both_regulation_beta'
beta_dir = f'data/beta/user_{user_id}/{beta_id}.pkl'
if_load_aligned_scans = False
aligned_scans_dir = "data/aligned_scans"

# mano
if_use_pca = False
num_comps = 12
num_reg_points = 778
mano_layer = ManoLayer(mano_root='data/mano/mano_v1_2/models', use_pca=if_use_pca, ncomps=num_comps).to(device)

# optimize
num_steps = 20001
lr_beta = 1e-3
lr_poses = 1e-3
lr_trans = 1e-3
wd_beta = 0 # 1e-3
wd_poses = 0 # 1e-4
wd_trans = 0
regulation_weight_beta = 1e-3
regulation_weight_poses = 1e-4
chamfer_dist = ChamferDistance()
i_chamfer_thresh = 0
k_loss_thresh = 0
s_loss_thresh = 0
i_chamfer_weight = 0.1
k_loss_weight = 2
s_loss_weight = 1
finger_indices = torch.tensor([4,8,12,16,20])

# visualize
visualize_step = 5000
visualize_list = {0,10,20,100,500,1000,2000,2500,3000,5000,7500,10000,20000,30000}
write_mesh_step = 100
required_scans_ids = {"02r"}
# required_scans_ids = {"01r","02r","ktest12","ktest13","ktest14","ktest15","r0"}
visualized_scans_ids = required_scans_ids
log_dir = "log"
run_time = datetime.now().strftime("%Y%m%d%H%M%S")
# img_dir = f"results/images/{run_time}"
# os.makedirs(img_dir,exist_ok=True)
log_run_dir = f"{log_dir}/{run_time}"
loss_dir = f"results/loss_data/{run_time}.pkl"
writer = SummaryWriter(log_dir=log_run_dir)
write_thresh = 5001
if_write_to_results = True
results_file = "results/cross_validation.txt"
if_save_results = True
save_dir = f"results/saved_registered_poses_and_meshes/{user_id}/{run_time}"


# load funcs
def load_scans():
    if if_load_aligned_scans:
        user_scans_files = os.listdir(aligned_scans_dir)
        scans_ids = [ p.split(".")[0] for p in user_scans_files if p.split(".")[0] in required_scans_ids]
        num_scans = len(scans_ids)
        print(f"{num_scans} hand poses of user {user_id} are scanned and aligned.")
        print(f"They are {scans_ids}.")
        scans = [torch.tensor(pkl.load(open(osp.join(aligned_scans_dir,f),"rb"),encoding='iso-8859-1')) for f in user_scans_files]
        scans = torch.stack(scans,dim=0)
        scans = scans.to(dtype=torch.float32,device=device)
        scans = scans - scans.mean(dim=1).view(num_scans, -1).unsqueeze(1)
        return scans,num_scans,scans_ids

    user_scans_files = os.listdir(user_dir)
    user_scans_files = [ p for p in user_scans_files if p.endswith("_preproc.stl") ]
    scans_ids = [ p.split("_")[1] for p in user_scans_files if p.split("_")[1] in required_scans_ids]
    num_scans = len(scans_ids)
    print(f"{num_scans} hand poses of user {user_id} are scanned.")
    print(f"They are {scans_ids}.")

    user_scans_meshes = [ tm.load(osp.join(user_dir, f)) for f in user_scans_files if f.split("_")[1] in scans_ids ]
    user_scans = torch.stack([ torch.tensor(m.sample(num_scan_points), dtype=torch.float32, device=device) for m in user_scans_meshes ], dim=0)
    return user_scans, num_scans, scans_ids

def load_poses(scans, num_scans, scans_ids): # poses, beta(only one), trans
    if not if_load_poses: # initialize with zeros
        poses = torch.zeros((num_scans,48),dtype=torch.float32,device=device)
        beta = torch.zeros(10,dtype=torch.float32,device=device)

        # betas = beta.unsqueeze(0).expand(num_scans, 10)
        # verts, _ = mano_layer(poses, betas)
        # trans_padding = torch.tensor([-0.05,0,0],dtype=torch.float32,device=device)
        # trans_padding = trans_padding.unsqueeze(0).expand(num_scans, 3)
        # trans = scans.mean(dim=1) - verts.mean(dim=1) + trans_padding
        
        trans = torch.zeros((num_scans,3),dtype=torch.float32,device=device)
        return poses, beta, trans
    
    else:
        poses = []
        for id in scans_ids:
            loaded = pkl.load(open(osp.join(init_poses_dir, f"{init_poses_id}_{id}.pkl"), 'rb'), encoding='iso-8859-1')
            poses.append(torch.tensor(loaded['pose'], dtype=torch.float32, device=device))
        poses = torch.stack(poses, dim=0)
        poses[..., :3] = 0.0
        poses = mano_layer.pca_decomp(poses)

        beta = torch.zeros(10,dtype=torch.float32,device=device)
        trans = torch.zeros((num_scans,3),dtype=torch.float32,device=device)
        return poses, beta, trans


def load_mano_poses(pose_id):
    dir = osp.join(init_poses_dir, f"{init_poses_id}_{pose_id}.pkl")
    loaded = pkl.load(open(dir,"rb"),encoding='iso-8859-1')
    pose = torch.tensor(loaded["pose"],dtype=torch.float32, device=device)
    beta = torch.tensor(loaded["betas"],dtype=torch.float32, device=device)
    return pose,beta
# pose_prior_k12, _ = load_mano_poses("10l_mirrored")
# pose_prior_k12[:3] = 0.0

def load_beta(beta_dir=beta_dir):
    loaded = pkl.load(open(beta_dir,"rb"))
    beta = torch.tensor(loaded['beta'],dtype=torch.float32,device=device)
    return beta



def generate_hand_verts(poses,beta,trans): # all epoches
    betas = beta.unsqueeze(0).expand(num_scans, 10)
    verts, joints = mano_layer(poses, betas) # [num_scans,num_reg_points,3] [num_scans,21,3]
    verts = verts + trans.unsqueeze(1).tile([1,num_reg_points,1])
    joints = joints + trans.unsqueeze(1).tile([1,num_reg_points,1])
    return verts, joints

def duplex_chamfer_dist(verts, scans, step):
    loss = 0
    f_chamfer = chamfer_dist(verts, scans, batch_reduction=None) 
    i_chamfer = chamfer_dist(scans, verts, batch_reduction=None)
    loss = loss + f_chamfer

    if (step < i_chamfer_thresh):
        # loss = loss + i_chamfer * (1 - i * 1.0 / i_chamfer_thresh) * i_chamfer_weight
        # loss = loss + i_chamfer * i_chamfer_weight
        loss = loss + i_chamfer * (1 - (step * 1.0 / i_chamfer_thresh)**2) * i_chamfer_weight
    return loss, f_chamfer, i_chamfer

def k_finger_match_dist(joints, scans_ids, step):
    thumbs = []
    targets = []
    if ("ktest15" in  scans_ids and step < k_loss_thresh):
        i_kt= scans_ids.index("ktest15")
        thumbs.append(joints[i_kt][4])
        targets.append(joints[i_kt][20])
    if ("ktest14" in  scans_ids and step < k_loss_thresh):
        i_kt= scans_ids.index("ktest14")
        thumbs.append(joints[i_kt][4])
        targets.append(joints[i_kt][16])
    if (thumbs == []):
        return 0
    else:
        thumbs = torch.stack(thumbs,dim=0)
        targets = torch.stack(targets,dim=0)
        return torch.norm(thumbs - targets,dim=1).mean() * k_loss_weight
    
def scan_finger_match_dist(joints, scans_ids, step):
    if ("ktest15" in  scans_ids and step < s_loss_thresh):
        i_kt= scans_ids.index("ktest15")
        
        # little = joints[i_kt][20]
        # scan_little = torch.tensor([-0.049,-0.014,-0.029],dtype=torch.float32,device=device)
        reg_fingers = joints[i_kt][finger_indices]
        scan_fingers = torch.tensor([[-0.051,-0.019,-0.025],
                                     [-0.099,-0.017,-0.007],
                                     [-0.110,0.003,-0.032],
                                     [-0.095,-0.012,-0.044],
                                     [-0.048,-0.014,-0.030]],dtype=torch.float32,device=device)
        return torch.norm(reg_fingers-scan_fingers,dim=1).mean() * s_loss_weight
    else:
        return 0

# def regulation_loss(poses, scans_ids):
#     if ("ktest12" in  scans_ids):
#         i_kt= scans_ids.index("ktest12")
#         pose_predict = poses[i_kt][3:]
#         pose_prior = pose_prior_k12[3:]
#         return torch.sum((pose_predict-pose_prior)**2) * 1e-3
#     else:
#         return 0
    

def gauss_regulation_beta(beta): # [10]
    return torch.sum(beta**2) * regulation_weight_beta

def gauss_regulation_poses(poses): # [num_scans, 48]
    return torch.sum(poses**2, dim=1) * regulation_weight_poses
 
def visualize(reg_points, reg_faces, reg_joints, scan_points, scan_id, step):

    fig = go.Figure([
        plot_mesh(tm.Trimesh(reg_points, faces=reg_faces), color='lightpink', opacity=0.6, name=f"registered hand mesh step {step}"),
        plot_point_cloud(scan_points, marker={"color": "red", "opacity": 0.8,"size": 1}, name=f"scanned point cloud"),
        #plot_point_cloud(reg_joints, marker={"color": "blue", "size": 3}, name=f"joints"),
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
    # reg_points = reg_points[None]
    # reg_faces = reg_faces[None]
    # if (scan_id in visualized_scans_ids):
    #     writer.add_mesh(f"Registered poses/{scan_id}",vertices=reg_points,faces=reg_faces,global_step=step)
    #     writer.add_mesh()
    # fig.write_image(f"{img_dir}/step_{step}.svg")

def save_registered_hands(dir,beta,scans_ids,poses,trans,verts,if_write_all=False,tag="training"): # one register, one set of hands

    # betas = beta.unsqueeze(0).expand(num_scans, 10)
    # verts, _ = mano_layer(poses, betas) # [num_scans,num_reg_points,3] [num_scans,21,3]
    # verts = verts + trans.view(num_scans, -1).unsqueeze(1)
    faces = mano_layer.th_faces.cpu().numpy()
    for id,pose,tran,points in zip(scans_ids,poses,trans,verts):
        os.makedirs(f"{dir}/{id}_{tag}",exist_ok=True)
        saved_poses = {"pose":pose.cpu().detach().numpy(),
                       "tran":tran.cpu().detach().numpy()}
        with open(f"{dir}/{id}_{tag}/poses.pkl","wb") as file:
            pickle.dump(saved_poses,file)
        tm.Trimesh(points.cpu().detach().numpy(), faces=faces).export(f"{dir}/{id}_{tag}/mesh.stl",file_type="stl")
    if if_write_all:
        saved_beta = {"beta":beta.cpu().detach().numpy()}   
        with open(f"{dir}/beta.pkl","wb") as file:
            pickle.dump(saved_beta,file)

def save_args(num_steps=num_steps): # one run time, one set of args
    with open(f"{save_dir}/args.txt","w") as file:
        file.write(f"user_id = {user_id}\n")
        file.write(f"scans_ids = {required_scans_ids}\n")
        file.write(f"num_scan_points = {num_scan_points}\n")
        file.write(f"if_use_pca = {if_use_pca}\n")
        file.write(f"num_comps = {num_comps}\n")
        file.write(f"num_steps = {num_steps}\n")
        file.write(f"lr_beta = {lr_beta}\n")
        file.write(f"lr_poses = {lr_poses}\n")
        file.write(f"lr_trans = {lr_trans}\n")
        file.write(f"wd_beta = {wd_beta}\n")
        file.write(f"wd_poses = {wd_poses}\n")
        file.write(f"wd_trans = {wd_trans}\n")
        file.write(f"i_chamfer_thresh = {i_chamfer_thresh}\n")
        file.write(f"k_loss_thresh = {k_loss_thresh}\n")
        file.write(f"s_loss_thresh = {s_loss_thresh}\n")
        file.write(f"i_chamfer_weight = {i_chamfer_weight}\n")
        file.write(f"k_loss_weight = {k_loss_weight}\n")
        file.write(f"s_loss_weight = {s_loss_weight}\n")    


# size: scans[num_scans,num_scan_points,3] poses[num_scans,48] beta[10] trans[num_scans,3]
def optimize(scans, scans_ids, poses=0, beta=0, trans=0,  num_steps=num_steps, lr_poses=lr_poses, lr_beta=lr_beta, lr_trans=lr_trans, i_chamfer_thresh=i_chamfer_thresh, i_chamfer_weight=i_chamfer_weight, if_fixed_beta=False):
    
    num_scans = len(scans)
    if num_scans == 0:
        return 0,0,0,0
    if type(poses) is int:
        poses = torch.zeros((num_scans,48),dtype=torch.float32,device=device)

    if type(beta) is int:
        beta = torch.zeros(10,dtype=torch.float32,device=device)
    
    if type(trans) is int:
        #trans = torch.zeros((num_scans,3),dtype=torch.float32,device=device)
        betas = beta.unsqueeze(0).expand(num_scans, 10)
        verts, _ = mano_layer(poses, betas)
        trans = - verts.mean(dim=1)
    
    optimizer = 0
    if not if_fixed_beta:
        optimizer = Adam([
            { "params": beta, "lr": lr_beta, "weight_decay": wd_beta},
            { "params": trans, "lr": lr_trans, "weight_decay": wd_trans },
            { "params": poses, "lr": lr_poses, "weight_decay": wd_poses }
        ])
        beta.requires_grad_(True)
    else:
        optimizer = Adam([
            { "params": trans, "lr": lr_trans, "weight_decay": wd_trans},
            { "params": poses, "lr": lr_poses, "weight_decay": wd_poses }
        ])
        beta.requires_grad_(False)
        tqdm.write("Beta is fixed.")
    trans.requires_grad_(True)
    poses.requires_grad_(True)

    poses_prior = torch.zeros((num_scans,48),dtype=torch.float32,device=device)
    if ("ktest12" in  scans_ids):
        i_kt= scans_ids.index("ktest12")
        poses_prior[i_kt] = pose_prior_k12
    
    graph_x = []
    graph_y = []


    for i in trange(num_steps):

        optimizer.zero_grad()
        betas = beta.unsqueeze(0).expand(num_scans, 10)
        verts, joints = mano_layer(poses+poses_prior, betas) # [num_scans,num_reg_points,3] [num_scans,21,3]
        verts = verts + trans.view(num_scans, -1).unsqueeze(1)
        joints = joints + trans.view(num_scans, -1).unsqueeze(1)
        # verts, joints = generate_hand_verts(poses,beta,trans) ???
        dloss, f_chamfer, i_chamfer = duplex_chamfer_dist(verts,scans,i)
        # kloss = k_finger_match_dist(joints,scans_ids,i)
        # sloss = scan_finger_match_dist(joints, scans_ids,i)
        # rloss = regulation_loss(poses,scans_ids)
        rloss_beta = gauss_regulation_beta(beta)
        rloss_poses = gauss_regulation_poses(poses)
        loss = dloss.mean() + rloss_beta + rloss_poses.mean()
        loss.backward()
        optimizer.step()
        

        v_f_chamfer = f_chamfer
        v_i_chamfer = i_chamfer / num_scan_points * num_reg_points
        # log_loss = torch.log10(loss)
        if i < write_thresh:
            writer.add_scalars("Loss",{"loss":loss.item()},i,walltime=math.log10(i+1))
            #graph_x.append(math.log10(i+1))
            graph_x.append(i)
            graph_y.append(loss.item())
        # writer.add_scalars("LossLogStep",{"loss":loss},math.log10(i+1))
        # writer.add_scalars("LogLoss",{"LogLoss":log_loss},i)
        writer.add_scalars("ChamferDistances/avg", 
                    {"f_chamfer":v_f_chamfer.mean().item(),"i_chamfer":v_i_chamfer.mean().item()},i)
        for i_scan in range(num_scans):
            if scans_ids[i_scan] in visualized_scans_ids:
                writer.add_scalars(f"ChamferDistances/{scans_ids[i_scan]}", 
                        {"f_chamfer":v_f_chamfer[i_scan],"i_chamfer":v_i_chamfer[i_scan]},i)        


        # if (i % visualize_step == 0 or i == num_steps - 1):
        if (i % write_mesh_step == 0 or i == num_steps-1):
            with torch.no_grad():
                all_verts = []
                all_faces = []
                for i_scan in range(num_scans): # each scan of user needs visualization
                    v_verts = verts[i_scan].cpu().numpy()
                    v_joints = joints[i_scan].cpu().numpy()
                    v_scan =  scans[i_scan].cpu().numpy()
                    v_faces = mano_layer.th_faces.cpu().numpy()
                    if (scans_ids[i_scan] in visualized_scans_ids):
                        all_verts.append(v_verts)
                        all_faces.append(v_faces)
                    if (i % visualize_step == 0 or i == num_steps-1):
                    # if (i in visualize_list or i == num_steps-1):
                        visualize(v_verts,v_faces,v_joints,v_scan,scans_ids[i_scan],i)
                writer.add_mesh(f"Registered poses",vertices=np.array(all_verts),faces=np.array(all_faces),global_step=i)
                
                # tqdm.write(f"f_chamfer {f_chamfer.cpu().numpy()}")
                # tqdm.write(f"i_chamfer {i_chamfer.cpu().numpy()}")
            if (i % visualize_step == 0):
            # if (i in visualize_list):
                tqdm.write(f"Step {i}: f_chamfer {v_f_chamfer.cpu().detach().numpy()}, loss {loss.cpu().detach().numpy()}")
                # if (input()=='n'):
                #      return poses, beta, trans, verts, v_f_chamfer.cpu().detach().numpy(), i
            if (i == num_steps-1):
                tqdm.write(f"Step {i}: f_chamfer {v_f_chamfer.cpu().detach().numpy()}, loss {loss.cpu().detach().numpy()}")
                with open(loss_dir,"wb") as file:
                    pickle.dump({"step":graph_x,"loss":graph_y},file)
                return poses, beta, trans, verts, v_f_chamfer.cpu().detach().numpy(), i


def register(training_scans, training_scans_ids, test_scans, test_scans_ids):
    if len(training_scans_ids) == 0:
        return
    # poses, beta, trans = load_poses(scans,num_scans, scans_ids)
    # optimize(scans,num_scans,poses,beta,trans, scans_ids)
    print(f"Training set is {training_scans_ids}.Test set is {test_scans_ids}")
    save_dir_this = '_'.join(training_scans_ids) + ';' + '_'.join(test_scans_ids)
    save_dir_this = f"{save_dir}/{save_dir_this}"

    poses, beta, trans, verts, training_f_chamfer, step = optimize(training_scans,training_scans_ids)
    if (if_save_results):
        save_registered_hands(save_dir_this,beta,training_scans_ids,poses,trans,verts,if_write_all=(len(test_scans_ids)==0))
    if len(test_scans_ids) == 0 and if_write_to_results:
        with open(results_file,'a') as file:
            file.write(f"train={training_scans_ids} test={test_scans_ids} train_fc={training_f_chamfer} test_fc=[] avg_f_chamfer={training_f_chamfer.mean()}\n")
        return
    
    poses, beta, trans, verts, test_f_chamfer, step = optimize(test_scans,test_scans_ids,beta=beta,if_fixed_beta=True)
    if (if_save_results):
        save_registered_hands(save_dir_this,beta,test_scans_ids,poses,trans,verts,if_write_all=True,tag="test")
    if (if_write_to_results):
        with open(results_file,'a') as file:
            file.write(f"train={training_scans_ids} test={test_scans_ids} train_fc={training_f_chamfer} test_fc={test_f_chamfer} avg_f_chamfer={np.concatenate((training_f_chamfer,test_f_chamfer)).mean()}\n")

if __name__ == "__main__":
    if (if_write_to_results):
        with open(results_file,'a') as file:
            file.write(f"{run_time}:\n")
    scans, num_scans, scans_ids = load_scans() # num_scans x num_scan_points x 3
    if if_fixed_beta:
        beta = load_beta()
        poses, beta, trans, verts, f_chamfer, step = optimize(scans,scans_ids,beta=beta,if_fixed_beta=True)
        if (if_save_results):
            save_registered_hands(save_dir,beta,scans_ids,poses,trans,verts,if_write_all=True,tag="fixed_beta")
            save_args(num_steps=step)
        if (if_write_to_results):
            with open(results_file,'a') as file:
                file.write(f"fixed_beta={scans_ids} fc={f_chamfer} avg={f_chamfer.mean()}\n\n")
    else:
        indices_list = [[0]]
        print(f"Indices for cross validation are {indices_list}.")
        for training_indices in indices_list:
            training = scans[torch.tensor(training_indices)] if len(training_indices) > 0 else None
            training_ids = [scans_ids[i] for i in training_indices]
            test_indices = [i for i in range(num_scans) if not i in training_indices]
            test = scans[torch.tensor(test_indices)] if len(test_indices) > 0 else None
            test_ids = [scans_ids[i] for i in test_indices]
            
            register(training,training_ids,test,test_ids)
        if (if_write_to_results):
            with open(results_file,'a') as file:
                file.write("\n")
        if (if_save_results):
            save_args()