import numpy as np
import trimesh as tm
import plotly.graph_objects as go
from scipy.spatial.distance import cdist

def plot_point_cloud(pts, **kwargs):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode='markers',
        **kwargs
    )


    
def plot_mesh(mesh, color='lightblue', opacity=1.0, name='mesh'):
    return go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        color=color, opacity=opacity, name=name, showlegend=True)

def plot_point_cloud_cmap(pts, color_levels=None, size_levels = 1,name="points"):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        name=name,
        mode='markers',
        marker={
            'color': color_levels,
            'size': size_levels,
            'opacity': 1
        }
    )

def get_color_levels(reg_points, scan_points, num_scan_points): # num_verts x 3  num_points x 3
    dists = cdist(reg_points, scan_points)
    indexs = np.argmin(dists, axis=1)
    return np.bincount(indexs, minlength=num_scan_points)

def plot_vector(x,y,z,color='blue',width=5):
    return go.Scatter3d(
    x=[0,x*0.1],
    y=[0,y*0.1],
    z=[0,z*0.1],
    mode='lines',
    line=dict(color=color, width=width),
    showlegend=False,
    hoverinfo='none'
    )