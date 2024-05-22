import torch
from torch import nn
import numpy as np

CANVAS_SIZE = 128

def action2traj(action):
    # action: n x 6
    n = len(action)
    length, bend1, bend2, start_x, start_y, theta = action[:,0], action[:,1], action[:,2], action[:,3], action[:,4], action[:,5]
    bend1 = bend1*0.1 - 0.05
    bend2 = bend2*0.1 - 0.05
    length = length * 0.5 + 0.05
    start_x = start_x * 0.8 + 0.1
    start_y = start_y * 0.8 + 0.1
    theta = theta * 2*np.pi

    zero = torch.zeros(n).to(action.device)
    p0 = torch.stack([zero, zero], dim=1)
    p1 = torch.stack([length/3, bend1], dim=1)
    p2 = torch.stack([length/3*2, bend2], dim=1)
    p3 = torch.stack([length, zero], dim=1)

    points = torch.stack([p0, p1, p2, p3], dim=1) # n x 4 x 2

    # Rotate and translate
    rot = torch.stack([
        torch.stack([torch.cos(theta), -torch.sin(theta)], dim=1),
        torch.stack([torch.sin(theta), torch.cos(theta)], dim=1)
    ], dim=1) # n x 2 x 2
    rot = torch.transpose(rot, 1, 2)
    points = points @ rot # n x 4 x 2
    
    trans = torch.stack([start_x, start_y], dim=1).unsqueeze(1) # n x 1 x 2
    points = points + trans # n x 4 x 2

    traj = []
    POINTS_PER_TRAJECTORY = 8
    for i in range(POINTS_PER_TRAJECTORY):
        t = i / (POINTS_PER_TRAJECTORY - 1)
        pos = (1-t)**3 * points[:,0] + 3*(1-t)**2 * t * points[:,1] + 3*(1-t) * t**2 * points[:,2] + t**3 * points[:,3] # n x 2
        traj.append(pos)
    traj = torch.stack(traj, dim=1) # n x POINTS_PER_TRAJECTORY x 2

    return traj

class DiffPathRenderer(nn.Module):
    def __init__(self):
        super(DiffPathRenderer, self).__init__()

        idxs_x = torch.arange(CANVAS_SIZE)
        idxs_y = torch.arange(CANVAS_SIZE)
        x_coords, y_coords = torch.meshgrid(idxs_y, idxs_x, indexing='ij') # CANVAS_SIZE x CANVAS_SIZE
        self.grid_coords = torch.stack((y_coords, x_coords), dim=2).reshape(1,CANVAS_SIZE,CANVAS_SIZE,2) # 1 x CANVAS_SIZE x CANVAS_SIZE x 2

    def forward(self, traj, thickness):
        # traj: B x n x 2
        traj = traj * CANVAS_SIZE
        B, n, _ = traj.shape

        vs = traj[:,:-1,:].reshape((B, n-1, 1, 1, 2)) # (B, n-1, 1, 1, 2)
        vs = torch.tile(vs, (1, 1, CANVAS_SIZE, CANVAS_SIZE, 1)) # (B, n-1, CANVAS_SIZE, CANVAS_SIZE, 2)

        ws = traj[:,1:,:].reshape((B, n-1, 1, 1, 2)) # (B, n-1, 1, 1, 2)
        ws = torch.tile(ws, (1, 1, CANVAS_SIZE, CANVAS_SIZE, 1)) # (B, n-1, CANVAS_SIZE, CANVAS_SIZE, 2)

        coords = torch.tile(self.grid_coords, (n-1,1,1,1)).to(ws.device) # (n-1, CANVAS_SIZE, CANVAS_SIZE, 2)

        # For each of the n-1 segments, compute distance from every point to the line
        def dist_line_segment(p, v, w):
            d = torch.linalg.norm(v-w, dim=4) # B x n-1 x CANVAS_SIZE x CANVAS_SIZE
            dot = (p-v) * (w-v) # B x n-1 x CANVAS_SIZE x CANVAS_SIZE x 2
            dot_sum = torch.sum(dot, dim=4) / (d**2 + 1e-5) # B x n-1 x CANVAS_SIZE x CANVAS_SIZE
            t = dot_sum.unsqueeze(4) # B x n-1 x CANVAS_SIZE x CANVAS_SIZE x 1
            t = torch.clamp(t, min=0, max=1) # N x CANVAS_SIZE x CANVAS_SIZE x 1
            proj = v + t * (w-v) # B x n-1 x CANVAS_SIZE x CANVAS_SIZE x 2
            return torch.linalg.norm(p-proj, dim=4)
        distances = dist_line_segment(coords, vs, ws) # (B, n-1, CANVAS_SIZE, CANVAS_SIZE)
        distances = torch.min(distances, dim=1).values # (B, CANVAS_SIZE, CANVAS_SIZE)

        radius = thickness/2
        darkness = torch.clamp((radius - distances) / radius, min=0.0, max=1.0)
        # darkness = 1 - (darkness-1) ** 4
        # darkness = darkness ** 4

        darkness = torch.sigmoid(70 * (darkness - 0.9))

        return darkness

renderer = DiffPathRenderer()

def render(action):
    traj = action2traj(action)
    stroke = renderer(traj, 20)
    return stroke
