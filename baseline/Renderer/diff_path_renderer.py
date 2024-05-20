import torch
from torch import nn
import numpy as np

CANVAS_SIZE = 128

def action2traj(action):
    length, bend1, bend2, start_x, start_y, theta = action
    length = length * 0.5 + 0.05
    start_x = start_x * 0.8 + 0.1
    start_y = start_y * 0.8 + 0.1
    theta = theta * 2*np.pi

    zero = torch.tensor(0.0).to(action.device)
    p0 = torch.stack([zero, zero])
    p1 = torch.stack([length/3, bend1])
    p2 = torch.stack([length/3*2, bend2])
    p3 = torch.stack([length, zero])

    points = torch.stack([p0, p1, p2, p3])

    # Rotate and translate
    rot = torch.stack([
        torch.stack([torch.cos(theta), -torch.sin(theta)]),
        torch.stack([torch.sin(theta), torch.cos(theta)])
    ])
    points = points @ rot.T
    points = points + torch.stack([start_x, start_y])

    traj = []
    POINTS_PER_TRAJECTORY = 16
    for i in range(POINTS_PER_TRAJECTORY):
        t = i / (POINTS_PER_TRAJECTORY - 1)
        pos = (1-t)**3 * points[0] + 3*(1-t)**2 * t * points[1] + 3*(1-t) * t**2 * points[2] + t**3 * points[3]
        traj.append(pos)
    traj = torch.stack(traj)

    return traj

class DiffPathRenderer(nn.Module):
    def __init__(self):
        super(DiffPathRenderer, self).__init__()

        idxs_x = torch.arange(CANVAS_SIZE)
        idxs_y = torch.arange(CANVAS_SIZE)
        x_coords, y_coords = torch.meshgrid(idxs_y, idxs_x, indexing='ij') # CANVAS_SIZE x CANVAS_SIZE
        self.grid_coords = torch.stack((y_coords, x_coords), dim=2).reshape(1,CANVAS_SIZE,CANVAS_SIZE,2) # 1 x CANVAS_SIZE x CANVAS_SIZE x 2

    def forward(self, traj, thickness):
        traj = traj * CANVAS_SIZE
        n = len(traj)

        vs = traj[:-1].reshape((-1,1,1,2)) # (n-1, 1, 1, 2)
        vs = torch.tile(vs, (1, CANVAS_SIZE, CANVAS_SIZE, 1)) # (n-1, CANVAS_SIZE, CANVAS_SIZE, 2)

        ws = traj[1:].reshape((-1,1,1,2)) # (n-1, 1, 1, 2)
        ws = torch.tile(ws, (1, CANVAS_SIZE, CANVAS_SIZE, 1)) # (n-1, CANVAS_SIZE, CANVAS_SIZE, 2)

        coords = torch.tile(self.grid_coords, (n-1,1,1,1)).to(ws.device) # (n-1, CANVAS_SIZE, CANVAS_SIZE, 2)

        # For each of the n-1 segments, compute distance from every point to the line
        def dist_line_segment(p, v, w):
            d = torch.linalg.norm(v-w, dim=3) # (n-1) x CANVAS_SIZE x CANVAS_SIZE
            dot = (p-v) * (w-v)
            dot_sum = torch.sum(dot, dim=3) / (d**2 + 1e-5)
            t = dot_sum.unsqueeze(3) # (n-1) x CANVAS_SIZE x CANVAS_SIZE x 1
            t = torch.clamp(t, min=0, max=1) # (n-1) x CANVAS_SIZE x CANVAS_SIZE x 1
            proj = v + t * (w-v) # (n-1) x CANVAS_SIZE x CANVAS_SIZE x 2
            return torch.linalg.norm(p-proj, dim=3)
        distances = dist_line_segment(coords, vs, ws) # (n-1, CANVAS_SIZE, CANVAS_SIZE)
        distances = torch.min(distances, dim=0).values

        radius = thickness/2
        darkness = torch.clamp((radius - distances) / radius, min=0.0, max=1.0)
        darkness = darkness ** 0.2

        return darkness

renderer = DiffPathRenderer()

def render(action):
    b = action.shape[0]
    strokes = []
    for i in range(b):
        traj = action2traj(action[i])
        stroke = renderer(traj, 1.5)
        strokes.append(stroke)

    stroke = torch.stack(strokes, dim=0)

    return stroke
