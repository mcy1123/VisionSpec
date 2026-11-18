"""Fixed video-memory transforms operating on [time, region, hidden]."""
import torch

LAYOUTS = (
    'appearance_motion', 'local_change', 'slow_motion', 'acceleration',
    'foreground_motion', 'scene_motion', 'innovation', 'spatial_contrast',
    'trajectory', 'keyframe_delta', 'peak_region', 'signed_motion',
)

def transform(layout, grid, clock):
    if layout not in LAYOUTS:
        raise ValueError(layout)
    frame = grid.mean(dim=1)
    t = clock[:, 0]
    delta = torch.diff(grid, dim=0, prepend=grid[:1])
    if layout == 'appearance_motion':
        return torch.cat((frame, delta.flatten(0, 1))), torch.cat((t, clock.flatten()))
    if layout == 'local_change':
        idx = delta.square().mean(-1).argmax(1)
        row = torch.arange(len(grid), device=grid.device)
        return torch.cat((frame, delta[row, idx])), torch.cat((t, t))
    if layout == 'slow_motion':
        previous = (torch.arange(len(grid), device=grid.device) - 4).clamp_min(0)
        out = grid - grid[previous]
    elif layout == 'acceleration':
        out = torch.diff(delta, dim=0, prepend=delta[:1])
    elif layout == 'foreground_motion':
        out = delta - delta.mean(dim=1, keepdim=True)
    elif layout == 'scene_motion':
        return torch.cat((frame[:1], frame[1:] - frame[:-1])), t
    elif layout == 'innovation':
        previous_mean = grid.cumsum(0) / torch.arange(1,len(grid)+1,device=grid.device)[:,None,None]
        out = grid - torch.cat((grid[:1], previous_mean[:-1]))
    elif layout == 'spatial_contrast':
        out = grid - frame[:, None]
    elif layout == 'trajectory':
        # Keep local appearance alongside displacement from the initial scene.
        return torch.cat((frame, (grid-grid[:1]).flatten(0,1))), torch.cat((t,clock.flatten()))
    elif layout == 'keyframe_delta':
        start = torch.arange(len(grid), device=grid.device)
        out = grid - grid[(start // 4)*4]
        out[::4] = grid[::4]
        return out.flatten(0,1), clock.flatten()
    elif layout == 'peak_region':
        row = torch.arange(len(grid), device=grid.device)
        idx = delta.square().mean(-1).argmax(1)
        return grid[row,idx], t
    elif layout == 'signed_motion':
        change = delta.mean(dim=1)
        return torch.cat((frame[:1],change[1:].clamp_min(0),change[1:].clamp_max(0))),torch.cat((t[:1],t[1:],t[1:]))
    else:
        raise ValueError(layout)
    return torch.cat((grid[:1],out[1:])).flatten(0,1),clock.flatten()
