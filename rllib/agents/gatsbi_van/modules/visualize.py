import torch
import numpy as np

import torchvision
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Subset, DataLoader
from PIL import Image, ImageDraw
from attrdict import AttrDict
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from ray.rllib.agents.gatsbi_van.modules.arch import ARCH
from ray.rllib.agents.gatsbi_van.modules.vis_utils import draw_boxes, figure_to_numpy, make_gif
from ray.rllib.agents.gatsbi_van.modules.utils import transform_tensors
from torch.distributions import Normal

def clean_log(log, num):
    log = transform_tensors(log, lambda x: x.cpu().detach())
    def take_batch(x):
        if isinstance(x, torch.Tensor) and x.size(0) > 1: return x[:num]
        else: return x
    log = transform_tensors(log, take_batch)
    log = AttrDict(log)
    return log

@torch.no_grad()
def train_vis(model, dataset, writer: SummaryWriter, global_step, indices, device,
    cond_steps, fg_sample, bg_sample, num_gen, optimizer):

    grid, gif = show_tracking(model, dataset, indices, device, optimizer)
    writer.add_image('tracking/grid', grid, global_step)
    for i in range(len(gif)):
        writer.add_video(f'tracking/video_{i}', gif[i:i+1], global_step)

    # Generation
    grid, gif = show_generation(model, dataset, indices, device, cond_steps, fg_sample=fg_sample, bg_sample=bg_sample,
            num=num_gen, optimizer=optimizer)
    writer.add_image('generation/grid', grid, global_step)
    for i in range(len(gif)):
        writer.add_video(f'generation/video_{i}', gif[i:i+1], global_step)

@torch.no_grad()
def  show_tracking_batch(model, obs, action, num):
    """
    Visualize the batch of episodes
    Args:
        imgs: (B, T, 3, H, W)
        recons: (B, T, 3, H, W)
        z_where: [(B, N, 4)] * T
        z_pres: [(B, N, 1)] * T
        ids: [(B, N)] * T
    Returns:
        grid: (3, H, W) obs[0, 0]
        gif: (B, T, 3, N*H, W)
    """
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.eval()
    mixture_out = model.mixture_module.encode(
        seq=obs, action=action) # [B, T, 3, 64, 64],

    obs_diff = obs - mixture_out['bg'] # [0]

    inpt = torch.cat([obs, obs_diff], dim=2)

    obj_out = model.obj_module.track(obs=inpt, mix=mixture_out['bg'],
        discovery_dropout=ARCH.DISCOVERY_DROPOUT,
        z_agent=mixture_out['z_masks'][:, :, model.agent_slot_idx], # state of the agent
        h_agent=mixture_out['h_masks_post'][:, :, model.agent_slot_idx].detach(), # history of the agent
        enhanced_act=mixture_out['enhanced_act'].detach(),
        agent_mask=mixture_out['masks'][:, :,model.agent_slot_idx].clone()
    )

    alpha_map = obj_out['alpha_map'] # [B, T, 1, H, W]
    # recon the observation
    fg = obj_out['fg'] # [B, T, 3, H, W]
    bg = mixture_out['bg'] # [B, T, 3, H, W]
    
    things = mixture_out.copy()
    things.update(obj_out.copy())

    recon = fg + (1 - alpha_map) * bg

    things.update(
        imgs=obs,
        recon=recon
    )

    # trim out batch indicated by indices
    log = clean_log(things, num)

    B, T, _, H, W = log.fg.size() # [B,] = len(indices)
    fg_boxes = torch.zeros_like(log.fg)
    fg_proposals = torch.zeros_like(log.fg)
    for t in range(T):
        fg_boxes[:, t] = draw_boxes(log.fg[:, t], log.z_where[:, t], log.z_pres[:, t], log.ids[:, t])
        fg_proposals[:, t] = draw_boxes(log.fg[:, t], log.proposal[:, t], log.z_pres[:, t], log.ids[:, t])
    # (B, 3T, 3, H, W)
    grid = torch.cat([log.imgs, log.recon, log.fg, fg_boxes, log.bg] +
        [m for m in log.masks.permute(2,0,1,3,4,5).repeat(1, 1, 1, 3, 1, 1)]
        , dim=1)


    grid = grid.view(-1, 3, H, W)
    # (3, H, W)
    grid = make_grid(grid, nrow=T, pad_value=1)
    # (B, T, 3, 3H, W)
    gif = torch.cat([log.imgs, log.recon, log.fg, fg_boxes, log.bg ] +
        [m * c for (m, c) in zip(log.masks.permute(2,0,1,3,4,5), log.comps.permute(2,0,1,3,4,5))], dim=-1)

    model.train()
    return grid, gif


@torch.no_grad()
def show_episode(model, obs):
    """
    Visualize a single episode.
    """
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.eval()

    #* reshape into (B, T, 3, H, W)
    gif = 3

@torch.no_grad()
def show_generation_batch(model, obs, action, num, cond_steps):
    """
    Args:
        obs: (B, T, 3, H, W)
        recons: (B, T, 3, H, W)
        z_where: [(B, N, 4)] * T
        z_pres: [(B, N, 1)] * T
        ids: [(B, N)] * T
        cond_steps # TODO (chmin): how many steps should be conditioned?
        sample
        num: number of generation samples
    Returns:
    """
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.eval()

    mixture_out = model.mixture_module.generate(
        obs, 
        action, 
        cond_steps=cond_steps,
        agent_slot=model.agent_slot_idx
    )
       
    bg_indices = torch.tensor([k for k in range(ARCH.K) if k != model.agent_slot_idx], device=obs.device).long()

    obs_diff = obs - mixture_out['bg']
    inpt = torch.cat([obs, obs_diff], dim=2)

    A = action.size(-1)
    if model.global_step < ARCH.KYPT_MIX_JOINT_UNTIL: 
        action = mixture_out['enhanced_act']
    else:
        action = mixture_out['enhanced_act'].detach()

    obj_out = model.obj_module.generate(obs=inpt, mix=mixture_out['bg'], cond_steps=cond_steps,
        sample=True, z_agent=mixture_out['z_masks'][:, :, model.agent_slot_idx],
        h_agent=mixture_out['h_masks'][:, :, model.agent_slot_idx].detach(),
        enhanced_act=mixture_out['enhanced_act'].detach(),
        agent_mask=mixture_out['masks'][:, :,model.agent_slot_idx].clone()
        )

    alpha_map = obj_out['alpha_map'] # [B, T, 1, 64, 64]
    fg = obj_out['fg'] # [B, T, 3, 64, 64]
    bg = mixture_out['bg'] # [B, T, 3, 64, 64]

    recon = fg + (1 - alpha_map) * bg

    things = mixture_out.copy()
    things.update(obj_out.copy())
    things.update(
        imgs=obs,
        recon=recon
    )

    log = clean_log(things, num)
    B, T, _, H, W = log.fg.size()
    fg_boxes = torch.zeros_like(log.fg)
    for t in range(T): # draw bounding boxes over a sequence of images
        fg_boxes[:, t] = draw_boxes(log.fg[:, t], log.z_where[:, t], log.z_pres[:, t], log.ids[:, t])
        # (B, 3T, 3, H, W)
    grid = torch.cat([log.imgs, log.recon, log.fg, fg_boxes, log.bg] +
        [m * c for (m, c) in zip(log.masks.permute(2,0,1,3,4,5), log.comps.permute(2,0,1,3,4,5))] 
        , dim=1)
    grid = grid.view(-1, 3, H, W)
    # (3, H, W)
    grid = make_grid(grid, nrow=T, pad_value=1)
    # (B, T, 3, H, 3W)
    gif = torch.cat([log.imgs, log.recon, log.fg, fg_boxes, log.bg] +
        [m * c for (m, c) in zip(log.masks.permute(2,0,1,3,4,5), log.comps.permute(2,0,1,3,4,5))], dim=-1)
    add_boundary(gif[:, cond_steps:])

    # # (B, T, 3, num*H, N*W)
    model.train()
    return grid, gif

def make_gswm_grid(
        imgs,
        recon,
        fg,
        bg,
        z_where,
        z_pres,
        proposal,
        ids
):
    """
    Args:
        imgs: (B, T, 3, H, W)
        recons: (B, T, 3, H, W)
        z_where: [(B, N, 4)] * T
        z_pres: [(B, N, 1)] * T
        proposal: [(B, N, 4)] * T
        ids: [(B, N)] * T
    Returns:
    """

    if fg is None and bg is None:
        B, T, _, H, W = recon.size()
        # (B, 3T, 3, H, W)
        grid = torch.cat([imgs, recon], dim=1)
        grid = grid.view(B * 2 * T, 3, H, W)
    else:
        B, T, _, H, W = fg.size()
        fg_boxes = torch.zeros_like(fg)
        fg_proposals = torch.zeros_like(fg)
        for t in range(T):
            fg_boxes[:, t] = draw_boxes(fg[:, t], z_where[:, t], z_pres[:, t], ids[:, t])
            fg_proposals[:, t] = draw_boxes(fg[:, t], proposal[:, t], z_pres[:, t], ids[:, t])
        # (B, 3T, 3, H, W)
        grid = torch.cat([imgs, recon, fg_boxes, fg_proposals, bg], dim=1)
        grid = grid.view(B * 5 * T, 3, H, W)
    # (3, H, W)
    grid = make_grid(grid, nrow=T, pad_value=1)

    return grid

@torch.no_grad()
def log_summary(model, batch, global_step, indices, device, cond_steps, fg_sample, bg_sample, num_gen):
    obs, action = batch # [B, T, 3, 64, 64], [B, T, A (enhanced)]
    obs = obs[:, :ARCH.VIS_TIMESTEP_TRUNC]
    action = action[:, :ARCH.VIS_TIMESTEP_TRUNC]

    track_grid, track_gifs = show_tracking_batch(model, obs, action, num_gen)
    # writer.add_image('tracking/grid', grid, global_step)
    # for i in range(len(gif)):
    #     writer.add_video(f'tracking/video_{i}', gif[i:i+1], global_step)
    # Generation
    gen_grid, gen_gifs = show_generation_batch(model, obs, action, num_gen, cond_steps)
    # writer.add_image('generation/grid', grid, global_step)
    # for i in range(len(gif)):
    #     writer.add_video(f'generation/video_{i}', gif[i:i+1], global_step)
    torch.cuda.empty_cache()
    return track_grid, track_gifs, gen_grid, gen_gifs

@torch.no_grad()
def train_vis_batch(model, batch, writer: SummaryWriter, global_step, indices, device, cond_steps, fg_sample, bg_sample, num_gen, optimizer):
    """
        Visualize the training procedure on the batch data.
    """
    grid, gif = show_tracking(model, batch, indices, device, optimizer)
    writer.add_image('tracking/grid', grid, global_step)
    for i in range(len(gif)):
        writer.add_video(f'tracking/video_{i}', gif[i:i+1], global_step)

    # Generation
    grid, gif = show_generation(model, batch, indices, device, cond_steps, fg_sample=fg_sample, bg_sample=bg_sample,
            num=num_gen, optimizer=optimizer)
    writer.add_image('generation/grid', grid, global_step)
    for i in range(len(gif)):
        writer.add_video(f'generation/video_{i}', gif[i:i+1], global_step)

def get_batch(dataset, indices, device):
    # For some chosen data we show something here
    dataset = Subset(dataset, indices)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    # Do not use multiple GPUs
    # (B, T, 3, H, W)
    imgs, actions, *_ = next(iter(dataloader))
    imgs = imgs.to(device)
    actions = actions.to(device)
    return imgs, actions

@torch.no_grad()
def show_gif(model, dataset, indices, device, cond_steps, gen_len, path, fps, optimizer):
    if isinstance(model, nn.DataParallel):
        model = model.module
    # model.eval()
    # get data
    imgs, ees = get_batch(dataset, indices, device)
    imgs = imgs[:, :gen_len]
    ees = ees[:, :gen_len]
    model_fn = lambda model, imgs: model.generate(imgs, cond_steps=cond_steps, fg_sample=False, bg_sample=False, optimizer=optimizer)
    log = model_fn(model, imgs)
    log = AttrDict(log)
    recon = log.recon

    ori = []
    gen = []
    def trans(img):
        # (T, 3, H, W)
        img = img.permute(0, 2, 3, 1).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        return img
    add_boundary(recon[:, cond_steps:])
    for b in range(len(indices)):
        ori.append(trans(imgs[b]))
        gen.append(trans(recon[b]))

    frames = []
    heights = [1] * 2
    widths = [1] * (len(indices) + 1)
    h, w = sum(heights), sum(widths)
    f, axes = plt.subplots(2, len(indices) + 1, figsize=(w, h), gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
    axes[0][0].text(0.95, 0.5, 'GT', fontdict={'fontsize': 12}, verticalalignment='center', horizontalalignment='right')
    axes[1][0].text(0.95, 0.5, 'GSWM', fontdict={'fontsize': 12}, verticalalignment='center', horizontalalignment='right')
    for ax in axes.ravel():
        ax.axis('off')
    for t in range(gen_len):
        im_list = []
        for b in range(len(indices)):
            im1 = axes[0][b + 1].imshow(ori[b][t])
            im2 = axes[1][b + 1].imshow(gen[b][t])
            im_list.extend((im1, im2))
        frames.append(figure_to_numpy(f))
        for x in im_list: x.remove()
    make_gif(images=frames, path=path, fps=fps)

def draw_grid(imgs):
    """
    Args:
        imgs: (..., 3, H, W)
    Returns:
    """
    # img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    G = 4
    IMG_H = 64
    IMG_W = 64
    SCALE_FACTOR = 0.8
    l = (-SCALE_FACTOR + 1.0) / 2 * (IMG_H - 1)
    l = int(l)
    r = (SCALE_FACTOR + 1.0) / 2 * (IMG_H - 1)
    r = int(r)
    for pos in torch.linspace(-1, 1, G + 1):
        pos *= SCALE_FACTOR
        pos = (pos + 1.0) / 2 * (IMG_H - 1)
        pos = int(pos)
        imgs[..., :, pos, l:r] = 1.0
        imgs[..., :, l:r, pos] = 1.0

def add_boundary(img, width=1, color=(1.0, 1.0, 0)):
    """
    Args:
        img: (..., 3, H, W)
        width:
        color:
    Returns:
    """
    color = torch.tensor(color, device=img.device, dtype=img.dtype)
    assert color.size() == (3,)
    color = color.view(-1, 1, 1)
    img[..., :, :width, :] = color
    img[..., :, -width:, :] = color
    img[..., :, :, :width] = color
    img[..., :, :, -width:] = color

def draw_trajectories(z_where, z_pres):
    """
    Args:
        z_where: (B, N, D) * T
        z_pres: (B, N, D) * T
    Returns:
        (B, 3, H, W)
    """
    # Transform z_where to image coordinates
    z_where = [((x[..., 2:] + 1.0) / 2 * 64) for x in z_where]
    T = len(z_where)
    B, N, _ = z_where[0].size()
    images = [Image.new('RGB', (64, 64)) for b in range(B)]

    for b in range(B):
        for t in range(T-1):
            for n in range(N):
                # (2,)
                start = tuple(int(x) for x in z_where[t][b][n])
                end = tuple(int(x) for x in z_where[t+1][b][n])
                draw = ImageDraw.Draw(images[b])
                color = int(255 * z_pres[t][b][n])
                draw.line([start, end], fill=(color,)*3)

    images = [torchvision.transforms.ToTensor()(x) for x in images]
    images = torch.stack(images, dim=0)
    return images