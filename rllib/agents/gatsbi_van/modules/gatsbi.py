import torch
import random
from torch import nn
from torch.distributions import Normal

from .module import anneal
from .arch import ARCH
from .mix import MixtureModule
from .keypoint import KeypointModule
from .obj import ObjModule
from ray.rllib.offline import JsonReader
import os


class GATSBI(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.slot_find_flag = False
        self.agent_slot_idx = 0

        self.T = ARCH.T[0]
        self.obj_module = ObjModule()
        self.mixture_module = MixtureModule()
        self.keypoint_module = KeypointModule()
        self.sigma = ARCH.SIGMA
        # add json reader
        
    def anneal(self, global_step):
        self.global_step = global_step
        self.obj_module.anneal(global_step)
        self.mixture_module.anneal(global_step)
        self.keypoint_module.anneal(global_step)
        
        assert len(ARCH.T) == len(ARCH.T_MILESTONES) + 1, 'len(T) != len(T_MILESTONES) + 1'
        i = 0
        while i < len(ARCH.T_MILESTONES) and global_step > ARCH.T_MILESTONES[i]:
            i += 1
        self.T = ARCH.T[i]

    def random_crop(self, seq, T, start=False):
        """
        Sample a subsequence of length T
        Args:
            seq: (B, Told, 3, H, W)
            T: length
        Returns:
            seq: (B, T, 3, H, W)
        """
        if start:  # crop the action.
            # action is given as a_{t-1}:t_{t+T} for action conditioning
            return seq[:, start:start + T], start
        else:
            t_seq = seq.size(1)
            assert t_seq >= T, f't_seq: {t_seq}, T: {T}'
            start = random.randint(0, t_seq - T)
            return seq[:, start:start + T], start

    # TODO (chmin): this should be connected to the 'forward' method of the agent.
    def compute_loss(self, seq, actions, global_step, vis_mode=False, optimizer=None):
        """
        A forward method for making a forward inference and backward training.
        Args:
            seq: (B, T, 3, H, W)
            ee_poses: (B, T, 7) - shape of end-effector poses (x, y, z & quaternions)
            global_step: global training step
        Returns:
        """


        self.anneal(global_step)
        # crop the training data
        seq, start_idx = self.random_crop(seq, self.T)  # crop the image sequence. 'start_idx' is the starting idx for cropping 
        action, _ = self.random_crop(actions, self.T, start_idx)  # crop the action sequence
        B, T, C, H, W = seq.size()
        log = self.track_agent(seq, action, ARCH.DISCOVERY_DROPOUT,
                        global_step=global_step, vis_mode=vis_mode, optimizer=optimizer)

        # (B, T, 1, H, W)
        alpha_map = log['alpha_map']
        fg = log['mix']
        bg = log['bg']
        # (B, T) => [B]
        kl_fg = log['kl_fg'].sum(-1)
        kl_bg = log['kl_bg'].sum(-1)

        if global_step and global_step < ARCH.FIX_ALPHA_STEPS:
            alpha_map = torch.full_like(alpha_map, ARCH.FIX_ALPHA_VALUES)
        # Compute total likelihood that consists of fg and bg; [B, ]
        loglikelihood = self.gaussian_likelihood(seq, fg, bg, alpha_map, global_step)

        kl = kl_bg if (ARCH.BG_ON and global_step < ARCH.MODULE_TRAINING_SCHEME[1]) else kl_bg + kl_fg
        # maximize elbo: maximize loglikelihood and minimize kl divergence
        elbo = loglikelihood - kl
        # Visualization
        assert elbo.size() == (B,)
        robot_embed_loss = torch.zeros_like(elbo).to(elbo.device)
        # --- training scheme ---
        # if keypoint network is trained and the agent slot is found. 
        if not self.slot_find_flag and self.global_step >= ARCH.MODULE_TRAINING_SCHEME[1]:
            self.slot_find_flag = True
            # find the agent slot from the masks using the keypoint map 
            self.agent_slot_idx = self.find_agent_slot(log['total_gaussians'].detach(), log['masks'].detach())
            print("AGENT SLOT : ", self.agent_slot_idx)
        # first few thousand steps are dedicated for keypoint learning
        if global_step < ARCH.MODULE_TRAINING_SCHEME[0]:
            loss = log['kypt_recon_loss'] + log['kypt_sep_loss'] + log['kypt_coord_pred_loss'] \
                 + log['kypt_kl_loss'] + log['kypt_reg_loss']
        # stops the keypoint learning
        elif global_step >= ARCH.MODULE_TRAINING_SCHEME[2]:
            # if train keypoint for additional steps 
            if ARCH.KYPT_MASK_LOSS_LONG:
                if ARCH.KYPT_MASK_JOINT_LOSS[1]: # keypoint map <-> mask joint training
                    robot_embed_loss = (log['total_gaussians'] - log['masks'][:, :, self.agent_slot_idx]) ** 2  # (B, T, C, H, W)
                else: # keypoint ->> mask training
                    robot_embed_loss = (log['total_gaussians'].detach() - log['masks'][:, :, self.agent_slot_idx]) ** 2  # (B, T, C, H, W)
                robot_embed_loss = robot_embed_loss.sum(-1).sum(-1).sum(-1).sum(-1) # [B, T, C, H, W] -> [B, ] 
            # accumulate loss tensors: maximize elbo, minimize embed loss and residual losses
            loss = -elbo + robot_embed_loss + \
                ARCH.RESIDUAL_RES * torch.norm(log['mask_residuals'], p=2, dim=-1).sum(dim=[1, 2]) + \
                ARCH.RESIDUAL_RES * torch.norm(log['comp_residuals'], p=2, dim=-1).sum(dim=[1, 2])
        else: # keypoint + background
            if ARCH.KYPT_MASK_LOSS and self.global_step >= ARCH.MODULE_TRAINING_SCHEME[1]:
                if ARCH.KYPT_MASK_JOINT_LOSS[0]: # keypoint map <-> mask joint training
                    robot_embed_loss = (log['total_gaussians'] - log['masks'][:, :, self.agent_slot_idx]) ** 2  # (B, T, C, H, W)
                else:
                    robot_embed_loss = (log['total_gaussians'].detach() - log['masks'][:, :, self.agent_slot_idx]) ** 2  # (B, T, C, H, W)
                robot_embed_loss = robot_embed_loss.sum(-1).sum(-1).sum(-1).sum(-1)
            # accumulate loss tensors: maximize elbo, minimize keypoint losses
            loss = -elbo + log['kypt_recon_loss'] + log['kypt_sep_loss'] + log['kypt_coord_pred_loss'] + log['kypt_kl_loss'] + \
            log['kypt_reg_loss'] + log['bg_reg_loss'].sum(-1) + robot_embed_loss + \
            ARCH.RESIDUAL_RES * torch.norm(log['mask_residuals'], p=2, dim=-1).sum(dim=[1, 2]) + \
            ARCH.RESIDUAL_RES * torch.norm(log['comp_residuals'], p=2, dim=-1).sum(dim=[1, 2]) 

        log.update(
            alpha_mean=alpha_map,
            elbo=elbo,
            mse=(seq - log['recon']) ** 2,
            loglikelihood=loglikelihood,
            kl=kl,
            robot_embed_loss=robot_embed_loss,
            mask_res_norm=torch.norm(log['mask_residuals'], p=2, dim=-1).sum(dim=[1, 2]),
            comp_res_norm=torch.norm(log['comp_residuals'], p=2, dim=-1).sum(dim=[1, 2]),
        )
        return loss, log

    def find_agent_slot(self, kypt, masks):
        # kypt (B, T, 1, H, W)
        # masks(B, T, K, 1, H, W)
        with torch.no_grad():
            K = masks.size(2)
            kypt = torch.stack([kypt] * K, dim=2) # [B, K, T, 1, H, W]
            crit = ((kypt > 0.5) * (masks > 0.5)).sum(-1).sum(-1).sum(-1)
            tot = (masks > 0.5).sum(-1).sum(-1).sum(-1)
            dist = crit / (tot + 1e-5)
            _, si = torch.sort(dist, 2, descending=True)
            si = si.float()
            first = int(si[..., 0].mean().round())
        return first

    def track_agent(self, seq, states, discovery_dropout, global_step=0):
        """ conditioned sequence on ee_poses in the backgrounds; the actions or joint values condition
            The agent layer is segmented through scene mixture models like GENESIS / SPACE.
        """
        B, T, C, H, W = seq.size()
        # Process background
        # first infer the background.
        mixture_out = self.mixture_module.encode(seq, states, global_step)  # T

        # (B, T, >C<, H, W)
        seq_diff = seq - mixture_out['bg']  # only_fg [B, T, C, H, W] - [B, T, C, H, W] ; get explict masked foreground
        inpt = torch.cat([seq, seq_diff], dim=2)  # Input of Eq.(31) of supl.along channel axis ; [B, T, 6, H, W]
        kypt_out = self.keypoint_module.predict_keypoints(seq, states)

        gaussian_heatmap = kypt_out['gaussian_maps'] # gaussian heatmap of the agent
        fg_out = self.obj_module.track(inpt, mixture_out['bg'], discovery_dropout=discovery_dropout,
            z_agent=mixture_out['z_masks'][:, :, self.agent_slot_idx].detach(), # state of the agent
            h_agent=mixture_out['h_masks'][:, :, self.agent_slot_idx].detach(), # history of the agent
            enhanced_act=mixture_out['enhanced_act'].detach(),
            ori_seq=seq)

        # Prepares things to compute reconstruction
        # (B, T, 1, H, W)
        alpha_map = fg_out['alpha_map']
        fg = fg_out['fg']
        bg = mixture_out['bg']

        log = fg_out.copy()
        log.update(mixture_out.copy())
        if ARCH.KEYPOINTS:  # Do keypoint inference?
            log.update(kypt_out.copy())
        log['gaussian_heatmap'] = gaussian_heatmap
        log['seq_diff'] = seq_diff  # To be deleted
        log.update(
            imgs=seq,
            recon=fg + (1 - alpha_map) * bg,
        ) 
        return log

    def generate_agent(self, seq, states, cond_steps, optimizer):
        # generate video sequence conditioned on robot actions into background
        B, T, C, H, W = seq.size()
        mixture_out = self.mixture_module.generate(seq, states, cond_steps)  # we don't generate bg conditioned on action
        seq_diff = seq - mixture_out['bg'] 
        # (B, T, >C<, H, W)
        kypt_out = self.mixture_module.predict_keypoints(
            torch.cat([seq[:, :cond_steps], mixture_out['bg'][:, cond_steps:]], dim=1), states)
        gaussian_heatmap = kypt_out['gaussian_maps'] # keypoint maps
        inpt = torch.cat([seq, seq_diff.detach()], dim=2)  # ! Input of Eq.(31) of supl.along channel axis ; [B, T, 6, H, W]
        obj_out = self.obj_module.generate(inpt, mixture_out['bg'], cond_steps, 
                    z_agent=mixture_out['z_masks'][:, :, self.agent_slot_idx].detach(),
                    h_agent=mixture_out['h_masks'][:, :, self.agent_slot_idx].detach(),
                    enhanced_act=mixture_out['enhanced_act'].detach(),
                    ori_seq=seq)

        alpha_map = obj_out['alpha_map']
        fg = obj_out['fg']
        bg = mixture_out['bg']

        log = obj_out.copy()
        log.update(mixture_out.copy())
        log['gaussian_heatmap'] = gaussian_heatmap
        log['seq_diff'] = seq_diff
        log.update(imgs=seq, recon=fg + (1 - alpha_map) * bg
        )
        return log

    def gaussian_likelihood(self, x, fg, bg, alpha_map, global_step):

        if ARCH.BG_ON and global_step < ARCH.MODULE_TRAINING_SCHEME[1]:
            recon = bg  # bg_only
        else:
            recon = fg + (1. - alpha_map) * bg

        dist = Normal(recon, self.sigma) 
        # (B, T, 3, H, W)
        loglikelihood = dist.log_prob(x)
        # (B,)
        loglikelihood = loglikelihood.flatten(start_dim=1).sum(-1)

        return loglikelihood 