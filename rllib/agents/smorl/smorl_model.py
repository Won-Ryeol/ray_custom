from ssl import ALERT_DESCRIPTION_HANDSHAKE_FAILURE
from numpy.core.getlimits import _discovered_machar
import torch
from torch.cuda import is_available

import numpy as np
from typing import Any, List, Tuple
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.framework import TensorType

#* import reference
torch, nn = try_import_torch()
from torch import distributions as td
from .utils import Linear, TanhBijector, scale_action

# for policy.
from ray.rllib.agents.gswm.utils import Linear, TanhBijector

from .utils import scale_action
# import visualizer of training GSWM.
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

ActFunc = Any

# TODO (chmin): add smorl-related sub modules here.
from ray.rllb.agents.smorl.modules.modules import ImgEncoder, ZWhatEnc, GlimpseDec, BgDecoder, BgEncoder, ConvLSTMEncoder
from ray.rllb.agents.smorl.modules.discovery import ProposalRejectionCell
from ray.rllb.agents.smorl.modules.propagation import PropagationCell
from ray.rllb.agents.smorl.modules.common import *
from ray.rllib.agents.smorl.modules.arch import ARCH


class SMORLModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, agent_slot_idx=0):
        """
            SMORL (https://github.com/martius-lab/SMORL) but no goal-conditioned.
            TODO (chmin): refactoring required after publication.
        """
        super().__init__(obs_space, action_space, num_outputs, 
                model_config, name)

        nn.Module.__init__(self)
        self.action_size = action_space.shape[0] # 8

        self.bg_what_std_bias = 0
        self.no_discovery = ARCH.NO_DISCOVERY

        self.image_enc = ConvLSTMEncoder()

        self.z_what_net = ZWhatEnc()
        self.glimpse_dec_net = GlimpseDec()

        self.propagate_cell = PropagationCell(
                z_what_net=self.z_what_net,
                glimpse_dec_net=self.glimpse_dec_net
        )
        if not ARCH.PHASE_NO_BACKGROUND:
            self.bg_enc = BgEncoder()
            self.bg_dec = BgDecoder()
            self.bg_prior_rnn = nn.GRUCell(ARCH.BG_WHAT_DIM, ARCH.BG_PRIOR_RNN_HID_DIM)
            self.bg_prior_net = nn.Linear(ARCH.BG_PRIOR_RNN_HID_DIM, ARCH.BG_WHAT_DIM * 2)

        self.proposal_rejection_cell = ProposalRejectionCell(
                z_what_net=self.z_what_net,
                glimpse_dec_net=self.glimpse_dec_net
        )

        self.register_buffer('z_pres_disc_threshold', torch.tensor(0.7))
        self.register_buffer('prior_bg_mean_t1', torch.zeros(1))
        self.register_buffer('prior_bg_std_t1', torch.ones(1))
        self.register_buffer('color_t', self.ARCH.COLOR_T)

        self.prior_rnn_init_out = None
        self.prior_rnn_init_hid = None

        self.bg_prior_rnn_init_hid = None
        self.restart = True

    @property
    def p_bg_what_t1(self):
        return Normal(self.prior_bg_mean_t1, self.prior_bg_std_t1)

    def init(self, x):
        bs = x.size(0)
        device = x.device
        temporal_rnn_out_pre = x.new_zeros(bs, 1, ARCH.TEMPORAL_RNN_OUT_DIM).to(device)
        temporal_rnn_hid_pre = x.new_zeros(bs, 1, ARCH.TEMPORAL_RNN_HID_DIM)
        prior_rnn_out_pre = x.new_zeros(bs, 1, ARCH.PRIOR_RNN_OUT_DIM)
        prior_rnn_hid_pre = x.new_zeros(bs, 1, ARCH.PRIOR_RNN_HID_DIM)
        z_what_pre = x.new_zeros(bs, 1, ARCH.Z_WHAT_DIM)
        z_where_pre = x.new_zeros(bs, 1, (ARCH.Z_WHERE_SHIFT_DIM + ARCH.Z_WHERE_SCALE_DIM))
        z_where_bias_pre = x.new_zeros(bs, 1, (ARCH.Z_WHERE_SHIFT_DIM + ARCH.Z_WHERE_SCALE_DIM))
        z_depth_pre = x.new_zeros(bs, 1, ARCH.Z_DEPTH_DIM)
        z_pres_pre = x.new_zeros(bs, 1, ARCH.Z_PRES_DIM)
        cumsum_one_minus_z_pres_prop_pre = x.new_zeros(bs, 1, ARCH.Z_PRES_DIM)
        ids_pre = x.new_zeros(bs, 1)

        lengths = x.new_zeros(bs)
        i = 0 
        self.log_disc_list = []
        self.log_prop_list = []
        self.scalor_log_list = []
        self.counting_list = []
        bg_rnn_hid_pre = self.initial_bg_prior_rnn_hid(device).expand(bs, -1)
        self.image_enc.reset()
        self._state = {"z_what_pre": z_what_pre,
                        "bg_rnn_hid_pre": bg_rnn_hid_pre,
                        "z_where_pre": z_where_pre,
                        "z_depth_pre": z_depth_pre,
                        "z_pres_pre": z_pres_pre,
                        "temporal_rnn_out_pre": temporal_rnn_out_pre,
                        "temporal_rnn_hid_pre": temporal_rnn_hid_pre,
                        "prior_rnn_out_pre": prior_rnn_out_pre,
                        "prior_rnn_hid_pre": prior_rnn_hid_pre,
                        "cumsum_one_minus_z_pres_prop_pre": cumsum_one_minus_z_pres_prop_pre,
                        "z_where_bias_pre": z_where_bias_pre,
                        "ids_pre": ids_pre,
                        "lengths": lengths,
                        "i": 0}

    def initial_temporal_rnn_hid(self, device):
        return torch.zeros((1, temporal_rnn_out_dim)).to(device), \
               torch.zeros((1, temporal_rnn_hid_dim)).to(device)

    def initial_prior_rnn_hid(self, device):
        if self.prior_rnn_init_out is None or self.prior_rnn_init_hid is None:
            self.prior_rnn_init_out = torch.zeros(1, prior_rnn_out_dim).to(device)
            self.prior_rnn_init_hid = torch.zeros(1, prior_rnn_hid_dim).to(device)

        return self.prior_rnn_init_out, self.prior_rnn_init_hid

    def initial_bg_prior_rnn_hid(self, device):
        if self.bg_prior_rnn_init_hid is None:
            self.bg_prior_rnn_init_hid = torch.zeros(1, bg_prior_rnn_hid_dim).to(device)

        return self.bg_prior_rnn_init_hid

    def reset(self):
        # should trigger init() on next one_step call
        if hasattr(self, "_state"):
            delattr(self, "_state") 

    def one_step(self, x, action, img_enc=None, eps=1e-15):
        if img_enc is None: 
            img_enc = self.image_enc.one_step(x, action)
        bs = x.size(0)
        device = x.device
        if not hasattr(self,"_state"):
            self.init(x)
        z_what_pre = self._state["z_what_pre"]
        z_where_pre = self._state["z_where_pre"]
        z_depth_pre = self._state["z_depth_pre"]
        z_pres_pre = self._state["z_pres_pre"]
        ids_pre = self._state["ids_pre"]
        temporal_rnn_out_pre = self._state["temporal_rnn_out_pre"]
        temporal_rnn_hid_pre = self._state["temporal_rnn_hid_pre"]
        prior_rnn_out_pre = self._state["prior_rnn_out_pre"]
        prior_rnn_hid_pre = self._state["prior_rnn_hid_pre"]
        cumsum_one_minus_z_pres_prop_pre = self._state["cumsum_one_minus_z_pres_prop_pre"]
        z_where_bias_pre = self._state["z_where_bias_pre"]
        lengths = self._state["lengths"]
        i = self._state["i"]

        kl_z_what_prop = x.new_zeros(bs)
        kl_z_where_prop = x.new_zeros(bs)
        kl_z_depth_prop = x.new_zeros(bs)
        kl_z_pres_prop = x.new_zeros(bs)
        log_imp_prop = x.new_zeros(bs)
        log_prop = None

        n_objects_to_propagate = lengths.max()
        if n_objects_to_propagate != 0:

            max_length = int(torch.max(lengths))

            y_each_obj_prop, alpha_map_prop, importance_map_prop, z_what_prop, z_where_prop, \
            z_where_bias_prop, z_depth_prop, z_pres_prop, ids_prop, kl_z_what_prop, kl_z_where_prop, \
            kl_z_depth_prop, kl_z_pres_prop, temporal_rnn_out, temporal_rnn_hid, prior_rnn_out, prior_rnn_hid, \
            cumsum_one_minus_z_pres_prop, log_imp_prop, log_prop, representation_prop, only_prop_representation = \
                self.propagate_cell(
                    x, img_enc, temporal_rnn_out_pre, temporal_rnn_hid_pre, prior_rnn_out_pre, prior_rnn_hid_pre,
                    z_what_pre, z_where_pre, z_where_bias_pre, z_depth_pre, z_pres_pre,
                    cumsum_one_minus_z_pres_prop_pre, ids_pre, lengths, max_length, i, no_disc=self.no_discovery, eps=eps
                )
        else:
            z_what_prop = x.new_zeros(bs, 1, ARCH.Z_WHAT_DIM)
            z_where_prop = x.new_zeros(bs, 1, (ARCH.Z_WHERE_SCALE_DIM + ARCH.Z_WHERE_SHIFT_DIM))
            z_where_bias_prop = x.new_zeros(bs, 1, (ARCH.Z_WHERE_SCALE_DIM + ARCH.Z_WHERE_SHIFT_DIM))
            z_depth_prop = x.new_zeros(bs, 1, ARCH.Z_DEPTH_DIM)
            z_pres_prop = x.new_zeros(bs, ARCH.Z_PRES_DIM)
            cumsum_one_minus_z_pres_prop = x.new_zeros(bs, 1, ARCH.Z_PRES_DIM)
            y_each_obj_prop = x.new_zeros(bs, 1, 3, ARCH.IMG_H, ARCH.IMG_W)
            alpha_map_prop = x.new_zeros(bs, 1, 1, ARCH.IMG_H, ARCH.IMG_W)
            importance_map_prop = x.new_zeros(bs, 1, 1, ARCH.IMG_H, ARCH.IMG_W)
            ids_prop = x.new_zeros(bs, 1)
            only_prop_representation = {"z_where": x.new_zeros(1, (ARCH.Z_WHERE_SCALE_DIM + ARCH.Z_WHERE_SHIFT_DIM)),
                "z_what": x.new_zeros(1, ARCH.Z_WHAT_DIM), "z_depth": x.new_zeros(1, ARCH.Z_DEPTH_DIM)}

        alpha_map_prop_sum = alpha_map_prop.sum(1)
        alpha_map_prop_sum = \
            alpha_map_prop_sum + (alpha_map_prop_sum.clamp(eps, 1 - eps) - alpha_map_prop_sum).detach()
        y_each_obj_disc, alpha_map_disc, importance_map_disc, \
        z_what_disc, z_where_disc, z_where_bias_disc, z_depth_disc, \
        z_pres_disc, ids_disc, kl_z_what_disc, kl_z_where_disc, \
        kl_z_pres_disc, kl_z_depth_disc, log_imp_disc, log_disc, representation_disc = \
            self.proposal_rejection_cell(
                x, img_enc, alpha_map_prop_sum, ids_prop, lengths, i, no_disc=self.no_discovery, eps=eps
            )
        importance_map = torch.cat((importance_map_prop, importance_map_disc), dim=1)

        importance_map_norm = importance_map / (importance_map.sum(dim=1, keepdim=True) + eps)

        # (bs, 1, img_h, img_w)
        alpha_map = torch.cat((alpha_map_prop, alpha_map_disc), dim=1).sum(dim=1)

        alpha_map = alpha_map + (alpha_map.clamp(eps, 1 - eps) - alpha_map).detach()

        y_each_obj = torch.cat((y_each_obj_prop, y_each_obj_disc), dim=1)

        y_nobg = (y_each_obj.view(bs, -1, 3, ARCH.IMG_H, ARCH.IMG_W) * importance_map_norm).sum(dim=1)


        if i == 0:
            p_bg_what = self.p_bg_what_t1
        else:
            bg_what_pre = self._state["bg_what_pre"]
            bg_rnn_hid_pre = self._state["bg_rnn_hid_pre"]
            bg_rnn_hid_pre = self.bg_prior_rnn(bg_what_pre, bg_rnn_hid_pre)
            self._state["bg_rnn_hid_pre"] = bg_rnn_hid_pre
            # bg_rnn_hid_pre = self.layer_norm_h(bg_rnn_hid_pre)
            p_bg_what_mean_bias, p_bg_what_std = self.bg_prior_net(bg_rnn_hid_pre).chunk(2, -1)
            p_bg_what_mean = p_bg_what_mean_bias + bg_what_pre
            p_bg_what_std = F.softplus(p_bg_what_std + self.bg_what_std_bias)
            p_bg_what = Normal(p_bg_what_mean, p_bg_what_std)


        x_alpha_cat = torch.cat((x, (1 - alpha_map)), dim=1)
        # Background
        z_bg_mean, z_bg_std = self.bg_enc(x_alpha_cat)
        z_bg_std = F.softplus(z_bg_std + self.bg_what_std_bias)
        if ARCH.PHASE_GENERATE and i >= ARCH.OBSERVE_FRAMES:
            q_bg = p_bg_what
        else:
            q_bg = Normal(z_bg_mean, z_bg_std)
        z_bg = q_bg.rsample()
        # bg, one_minus_alpha_map = self.bg_dec(z_bg)
        bg = self.bg_dec(z_bg)

        bg_what_pre = z_bg
        self._state["bg_what_pre"] = bg_what_pre

        y = y_nobg + (1 - alpha_map) * bg

        p_x_z = Normal(y.flatten(1), ARCH.SIGMA)
        log_like = p_x_z.log_prob(x.view(-1, 3, ARCH.IMG_H, ARCH.IMG_W).
                                    expand_as(y).flatten(1)).sum(-1)  # sum image dims (C, H, W)


        ########################################### Compute log importance ############################################
        if not self.training and ARCH.PHASE_NLL:
            # (bs, dim)
            log_imp_bg = (p_bg_what.log_prob(z_bg) - q_bg.log_prob(z_bg)).sum(1)

        ######################################## End of Compute log importance #########################################
        kl_z_bg = kl_divergence(q_bg, p_bg_what).sum(1)
        kl_z_pres = kl_z_pres_disc + kl_z_pres_prop
        kl_z_what = kl_z_what_disc + kl_z_what_prop
        kl_z_where = kl_z_where_disc + kl_z_where_prop
        kl_z_depth = kl_z_depth_disc + kl_z_depth_prop
        if not self.training and ARCH.PHASE_NLL:
            log_imp = log_imp_disc + log_imp_prop + log_imp_bg
        else:
            log_imp = None

        prior_rnn_out_init, prior_rnn_hid_init = self.initial_prior_rnn_hid(device)
        temporal_rnn_out_init, temporal_rnn_hid_init = self.initial_temporal_rnn_hid(device)

        new_prior_rnn_out_init = prior_rnn_out_init.unsqueeze(0). \
            expand((bs, z_what_disc.size(1), prior_rnn_out_dim))
        new_prior_rnn_hid_init = prior_rnn_hid_init.unsqueeze(0). \
            expand((bs, z_what_disc.size(1), prior_rnn_hid_dim))
        new_temporal_rnn_out_init = temporal_rnn_out_init.unsqueeze(0). \
            expand((bs, z_what_disc.size(1), temporal_rnn_out_dim))
        new_temporal_rnn_hid_init = temporal_rnn_hid_init.unsqueeze(0). \
            expand((bs, z_what_disc.size(1), temporal_rnn_hid_dim))

        if lengths.max() != 0:
            representation = {}
            for key in representation_prop.keys():
                z_prop = representation_prop[key]
                z_disc = representation_disc[key]
                representation[key] = torch.cat((z_prop, z_disc), dim=1) 

            z_what_prop_disc = torch.cat((z_what_prop, z_what_disc), dim=1)
            z_where_prop_disc = torch.cat((z_where_prop, z_where_disc), dim=1)
            z_where_bias_prop_disc = torch.cat((z_where_bias_prop, z_where_bias_disc), dim=1)
            z_depth_prop_disc = torch.cat((z_depth_prop, z_depth_disc), dim=1)
            z_pres_prop_disc = torch.cat((z_pres_prop, z_pres_disc), dim=1)
            z_mask_prop_disc = torch.cat((
                (z_pres_prop > 0).float(),
                (z_pres_disc > self.z_pres_disc_threshold).float()
            ), dim=1)
            temporal_rnn_out_prop_disc = torch.cat((temporal_rnn_out, new_temporal_rnn_out_init), dim=1)
            temporal_rnn_hid_prop_disc = torch.cat((temporal_rnn_hid, new_temporal_rnn_hid_init), dim=1)
            prior_rnn_out_prop_disc = torch.cat((prior_rnn_out, new_prior_rnn_out_init), dim=1)
            prior_rnn_hid_prop_disc = torch.cat((prior_rnn_hid, new_prior_rnn_hid_init), dim=1)
            cumsum_one_minus_z_pres_prop_disc = torch.cat([cumsum_one_minus_z_pres_prop,
                                                            x.new_zeros(bs, z_what_disc.size(1), z_pres_dim)],
                                                            dim=1)
            ids_prop_disc = torch.cat((ids_prop, ids_disc), dim=1)
        else:
            representation = representation_disc
            z_what_prop_disc = z_what_disc
            z_where_prop_disc = z_where_disc
            z_where_bias_prop_disc = z_where_bias_disc
            z_depth_prop_disc = z_depth_disc
            z_pres_prop_disc = z_pres_disc
            temporal_rnn_out_prop_disc = new_temporal_rnn_out_init
            temporal_rnn_hid_prop_disc = new_temporal_rnn_hid_init
            prior_rnn_out_prop_disc = new_prior_rnn_out_init
            prior_rnn_hid_prop_disc = new_prior_rnn_hid_init
            z_mask_prop_disc = (z_pres_disc > self.z_pres_disc_threshold).float()
            cumsum_one_minus_z_pres_prop_disc = x.new_zeros(bs, z_what_disc.size(1), z_pres_dim)
            ids_prop_disc = ids_disc

        num_obj_each = torch.sum(z_mask_prop_disc, dim=1)
        max_num_obj = int(torch.max(num_obj_each).item())
        if ARCH.USE_DISC:
            final_representation = {"z_what": x.new_zeros(bs, max_num_obj, z_what_dim), 
                                    "z_where": x.new_zeros(bs, max_num_obj, z_where_scale_dim + z_where_shift_dim), 
                                    "z_depth": x.new_zeros(bs, max_num_obj, z_depth_dim)}
        z_what_pre = x.new_zeros(bs, max_num_obj, z_what_dim)
        z_where_pre = x.new_zeros(bs, max_num_obj, z_where_scale_dim + z_where_shift_dim)
        z_where_bias_pre = x.new_zeros(bs, max_num_obj, z_where_scale_dim + z_where_shift_dim)
        z_depth_pre = x.new_zeros(bs, max_num_obj, z_depth_dim)
        z_pres_pre = x.new_zeros(bs, max_num_obj, z_pres_dim)
        z_mask_pre = x.new_zeros(bs, max_num_obj, z_pres_dim)
        temporal_rnn_out_pre = x.new_zeros(bs, max_num_obj, temporal_rnn_out_dim)
        temporal_rnn_hid_pre = x.new_zeros(bs, max_num_obj, temporal_rnn_hid_dim)
        prior_rnn_out_pre = x.new_zeros(bs, max_num_obj, prior_rnn_out_dim)
        prior_rnn_hid_pre = x.new_zeros(bs, max_num_obj, prior_rnn_hid_dim)
        cumsum_one_minus_z_pres_prop_pre = x.new_zeros(bs, max_num_obj, z_pres_dim)
        ids_pre = x.new_zeros(bs, max_num_obj)

        for b in range(bs):
            num_obj = int(num_obj_each[b])

            idx = z_mask_prop_disc[b].nonzero()[:, 0]
            if ARCH.USE_DISC:
                for key in final_representation.keys():
                    final_representation[key][b, :num_obj] = representation[key][b, idx]
            z_what_pre[b, :num_obj] = z_what_prop_disc[b, idx]
            z_where_pre[b, :num_obj] = z_where_prop_disc[b, idx]
            z_where_bias_pre[b, :num_obj] = z_where_bias_prop_disc[b, idx]
            z_depth_pre[b, :num_obj] = z_depth_prop_disc[b, idx]
            z_pres_pre[b, :num_obj] = z_pres_prop_disc[b, idx]
            z_mask_pre[b, :num_obj] = z_mask_prop_disc[b, idx]
            temporal_rnn_out_pre[b, :num_obj] = temporal_rnn_out_prop_disc[b, idx]
            temporal_rnn_hid_pre[b, :num_obj] = temporal_rnn_hid_prop_disc[b, idx]
            prior_rnn_out_pre[b, :num_obj] = prior_rnn_out_prop_disc[b, idx]
            prior_rnn_hid_pre[b, :num_obj] = prior_rnn_hid_prop_disc[b, idx]
            cumsum_one_minus_z_pres_prop_pre[b, :num_obj] = cumsum_one_minus_z_pres_prop_disc[b, idx]
            ids_pre[b, :num_obj] = ids_prop_disc[b, idx]
        if not ARCH.PHASE_DO_REMOVE_DETACH or self.global_step < ARCH.REMOVE_DETACH_STEP:
            z_what_pre = z_what_pre.detach()
            z_where_pre = z_where_pre.detach()
            z_depth_pre = z_depth_pre.detach()
            z_pres_pre = z_pres_pre.detach()
            z_mask_pre = z_mask_pre.detach()
            temporal_rnn_out_pre = temporal_rnn_out_pre.detach()
            temporal_rnn_hid_pre = temporal_rnn_hid_pre.detach()
            prior_rnn_out_pre = prior_rnn_out_pre.detach()
            prior_rnn_hid_pre = prior_rnn_hid_pre.detach()
            cumsum_one_minus_z_pres_prop_pre = cumsum_one_minus_z_pres_prop_pre.detach()
            z_where_bias_pre = z_where_bias_pre.detach()
        lengths = torch.sum(z_mask_pre, dim=(1, 2)).view(bs)
        self._state["z_what_pre"] = z_what_pre
        self._state["bg_what_pre"] = bg_what_pre
        self._state["z_where_pre"] = z_where_pre
        self._state["z_depth_pre"] = z_depth_pre
        self._state["z_pres_pre"] = z_pres_pre
        self._state["z_mask_pre"] = z_mask_pre
        self._state["temporal_rnn_out_pre"] = temporal_rnn_out_pre
        self._state["temporal_rnn_hid_pre"] = temporal_rnn_hid_pre
        self._state["prior_rnn_out_pre"] = prior_rnn_out_pre
        self._state["prior_rnn_hid_pre"] = prior_rnn_hid_pre
        self._state["cumsum_one_minus_z_pres_prop_pre"] = cumsum_one_minus_z_pres_prop_pre
        self._state["z_where_bias_pre"] = z_where_bias_pre
        self._state["lengths"] = lengths
        self._state["ids_pre"] = ids_pre
        self._state["i"] = i+1
        scalor_step_log = {}
        if ARCH.LOG_PHASE:
            if ids_prop_disc.size(1) < importance_map_norm.size(1):
                ids_prop_disc = torch.cat((x.new_zeros(ids_prop_disc[:, 0:1].size()), ids_prop_disc), dim=1)
            id_color = self.color_t[ids_prop_disc.view(-1).long() % ARCH.COLOR_NUM]

            # (bs, num_obj_prop + num_cell_h * num_cell_w, 3, 1, 1)
            id_color = id_color.view(bs, -1, 3, 1, 1)
            # (bs, num_obj_prop + num_cell_h * num_cell_w, 3, img_h, img_w)
            id_color_map = (torch.cat((alpha_map_prop, alpha_map_disc), dim=1) > .3).float() * id_color
            mask_color = (id_color_map * importance_map_norm.detach()).sum(dim=1)
            x_mask_color = x - 0.7 * (alpha_map > .3).float() * (x - mask_color)
            scalor_step_log = {
                'y_each_obj': y_each_obj.cpu().detach(),
                'importance_map_norm': importance_map_norm.cpu().detach(),
                'importance_map': importance_map.cpu().detach(),
                'bg': bg.cpu().detach(),
                'alpha_map': alpha_map.cpu().detach(),
                'x_mask_color': x_mask_color.cpu().detach(),
                'mask_color': mask_color.cpu().detach(),
                'p_bg_what_mean': p_bg_what_mean.cpu().detach() if i > 0 else self.p_bg_what_t1.mean.cpu().detach(),
                'p_bg_what_std': p_bg_what_std.cpu().detach() if i > 0 else self.p_bg_what_t1.stddev.cpu().detach(),
                'z_bg_mean': z_bg_mean.cpu().detach(),
                'z_bg_std': z_bg_std.cpu().detach(),
            }
            if log_disc:
                for k, v in log_disc.items():
                    log_disc[k] = v.cpu().detach()
            if log_prop:
                for k, v in log_prop.items():
                    log_prop[k] = v.cpu().detach()
        self.log_disc_list.append(log_disc)
        self.log_prop_list.append(log_prop)
        self.scalor_log_list.append(scalor_step_log)
        self.counting_list.append(lengths)
        if ARCH.USE_DISC:
            for key in final_representation.keys():
                final_representation[key] = final_representation[key][0]
            if final_representation["z_what"].shape[0] == 0:
                final_representation = {"z_where": x.new_zeros(1, (z_where_scale_dim + z_where_shift_dim)), "z_what": x.new_zeros(1, z_what_dim), "z_depth": x.new_zeros(1, z_depth_dim)}
        else:
            final_representation = only_prop_representation

        return kl_z_bg, kl_z_pres, kl_z_what, kl_z_where, kl_z_depth, log_imp, log_like, y, final_representation

    @torch.no_grad()
    def policy(self, 
            obs: TensorType,
            state: List[TensorType],
            explore=True,
            start_infer_flag=False
            ) -> Tuple[TensorType, List[float], List[TensorType]]:
        """
        # TODO (chmin): define policy inference for SMORL.S
        Returns the action. Runs through the encoder, recurrent model, and policyMin
        to obtain action.
        The forward inference of the agent.
        Args:
            obs: [B, 3, H, W]
            state: length 15 list of detr and sto states. [
                z_masks, z_comps, h_masks, c_masks, h_comps, c_comps,
                z_pres, z_depth, z_where, z_what, z_dyna, h_objs, c_objs,
                ids, action
            ]
            start_infer_flag: whehter the policy inference has started. i.e. prefill steps -> inference
        """
        # [mean, std, sto, detr., action], of shape [B, D] or [B, Z] or [B, A] 
        if len(state[0].size()) == 4: 
            state = [torch.squeeze(t, 0) for t in state]
            self.episodic_step = 0
            state[-3] = state[-3].long() # exception handling

        if len(obs.size()) == 5:
            obs = obs.squeeze(0)

        if state is None: #! get_initial_state is called outside of this method.
            self.state = self.get_initial_state() 
        else:
            self.state = state
        
        post_ctx = state[:3] # z_ctx, h_ctx, c_ctx
        post_obj = state[3:-1] # z_objs, (where, what, ...). h_c_objs, ids
        action = state[-1] # [B, A]
        action = scale_action(action)
        # posterior inference from RNN states.
        is_first_obs = not self.episodic_step or start_infer_flag 

        # TODO (chmin): "infer" of bgmodule should be implemented.
        bg_out = self.bg_module.infer(history=post_ctx, obs=obs, action=action, 
            episodic_step=self.episodic_step, first=is_first_obs)

        obs_diff = obs - bg_out['bg'] # [1, 3, 64, 64]
        inpt = torch.cat([obs, obs_diff], dim=1) # channel-wise concat

        fg_out = self.fg_module.infer(history=post_obj, obs=inpt, bg=bg_out['bg'],
            discovery_dropout=ARCH.DISCOVERY_DROPOUT,
            enhanced_act=bg_out['enhanced_act'].detach(),
            first=is_first_obs,
            episodic_step=self.episodic_step
        )
        
        feat = self.get_feature_for_agent(bg_out['z_ctx'], torch.cat(fg_out['z_objs'], -1))

        if self.global_step < ARCH.JOINT_TRAIN_GSWM_START:
            action = 2.0 * torch.rand(1, self.action_space.shape[0]) - 1.0
            if action[0, 2] > 0 and self.episodic_step < 80:
                action[0, 2] = - action[0, 2] if torch.rand(1) > 0.25 else action[0, 2]
            if action[0, 0] < 0 and self.episodic_step < 80:
                action[0, 0] = - action[0, 0] if torch.rand(1) > 0.25 else action[0, 0]
            logp = torch.zeros((1,), dtype=torch.float32)
        else:
            if ARCH.AGENT_TYPE == 'model_based':
                if self.episodic_step % ARCH.HIGH_LEVEL_HORIZON == 0:
                    action_dist_high = self.actor_high(feat) # [B, N] -> N number of objects
                    # sample by straight-through gradient.
                    sub_policy_idx_inpt = action_dist_high.gsample() # in the form of [1, 0, ...]
                    sub_policy_idx = torch.multinomial(sub_policy_idx_inpt, 1)[0].long() # subpolicy idx in number
                    setattr(self, "sub_policy_idx", sub_policy_idx)
                    setattr(self, "sub_policy_idx_inpt", sub_policy_idx_inpt)

                indiv_feats_list = self.get_indiv_features(bg_out['z_ctx'], torch.cat(fg_out['z_objs'], -1))

                # feature for low-level policy inferred by high-level actor.
                low_level_feat = torch.cat([indiv_feats_list[self.sub_policy_idx], self.sub_policy_idx_inpt],dim=-1)
                actor_low_dist = self.actor_low(low_level_feat)
                if explore:
                    action = actor_low_dist.sample()
                else:
                    action = actor_low_dist.mean
                logp = actor_low_dist.log_prob(action)
            elif ARCH.AGENT_TYPE == 'model_free': # model-free agent. Do not leverage latent dynamics.
                raise ValueError("Model-free version of GSWM is not supported yet!")

        self.state = [ bg_out['z_ctx'], bg_out['h_ctx_post'][None],
            bg_out['c_ctx_post'][None], * fg_out['z_objs'], * fg_out['h_c_objs'], fg_out['ids']] + [action]

        self.episodic_step += 1 # episodic increment
        return action, logp, self.state

    def get_feature_for_agent(self, z_ctx, z_objs):

        # process agent-obj depth and position
        obj_position = z_objs[..., ARCH.Z_PRES_DIM + ARCH.Z_DEPTH_DIM:ARCH.Z_PRES_DIM 
            + ARCH.Z_DEPTH_DIM + ARCH.Z_WHERE_DIM][..., 2:] # [B, (T), N, 2]
        obj_depth =  z_objs[..., ARCH.Z_PRES_DIM:ARCH.Z_PRES_DIM + ARCH.Z_DEPTH_DIM] # [B, (T), N, 2]
        obj_depth = torch.sigmoid(-obj_depth)  

        obj_scale = z_objs[..., ARCH.Z_PRES_DIM + ARCH.Z_DEPTH_DIM:ARCH.Z_PRES_DIM 
            + ARCH.Z_DEPTH_DIM + ARCH.Z_WHERE_DIM][..., :2]  # [B, T, N, 2]
        obj_what = z_objs[..., ARCH.Z_PRES_DIM + ARCH.Z_DEPTH_DIM + ARCH.Z_WHERE_DIM:ARCH.Z_PRES_DIM 
            + ARCH.Z_DEPTH_DIM + ARCH.Z_WHERE_DIM + ARCH.Z_WHAT_DIM] # [B, T, N, 16]
        
        cat_latents = [z_ctx, obj_depth, obj_position, obj_what] # [B, T, *]
        if len(z_ctx.size()) == 4:
            B, T, _ = z_ctx.size()
            _, _, N, _ = z_objs.size()
            latents_to_cat = [l.reshape(B, T, -1) for l in cat_latents]
            return torch.cat(latents_to_cat, dim=-1)

        else: #! used in policy inference
            B, K, _ = z_ctx.size()
            latents_to_cat = [l.reshape(B, -1) for l in cat_latents]
            return torch.cat(latents_to_cat, dim=-1)

    def get_indiv_features(self, z_ctx, z_objs):
            """
                Constructs feature for input to reward, decoder, actor and critic.
                inputs consist of posterior history and stochastic latents of the scene.        
            """

            # process agent-obj depth and position
            obj_position = z_objs[..., ARCH.Z_PRES_DIM + ARCH.Z_DEPTH_DIM:ARCH.Z_PRES_DIM 
                + ARCH.Z_DEPTH_DIM + ARCH.Z_WHERE_DIM][..., 2:] # [B, (T), N, 2]
            obj_depth =  z_objs[..., ARCH.Z_PRES_DIM:ARCH.Z_PRES_DIM + ARCH.Z_DEPTH_DIM] # [B, (T), N, 2]
            obj_depth = torch.sigmoid(-obj_depth)  

            obj_scale = z_objs[..., ARCH.Z_PRES_DIM + ARCH.Z_DEPTH_DIM:ARCH.Z_PRES_DIM 
                    + ARCH.Z_DEPTH_DIM + ARCH.Z_WHERE_DIM][..., :2]  # [B, T, N, 2]
            obj_what = z_objs[..., ARCH.Z_PRES_DIM + ARCH.Z_DEPTH_DIM + ARCH.Z_WHERE_DIM:ARCH.Z_PRES_DIM 
                    + ARCH.Z_DEPTH_DIM + ARCH.Z_WHERE_DIM + ARCH.Z_WHAT_DIM] # [B, T, N, 16]
            # obj pres should be a scale parameter.

            latent_lists = []
            obj_indices = {ind for ind in range(ARCH.MAX)}

            for idx in range(ARCH.MAX):
                other_objs = list(obj_indices - {idx})

                obj_wise_latents = [z_ctx, 
                    obj_depth[..., idx, :], obj_position[..., idx, :], obj_what[..., idx, :],
                    obj_position[..., idx, :][..., None, :] - obj_position[..., other_objs, :]
                ]
                
                if len(z_ctx.size()) == 4:
                    B, T, K, _ = z_ctx.size()
                    _, _, N, _ = z_objs.size()
                    latents_to_cat = [l.reshape(B, T, -1) for l in obj_wise_latents]
                else: #! used in policy inference
                    B, K, _ = z_ctx.size()
                    # squash the entity dimension [B, K * (Hm / Hc / Zm / Zc)]
                    # [1, K * Zm + K * Zc + N * Z(w+w+p...) } + K * Hm + K * Hc + H(w+w+p...)]
                    latents_to_cat = [l.reshape(B, -1) for l in obj_wise_latents]

                obj_wise_latents =  torch.cat(latents_to_cat, dim=-1)
                latent_lists.append(obj_wise_latents)

            return latent_lists

    def imagine_ahead(self, deter_states, sto_states, imagine_horizon, action, raw_action):
        """ Generate imagined frames given the RNN state of masks and comps.
            Args:
                cond_obs: NOTE: we do not need this...
                deter_states:
                sto_states:
                imagine_horizon:
        NOTE that the RNN states should be that of prior.
            actor: the policy network.
            action: action from the last timestep. NOTE that it's enhanced.
        """
        H = imagine_horizon

        h_mask_t_prior, *_ = deter_states

        B, T, K, _ = h_mask_t_prior.size()
        sto_start = []
        for s in sto_states:
            s = s.contiguous().detach() #? should it be contiguous
            shape = [-1] + list(s.size())[2:]
            sto_start.append(s.view(*shape)) # chunk batch and time axes [B, T, K, D]

        deter_start = []
        for d in deter_states:
            d = d.contiguous().detach() #? should it be contiguous
            shape = [-1] + list(d.size())[2:]
            deter_start.append(d.view(*shape)) # chunk batch and time axes [B, T, K, D]
        action = action.detach().reshape(B*T, -1)
        raw_action = raw_action.detach().reshape(B*T, -1)

        # we need to add next_state function, since no action is provided in priori.
        def next_state(states, action, raw_action, cur_horizon):
            """
                First, given (z_t, h_t), infer a_t from policy. 
                Given a_t and (z_t, h_t), update the RNN state to h_{t+1}. 
                Return the updated RNN state
                cur_horizon: current step of the imagination horizon. Necessary for 
                        keypoint inference.
                NOTE: that cur_horizon determines how cur_horizon is related to cond_steps.
            """
            # given h_t and z_t.
            deter_states, sto_states = states # have the shapes of [B*T, *]
            # feature given 
            h_ctx_prior, c_ctx_prior, h_objs_prior, c_objs_prior = deter_states
            z_ctx_prior, z_objs_prior, ids_prior, proposal = sto_states
            # split the latents into (z_pres, z_depth, z_where, z_what, z_dyna)
            del deter_states
            del sto_states

            # z_pres: 1, z_depth: 1, z_where: 4 -> fixed & Z_WHAT_DIM and Z_DYNA_DIM are adjustable
            z_pres_prior, z_depth_prior, z_where_prior = z_objs_prior[..., :1], z_objs_prior[..., 1:2], z_objs_prior[..., 2:6]
            z_what_prior, z_dyna_prior = z_objs_prior[..., 6:6 + ARCH.Z_WHAT_DIM], z_objs_prior[..., 6 + ARCH.Z_WHAT_DIM:]

            # get the feature for the policy f_t = [h_t | z_t]
            features, = self.get_feature_for_agent(
                z_ctx=z_ctx_prior, z_objs=z_objs_prior
            ).detach(), # [B*T, D]

            # let's make sure that the horizon is at least 12 steps.
            if cur_horizon % ARCH.HIGH_LEVEL_HORIZON == 0:
                action_dist_high = self.actor_high(features) # [B, N] -> N number of objects
                sub_policy_ind_inpt = action_dist_high.gsample() # in the form of [1, 0, ...]
                sub_policy_ind = torch.multinomial(sub_policy_ind_inpt, 1) # subpolicy idx in number
                setattr(self, "sub_policy_ind", sub_policy_ind.long().squeeze())
                setattr(self, "sub_policy_ind_inpt", sub_policy_ind_inpt)

            with torch.no_grad():
                indiv_feats_list = self.get_indiv_features(
                    z_ctx=z_ctx_prior, z_objs=z_objs_prior
                ) # [B*T, D]

            low_level_feat_list = []
            for batch_idx in range(features.size(0)): # iterate for B*T dim.
                sub_policy_idx = self.sub_policy_ind[batch_idx]
                low_level_feat = torch.cat([indiv_feats_list[sub_policy_idx][batch_idx], 
                    self.sub_policy_ind_inpt[batch_idx]
                ], dim=-1) # [D]
                low_level_feat_list.append(low_level_feat)
            low_level_feat = torch.stack(low_level_feat_list, dim=0) # [B, D]
            actor_low_dist = self.actor_low(low_level_feat)
            # infer action a_t ~ \pi(a_t | f_t)
            action = actor_low_dist.rsample()
            low_actor_entropy = actor_low_dist.base_dist.base_dist.entropy()
            action = scale_action(action)

            pre_act_action = self.actor_low.get_pre_activation(low_level_feat)
            del features
            del indiv_feats_list

            bg_out = self.bg_module.imagine(
                history=(h_ctx_prior, c_ctx_prior),
                action=action, # action from the previous step
                z_prevs=z_ctx_prior,
                episodic_step=cur_horizon
                ) #

            fg_out = self.fg_module.imagine(
                mix=bg_out['bg'].detach(), # [B*T, 3, 64, 64], object motion is deterministic.
                history=(h_objs_prior, c_objs_prior), # [B*T, N, D]
                z_prop=((z_pres_prior, z_depth_prior, z_where_prior, z_what_prior, z_dyna_prior),
                         ids_prior, proposal) # [B*T, N,]
                )

            deter_states = (
                bg_out['h_prior'], bg_out['c_prior'], fg_out['h_c_objs'][0], fg_out['h_c_objs'][1]
            )

            sto_states = (bg_out['z_ctx'], fg_out['z_objs'],
                fg_out['ids'], fg_out['proposal'])

            del bg_out
            del fg_out

            return deter_states, sto_states, pre_act_action, action

        # execute imagination rollout.
        sto_last, deter_last = sto_start, deter_start

        # (Will be) len 5 list of lists of len H trajectories; z_masks, z_comps, z_objs, ids, proposal
        sto_outputs = [[] for s in range(len(sto_start))]

        # (Will be) len 6 list of lists of len H trajectories; h_masks, c_masks, h_comps, c_comps, h_objs, c_objs
        deter_outputs = [[] for d in range(len(deter_start))]
        preact_action_list = [] # list of policy output tensors to regularize.
        raw_action_list = [] # list of policy output tensors to regularize.
        sub_policy_idx_list = []
        sub_policy_ind_inpt_list = []
        low_actor_entropy_list = []
        for h in range(H): # imagination horizon h is necessary for the first keypoint
            deter_last, sto_last, preact_action, raw_action, sub_policy_idx, sub_policy_ind_inpt, \
                low_actor_entropy = next_state(states=(deter_last, sto_last), action=action, 
            raw_action=raw_action, cur_horizon=h)
            [so.append(sl) for sl, so in zip(sto_last, sto_outputs)]
            [do.append(dl) for dl, do in zip(deter_last, deter_outputs)]
            preact_action_list.append(preact_action)
            raw_action_list.append(raw_action)
            sub_policy_idx_list.append(sub_policy_idx)
            sub_policy_ind_inpt_list.append(sub_policy_ind_inpt)
            low_actor_entropy_list.append(low_actor_entropy)

        # stack or concat into the shape [H, B*T, ...]
        sto_outputs = [torch.stack(so, dim=0) for so in sto_outputs]
        deter_outputs = [torch.stack(do, dim=0) for do in deter_outputs]
        preact_action = torch.cat(preact_action_list, dim=0) # [B * T * H, A'], A' is action dim before tanh squashing
        raw_action = torch.cat(raw_action_list, dim=0) # [B * T * H, A'], A' is action dim before tanh squashing
        sub_policy_idx = torch.stack(sub_policy_idx_list, dim=0) #  [B * T * H, 1]
        sub_policy_ind_inpt = torch.stack(sub_policy_ind_inpt_list, dim=0) #  [B * T * H, ARCH.MAX]
        low_actor_entropy = torch.stack(low_actor_entropy_list, dim=0) #  [B * T * H, ARCH.MAX]

        # imag feat has both gradient for high- and low- level policies.
        imag_feat = self.get_feature_for_agent(
            sto_outputs[0], sto_outputs[1])

        indiv_latent_lists = self.get_indiv_features(
            sto_outputs[0], sto_outputs[1]
        )
        
        return imag_feat, preact_action, raw_action, indiv_latent_lists, \
            sub_policy_idx, sub_policy_ind_inpt, low_actor_entropy

    def get_initial_state(self) -> List[TensorType]:
        z_masks = self.mixture_module.z_mask_0.expand(1, ARCH.K, ARCH.Z_MASK_DIM) # 0
        z_comps = self.mixture_module.z_comp_0.expand(1, ARCH.K, ARCH.Z_COMP_DIM) # 1
       
        z_pres = torch.zeros(1, ARCH.MAX, ARCH.Z_PRES_DIM, device=z_masks.device) # 6
        z_depth = torch.zeros(1, ARCH.MAX, ARCH.Z_DEPTH_DIM, device=z_masks.device) # 7
        z_where = torch.zeros(1, ARCH.MAX, ARCH.Z_WHERE_DIM, device=z_masks.device) # 8
        z_what = torch.zeros(1, ARCH.MAX, ARCH.Z_WHAT_DIM, device=z_masks.device) # 9
        z_dyna = torch.zeros(1, ARCH.MAX, ARCH.Z_DYNA_DIM, device=z_masks.device) # 10
        ids = torch.zeros(1, ARCH.MAX, device=z_masks.device).long() # 13, ids_prop
        action = torch.zeros(1, ARCH.ACTION_DIM, device=z_masks.device)

        mix_states = self.mixture_module.get_init_recur_state()
        obj_states = self.obj_module.get_init_recur_state()

        post_states = [z_masks, z_comps]
        prior_states = []

        post_states.extend([t for t in mix_states['post']]) # 2, 3, 4, 5
        post_states.extend([ids, action]) # 13

        return post_states
