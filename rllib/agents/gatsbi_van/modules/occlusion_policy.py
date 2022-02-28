import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.nn import functional as F
from torch.distributions import RelaxedBernoulli
from torch.distributions.kl import kl_divergence
from gatsbi_rl.gatsbi.arch import ARCH
from gatsbi_rl.baselines.reward_utils import tensor_tolerance
from .utils import spatial_transform
from .module import kl_divergence_bern_bern
import numpy as np
from IQA_pytorch import SSIM
from torchvision.models import resnet18
from torchvision.models import inception_v3
import enum
from gatsbi_rl.gatsbi.uncertain_attention import Attention







class ImgEncoder(nn.Module):
    """
    Used in encoding for propagation. Input is image plus image - bg
    """
    
    def __init__(self):
        super(ImgEncoder, self).__init__()
        
        last_stride = ARCH.IMG_SIZE // (8 * ARCH.G)
        # last_stride = 1 if ARCH.G == 8 else 2


        self.last = nn.Conv2d(256, ARCH.IMG_ENC_DIM, 3, last_stride, 1) # halves dim
        # 3 channels
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) # reduce spatially 1/2
        
        resnet = resnet18(pretrained=True)
        self.reset18 = resnet18(pretrained=True)
        incept = inception_v3(pretrained=True)
        self.enc = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.layer1,
            resnet.layer2, # halves dim
            resnet.layer3, # halves dim
            resnet.layer4, # halves dim
            # self.last,
            nn.Flatten(),
            # nn.CELU(),
            nn.Linear(512, ARCH.IMG_ENC_DIM)
        )

        self.enc2 = nn.Sequential(
            incept.Conv2d_1a_3x3,
            incept.Conv2d_2a_3x3,
            incept.Conv2d_2b_3x3,
            incept.maxpool1,
            incept.Conv2d_3b_1x1,
            nn.Flatten(),
        )



    
    def forward(self, x):
        """
        Get image feature
        Args:
            x: (B, 3, H, W)
        Returns:
            enc: (B, 128, G, G)
        """
        B = x.size(0)
        x = x.reshape(-1, 3, 16, 16)
        x = self.enc(x) # [B*K*N, D]
        # self.reset18.conv1(x)[:, 0][:, None]
        # x = self.enc2(x)
        return x


class OcclusionPolicy(nn.Module):
    """ 
        It should be the submodule of Object module.

        Predict the z_where of occluded objects in a way that maximzes the following reward.

        Observation: local propagation glimpse o^g_t, glimpse of the fg from the 
            previous step fg^g_t and the action of the robot agent a_{t-1}.

        NOTE: this is not just like an normal agent.
        We just regulates the z_where by a new posterior inference, since we have the agent's action.

        What is posterior?
            Access to image observation.
            + agent action, and other RNN states
        What is prior?


        TODO (chmin): how to?

    """
    def __init__(self):
        nn.Module.__init__(self)
        
        self.delta_where_enc = nn.Sequential(
            nn.Linear(ARCH.Z_MASK_DIM + ARCH.ACTION_ENHANCE +
            ARCH.RNN_CTX_MASK_HIDDEN_DIM, ARCH.MASK_COND_HIDDEN_DIM),
            nn.CELU(),
            nn.Linear(ARCH.MASK_COND_HIDDEN_DIM, 2 * ARCH.Z_DEPTH_DIM)
        )

        # TODO (chmin): infer z_occ given rich information
        self.z_occ_post_enc = nn.Sequential(
            nn.Linear(ARCH.PROPOSAL_ENC_DIM + 2 * ARCH.Z_DEPTH_DIM +
                2 + ARCH.Z_MASK_DIM + ARCH.ACTION_ENHANCE, 128),
            nn.CELU(),
            nn.Linear(128, 64),
            nn.CELU(),
            nn.Linear(64, 1)
        )

        self.z_occ_prior_enc = nn.Sequential(
            nn.Linear(2 * ARCH.Z_DEPTH_DIM + ARCH.Z_MASK_DIM + 
                2 + ARCH.ACTION_ENHANCE, 64),
            nn.CELU(),
            nn.Linear(64, 1)
        )

        self.pred_under_occlu = nn.Sequential(
            nn.Linear(100, 100),
            nn.CELU(),
            nn.Linear(100, 100)
        )

        self.sample_attention = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7,
                      stride=2, padding=3),
            nn.CELU(),
            nn.GroupNorm(num_groups=4, num_channels=32),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(32, 256),

            nn.Flatten(),
            nn.Linear(16 * 256, ARCH.IMG_ENC_DIM)
        )

        self.uncertain_pose = nn.Sequential(
            nn.Linear(ARCH.IMG_ENC_DIM + ARCH.ACTION_ENHANCE, 128),
            nn.CELU(),
            nn.Linear(128, 3 * ARCH.Z_SHIFT_DIM)
        )

        self.occlusion_diff_encoder = ProposalEncoder()

        # TODO (chmin): initialize with proper arguments
        # TODO (chmin): leave it out when loading the models.
        self.curl = CURL()
        self.freeze_teacher = False

        self.uncertain_attention = Attention(embed_dim=ARCH.ATTENTION_DIM, 
            obj_pos_dim=2, agent_kypt_dim=2, num_heads=3)


    def z_occ_post(self, proposal_enc, proposal, agent_kypt, z_depth, 
            z_agent_depth, z_agent, enhanced_act):
        """
            Input:
                proposal_enc: [B, N, D]

            obs_diff[:, 3:]
            It observes the agent_mask, obj_mask, z_pres, z_where, z_depth, and z_agent_depth
            It is modeled as an RelaxedBernoulli distribution.
        """ 

        B, N, _ = z_depth.size()

        kypt_pos = agent_kypt[..., :2] # [B, K, 2]
        kypt_weight = agent_kypt[...,-1][..., None] # [B, K, 1]
        # kypt_weight_sum = 
        # agent_kypt_mean = torch.mean(kypt_pos, dim=1, keepdim=True) # [B, 1, 2] mean position of agent keypoints.
        # agent_kypt_mean = torch.sum(kypt_pos * kypt_weight, dim=1, keepdim=True)  # [B, 1, 2] mean position of agent keypoints.
        agent_kypt_mean = (kypt_pos * kypt_weight).sum(1, keepdim=True) / kypt_weight.sum(1, keepdim=True)
        obj_pos = proposal[..., 2:] # [B, N, 2] 
        # obj_pos = 2. * obj_pos - 1. # [B, N, 2] scale and shift to [-1, 1]

        z_agent_depth = z_agent_depth[:, None] # [B, 1, 1]
        obj_to_agent_depth = z_agent_depth - z_depth  # [B, N, 1]
        obj_to_agent_where = agent_kypt_mean - obj_pos # [B, N, 2] positional vector from agent to each object.

        # TODO (chmin): embed the agent keypoint and the query object positions.
        kypt_embeds = self.uncertain_attention.embed(kypt_pos)
        obj_embeds = self.uncertain_attention.embed(obj_pos)

        z_occ_post_prob, occ_feat_cat = self.uncertain_attention.occ_feature_post(
                obj_to_agent_depth=obj_to_agent_depth,
                obj_to_agent_where=obj_to_agent_where,
                proposal_enc=proposal_enc
            )

        z_occ_post_prob = torch.sigmoid(z_occ_post_prob) # [B, N, 1]
        z_occ_post = RelaxedBernoulli(temperature=torch.tensor(ARCH.OCC_BERN_TEMP, device=proposal_enc.device), probs=z_occ_post_prob)

        z_occ = z_occ_post.rsample() # [B, N, 1]
        # minimize log_prob <> maximize ent
        z_occ_ent = - z_occ_post.log_prob(z_occ_post.probs)

        # prior distribution
        _, z_occ_prior_prob = self.z_occ_prior(
                agent_kypt_embedding=kypt_embeds,
                obj_embedding=obj_embeds,
                obj_to_agent_depth=obj_to_agent_depth,
                obj_to_agent_where=obj_to_agent_where,
                enhanced_act=enhanced_act
            )

        # compute kl divergence between post and prior
        kl_occ = kl_divergence_bern_bern(z_occ_post_prob, z_occ_prior_prob) # [B, N, 1]

        return z_occ, kl_occ, z_occ_post, z_occ_ent, z_occ_post_prob, occ_feat_cat

    def z_occ_prior(self, agent_kypt_embedding, obj_embedding, 
            obj_to_agent_depth, obj_to_agent_where, enhanced_act):
        """
            z_occ inference during prior propagate generation.

            We don't yield fg images during generation. Thus,
            z_occ_prior can only be inferred from z_pres, z_depth, and z_agent_depth.

            It observes the agent_mask, obj_mask, z_pres, z_where, z_depth, and z_agent_depth
            It is modeled as an RelaxedBernoulli distribution.
        """
        B, N, *_  = obj_to_agent_depth.size()

        z_occ_prior_prob = self.uncertain_attention.occ_feature_prior(
                agent_kypt_embedding=agent_kypt_embedding,
                obj_embedding=obj_embedding,
                obj_to_agent_depth=obj_to_agent_depth,
                obj_to_agent_where=obj_to_agent_where,
                enhanced_act=enhanced_act
            )
        z_occ_prior_prob = torch.sigmoid(z_occ_prior_prob) # [B, N, 1]

        z_occ_prior = RelaxedBernoulli(temperature=torch.tensor(ARCH.OCC_BERN_TEMP, 
            device=obj_embedding.device), 
            probs=z_occ_prior_prob)
        z_occ = z_occ_prior.rsample()
        return z_occ, z_occ_prior_prob

    def sample_object(self, agent_kypt, proposal, enhanced_act, z_occ,
        z_agent, h_agent, agent_kypt_prev):
        """
            Forward inference of occlusion policy.

            Input: 
                agent_kypt: [B, K, 3], K is the number of kypts. In range [-1, 1]
                proposal: [B, N, 4], where the last two is object position. In range [-1, 1]

            Input: agent keypoints (not in Gaussian) and z_where
            Given z_where, create proposal glimpse and acquire keypoints allocated to the glimpse.
            Generate a Gaussian distribution 


        """
        B, N, *_ = proposal.size()

        # TODO (chmin): debug the self-attention here.

        kypt_pos = agent_kypt[..., :2] # [B, K, 2]
        kypt_weight = agent_kypt[...,-1][..., None] # [B, K, 1]
        agent_kypt_mean = (kypt_pos * kypt_weight).sum(1, keepdim=True) / kypt_weight.sum(1, keepdim=True)

        kypt_prev_pos = agent_kypt_prev[..., :2]
        kypt_prev_weight = agent_kypt_prev[..., -1][..., None]
        agent_kypt_prev_mean = (kypt_prev_pos * kypt_prev_weight).sum(1, keepdim=True) / kypt_prev_weight.sum(1, keepdim=True)

        obj_pos = proposal[..., 2:] # [B, N, 2] 
        # obj_pos = 2. * obj_pos - 1. # [B, N, 2] scale and shift to [-1, 1]

        with torch.no_grad():
            obj_heatmap_input = torch.cat([obj_pos, torch.ones(B, N, 1,  device=z_occ.device)], dim=-1)
            agent_heat_map = generate_heatmaps(keypoints=agent_kypt)
            flipped_agent = agent_heat_map.sum(1, keepdim=True).flip(dims=[-2])
            obj_heat_map = generate_heatmaps(keypoints=obj_heatmap_input)
            flipped_obj =  obj_heat_map.sum(1, keepdim=True).flip(dims=[-2])


        # TODO (chmin): embed the agent keypoint and the query object positions.
        kypt_embeds = self.uncertain_attention.embed(kypt_pos)
        obj_embeds = self.uncertain_attention.embed(obj_pos)

        # TODO (chmin): do positional encoding based on the decaying w.r.t. centroid of agent kypt.

        # agent_kypt_diff = kypt_pos.mean(1, keepdim=True) - agent_kypt_prev[..., :2].mean(1, keepdim=True)
        agent_kypt_diff = agent_kypt_mean - agent_kypt_prev_mean
        agent_kypt_diff = agent_kypt_diff.repeat(1, N, 1) # [B, N, 2]

        if len(h_agent.size()) == 1:
            h_agent = h_agent[None]
        agent_feature = torch.cat([z_agent, h_agent], dim=-1)
        agent_feature = agent_feature[:, None].repeat(1, N, 1) # [B, N , D]

        uncertain_out = self.uncertain_attention(
                agent_kypt_embedding=kypt_embeds,
                obj_embedding=obj_embeds,
                obj_where=obj_pos,
                z_occ=z_occ,
                enhanced_act=enhanced_act,
                agent_kypt_diff=agent_kypt_diff
            )

        kypt_pos = kypt_pos[:, None]   # [B, 1, K, 2]
        obj_pos = obj_pos[:, :, None]  # [B, N, 1, 2]

        # this should be keypoint position relative to each object's position
        pos_diff = kypt_pos - obj_pos # [B, N, K, 2]

        # make it as distance 
        pos_diff_norm = torch.norm(pos_diff, dim=-1) # [B, N, K]

        # assert pos_diff_norm.max() < 2. * np.sqrt(2) 
        # construct a euclidian distance matrix

        # weight the sampling prob by pos_diff_norm
        max_dists = torch.max(pos_diff_norm, dim=-1)[0].detach() # [B, N]

        heatmap_decay = tensor_tolerance( # [B, N, K]
            x=pos_diff_norm, 
            bounds=(0.0, 0.0), 
            margin=max_dists, 
            sigmoid='long_tail')

        heatmap_decay = heatmap_decay

        uncertain_loc, uncertain_scale, uncertain_gate = torch.split(uncertain_out, 
            3 * [ARCH.Z_SHIFT_DIM], dim=-1) # 2 * [B, N, 2]
        uncertain_scale = F.softplus(uncertain_scale) + 1e-4 # [B, N, 2]

        uncertain_dist = Normal(uncertain_loc, uncertain_scale) # [B, N, 2]
        uncertain_pos = uncertain_dist.rsample()
        # fit the loc to the mean of decayed keypoints

        normalized_decay = heatmap_decay / (torch.sum(heatmap_decay, dim=2, keepdim=True)+ 1e-5)
        decayed_diffs = pos_diff * normalized_decay[..., None] # [B, N, K, 2] * [B, N, K, 1] -> [B, N, K, 2]

        uncertain_prior = Normal(decayed_diffs.detach().mean(2), 0.1)
        kl_uncertain = kl_divergence(uncertain_dist, uncertain_prior)
        # this should be optimized under occlusion scenarios
        return uncertain_pos, uncertain_gate, kl_uncertain

    def train_teacher_student(self, diff_enc, obs, obs_diff, agent_kypt, 
        teacher_dist, z_occ_post, obj_diffs=None, proposal_enc=None, prop_enc=None, proposals=None, z_where=None):
        """
            Train teacher (z_occ classifier) based on the student's performance (recon loss?)/
            Basic concept is similar to "Meta Pseudo Labels": https://arxiv.org/abs/2003.10580
            Code reference (official): https://github.com/google-research/google-research/blob/master/meta_pseudo_labels/training_utils.py

            This method aims a contrastive learning for training z_occ.
            We implement InfoNCE style contrastive learning.
                z_occ_post: [B, N, 1]
            
            What dot_product implies:
                https://github.com/kekmodel/MPL-pytorch/issues/6

            TODO (chmin): please read through this thread.
            https://github.com/google-research/google-research/issues/534
            # bout dot_product:
            Yes. dot_product is intended to be cross_entropy['s_on_l_new'] -
            cross_entropy['s_on_l_old']

            NOTE: we can just use lob_prob from RelaxedBernoulli that inherits 
                RelaxedBernoulliLogit.

            Anchor: Samples augmented around the agent keypoints.
            Positive: Samples that the teacher classified as z_occ ~1.
            Negatives: Samples that the teacher classified as z_occ ~0. 

        """
        # student should perform well on contrastive loss from the teacher anchors.reshape(-1, 3, 16, 16)

        anchors = self.curl.generate_anchors(agent_kypt=agent_kypt, obs=obs, z_where=z_where) # [B, K, 3, 16, 16]
        # Compute cpc loss between anchor, positives and negatives.
        # TODO (chmin): this should be deprecated.
        # student_loss_opt = self.curl.compute_cpc(proposal_enc=proposal_enc, obj_diffs.reshape(-1, 3, 16, 16)
        #     prop_enc=prop_enc, anchors=anchors, z_occ=z_occ_post) # [B * N, ] positives.reshape(-1, 3, 16, 16)

        student_loss_opt = self.curl.contrastive_loss(proposal_enc=proposal_enc,
            prop_enc=prop_enc, anchors=anchors, z_occ=z_occ_post, proposals=proposals, z_where=z_where)

        # _ = self.curl.contrastive_diff_loss(
        #     diff_encoder=self.occlusion_diff_encoder,
        #     diff_enc=diff_enc, anchors=anchors_diff, z_occ=z_occ_post, obj_diffs=obj_diffs)
        # with torch.no_grad():
        _ = self.curl.contrastive_loss(
            diff_encoder=prop_enc,
            diff_enc=proposal_enc, anchors=anchors, z_occ=z_occ_post, obj_diffs=proposals)

        teacher_likelihood = teacher_dist.log_prob(z_occ_post).sum(-1).reshape(student_loss_opt.size(0)) # [B ,]
        
        # TODO (chmin): compute the actual teacher gradient in the main training loop.
        return student_loss_opt, teacher_likelihood

class CURL(nn.Module):
    """
    CURL adopted from https://github.com/MishaLaskin/curl/blob/master/curl_sac.py
    """

    def __init__(self):
        super(CURL, self).__init__()

        # TODO (chmin): create target encoder for ema?

        self.W = nn.Parameter(torch.rand(ARCH.PROPOSAL_ENC_DIM, ARCH.PROPOSAL_ENC_DIM))
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none') 
        # self.img_encoder = ImgEncoder()

    def compute_logits(self, z_a, z_pos):
        """
            z_a: [B, N, D]
            z_pos: [B, N, D]
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        z_a = z_a.reshape(-1, ARCH.IMG_ENC_DIM)
        z_pos = z_pos.reshape(-1, ARCH.IMG_ENC_DIM)

        Wz = torch.matmul(self.W, z_pos.T)  # [D, B*N]
        logits = torch.matmul(z_a, Wz)  # [B*N, B*N]
        # find maximum logits along the positive axis.
        # subtract max from logits for stability
        # https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning
        logits = logits - torch.max(logits, dim=1)[0][:, None] # [B * N, B * N] - [B * N, 1]
        return logits

    def generate_anchors(self, agent_kypt, obs, z_where=None):
        """
            Generate positive samples using spatial transformers.

            Input for spatial_transform should have the following shape.
                agent_kypt: [B, T, K, 3] 
                obs: [B, T, 3, 64, 64] 
                z_where: [B, T, N, 4]

                the first two elements are the diagonal term for SO(2)
                the las two terms are the translation.
        """
        B, T, K, _ = agent_kypt.size()

        obs = obs.reshape(-1, 3, 64, 64) # [B*T, 3, 64, 64]

        obs_repeat = torch.repeat_interleave(obs, K, dim=0) # [B*T*K, 3, 64, 64]
        # uniformly sample the rotation params for the scaling
        max_rot1 = torch.minimum(z_where[..., 0].max(), torch.tensor(0.3, device=obs.device))
        min_rot1 = torch.maximum(z_where[..., 0].min(), torch.tensor(0.1, device=obs.device))
        max_rot2 = torch.minimum(z_where[..., 1].max(), torch.tensor(0.3, device=obs.device))
        min_rot2 = torch.maximum(z_where[..., 1].min(), torch.tensor(0.1, device=obs.device))
        rot1 = ((min_rot1 - max_rot1) * torch.rand(B, T, K, 1).to(obs.device) + \
            max_rot1) * torch.ones((B, T, K, 1), device=obs.device)
        rot2 = ((min_rot2 - max_rot2) * torch.rand(B, T, K, 1).to(obs.device) + \
            max_rot2) * torch.ones((B, T, K, 1), device=obs.device)
        
        agent_kypt = agent_kypt[..., :2] # [B, T, K, 2]
        
        agent_kypt = torch.cat([agent_kypt[..., :1], - agent_kypt[..., -1:]], dim=-1)
        where = torch.cat([rot1, rot2, agent_kypt], dim=-1) # [B, T, K, 4]
        where = where.reshape(-1, where.size(-1)) # [B*T*K, 4]
        # TODo (chmin): generate kypt_where
        anchors = spatial_transform(image=obs_repeat, z_where=where, 
            out_dims=(B * T * K, 3, *ARCH.GLIMPSE_SHAPE)) # [B*T*K, 3, 16, 16]

        return anchors.reshape(B, T, K, 3, *ARCH.GLIMPSE_SHAPE)


    def compute_cpc(self, proposal_enc, prop_enc, anchors, z_occ):
        """
            anchor: [B, K, 3, 64, 64]
            proposal_enc: [B, N, D]
            z_occ_post: [B, N, 1]

            if step % self.cpc_update_freq == 0 and self.encoder_type == 'pixel':
                obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
                self.update_cpc(obs_anchor, obs_pos,cpc_kwargs, L, step)

        """
        B, N, _ = proposal_enc.size()
        z_a = proposal_enc # encoding of anchor
        # TODO (chmin): do something with z_occ...
        # truncate anchors to create squared matrix

        anchors = anchors[:, :N]
        with torch.no_grad():
            z_pos = prop_enc(anchors) # [B, K, D]

        # TODO (chmin): check the shape of in&outs below.
        logits = self.compute_logits(z_a, z_pos) # [B * N, B * N]
        # TODO (chmin): compute cossim or something...

        labels = torch.arange(logits.shape[0]).long().to(proposal_enc.device)
        curl_loss = self.cross_entropy_loss(logits, labels) # [B * N, ]
        # Input: logits [B, C=number of classes] logits[None]
        # Input: lables [0, 1, 2, ..., B-1] class indices   -> diagonal terms
        # please refer this https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

        return curl_loss
    # def contrastive_loss(self, proposal_enc, prop_enc, anchors, z_occ, proposals, z_where):
    #     """
    #         anchor: [B, K, N, 3, 64, 64] anchors.reshape(-1, 3, 16, 16)
    #         proposal_enc: [B, N, D]
    #         z_occ_post: [B, N, 1]

    #         if step % self.cpc_update_freq == 0 and self.encoder_type == 'pixel':
    #             obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
    #             self.update_cpc(obs_anchor, obs_pos,cpc_kwargs, L, step)
    #     """
    #     B, N, _ = proposal_enc.size()
    #     K = anchors.size(1)
    #     # TODO (chmin): do something with z_occ...
    #     # truncate anchors to create squared matrix

    #     proposals = proposals[:, None].repeat(1, K, 1, 1, 1, 1) # [B, K, N, 3, 64, 64]
    #     # anchors = anchors[:, :N]
    #     with torch.no_grad():
    #         anchor_enc = prop_enc(anchors) # [B, N, D]
    #     # proposals.reshape(-1, 3, 16, 16) anchors.reshape(-1, 3, 16, 16)
    #     # construct our own contrastive? loss anchors[0]
    #     # scale z_occ to fit into tanh.
    #     z_occ = 8. * z_occ - 4.
    #     z_occ = torch.tanh(z_occ) # check if outputs mostly lie near -1 or 1.
        
    #     # we aim to map samples (pseudo) inferr ed as occlusion (z_occ ~ 1) to 1. and
    #     # non-occlusion (z_occ ~ 0) as -1.
    #     cossim = nn.CosineSimilarity(dim=-1, eps=1e-6)
    #     # TODO (chmin): tentative measurment
    #     mean_anchor_enc = anchor_enc.mean([0, 1, 2])[None, None]
         
    #     sim = cossim(mean_anchor_enc, proposal_enc.detach())
    #     loss = torch.nn.MSELoss(reduction='none')
    #     cont_loss = loss((z_occ * sim[..., None]).sum(-1).reshape(-1), 
    #         torch.ones_like((z_occ * sim[..., None]).sum(-1).reshape(-1)).to(z_occ.device))

    #     # ssim = SSIM()
    #     # ssim(X=anchors.reshape(-1, 3, 16, 16), Y=proposals.reshape(-1, 3, 16, 16), as_loss=False)
    #     # where_sim = torch.where(ssim(X=anchors.reshape(-1, 3, 16, 16), Y=proposals.reshape(-1, 3, 16, 16), as_loss=False) > 0.3)
    #     # where_sim_ex = torch.where(ssim(X=anchors.reshape(-1, 3, 16, 16), Y=proposals.reshape(-1, 3, 16, 16), as_loss=False) <= 0.3)
        
    #     # anchors.reshape(-1, 3, 16, 16)[where_sim]
    #     # proposals.reshape(-1, 3, 16, 16)[where_sim]
    #     # proposals.reshape(-1, 3, 16, 16)[where_sim_ex]
 
    #     return cont_loss


    def contrastive_loss(self, diff_encoder, diff_enc, anchors, z_occ, obj_diffs=None):
        """
            anchor: [B, K, 3, 64, 64]
            diff_enc: [B, N, D]
            z_occ_post: [B, N, 1]
            anchors[0]
            if step % self.cpc_update_freq == 0 and self.encoder_type == 'pixel':
                obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
                self.update_cpc(obs_anchor, obs_pos,cpc_kwargs, L, step)

        """


        B, N, _ = diff_enc.size()
        K = anchors.size(1)
        # TODO (chmin): do something with z_occ...
        # truncate anchors to create squared matrix
        # anchors = anchors[:, :N] anchors.reshape(-1, 3, 16, 16)[27]
        with torch.no_grad():
            anchor_enc = diff_encoder(anchors) # [B, N, D]
            anchor_resnet = self.img_encoder(anchors)
        obj_diffs = obj_diffs[:, None].repeat(1, K, 1, 1, 1, 1) # [B, K, N, 3, 64, 64]
        obs_resnet = self.img_encoder(obj_diffs)
        diff_enc = diff_enc[:, None].repeat(1, K, 1, 1)
        
        # construct our own contrastive? loss anchors[0]
        # scale z_occ to fit into tanh. obj_diffs.reshape(-1, 3, 16, 16)[torch.where(cossim(anchor_resnet.mean(0)[None], obs_resnet) > 0.7)]
        # z_occ = 8. * z_occ - 4.
        # z_occ = torch.tanh(z_occ) # check if outputs mostly lie near -1 or 1.
        
        z_occ = z_occ[:, None].repeat(1, K, 1, 1).reshape(B, K*N, 1) # [B, K, N, 1]
        # we aim to map samples (pseudo) inferr ed as occlusion (z_occ ~ 1) to 1. and
        # non-occlusion (z_occ ~ 0) as -1. torch.matmul(ancs, diffs.T)[None]
        cossim = nn.CosineSimilarity(dim=-1, eps=1e-6)
        mean_anchor_enc = anchor_enc.mean([0, 1])[None, None]

        sim = cossim(mean_anchor_enc, diff_enc.detach())

        # TODO (chmin): experimental do cpc here.
        # ancs = anchor_enc.reshape(-1, ARCH.IMG_ENC_DIM)
        # diffs = diff_enc.reshape(-1, ARCH.IMG_ENC_DIM)

        # Wz = torch.matmul(self.W, diffs.T)  # [D, B*N]
        # logits = torch.matmul(ancs, Wz)  # [B*N, B*N] logits[None]
        # # TODO (chmin): experimental
        # # these are normalized version bilinear_logits[None]
        # # torch.matmul(F.normalize(ancs, dim=-1), F.normalize(diffs, dim=-1).T)
        # bilinear_logits = torch.matmul(F.normalize(ancs, dim=-1), F.normalize(Wz, dim=0))

        # ssim = SSIM()
        # ssim(X=anchors.reshape(-1, 3, 16, 16), Y=obj_diffs.reshape(-1, 3, 16, 16), as_loss=False)
        # ssim(X=anchors.reshape(-1, 3, 16, 16), Y=obj_diffs.reshape(-1, 3, 16, 16), as_loss=False)[None, None]
        loss = torch.nn.MSELoss(reduction='none')
        # cont_loss = loss((z_occ * sim[..., None]).sum(-1).reshape(-1), 
        #     torch.ones_like((z_occ * sim[..., None]).sum(-1).reshape(-1)).to(z_occ.device))
        # TODO (Chmin): we compare ssim for every combination between agent keypoints (K) and objects (N). 
        # so, there should be maximum of [B, N*K] indices.

        guide_sim = SSIM()
        # ssim(X=anchors.reshape(-1, 3, 16, 16), Y=proposals.reshape(-1, 3, 16, 16), as_loss=False)
        # where_sim = torch.where(ssim(X=anchors.reshape(-1, 3, 16, 16), Y=obj_diffs.reshape(-1, 3, 16, 16), as_loss=False) > 0.3)[0]
        # obj_diffs.reshape(-1, 3, 16, 16)
        # TODO (chmin) compute total (N*K) * (N*K) ssim values 
        #  iterate over N*K times
        max_inds = []
        for anc in anchors.reshape(B, K * N, 3, 16, 16).permute(1, 0, 2, 3, 4): # iterate over N*K
            ssim = guide_sim(X=anc.repeat(K * N, 1, 1, 1), Y=obj_diffs.reshape(-1, 3, 16, 16), as_loss=False) # [B*N*K]
            ssim = ssim.reshape(B, K * N)
            # find maximum index obj_diffs.reshape(-1, K * N, 3, 16, 16)[max_ind[0].long(), max_ind[1].long()]
            max_ind = torch.argmax(ssim, dim=-1) # [B,] anc.repeat(K * N, 1, 1, 1).reshape(-1, K * N, 3, 16, 16)[max_ind[0].long(), max_ind[1].long()]
            # occl_ind = torch.where(ssim > 0.3) # tuple of tensors. each tuple dim indicates Batch and K*N axis
            max_inds.append(max_ind)
        max_inds = torch.stack(max_inds, dim=1).long() # [B, K*N,]
        # max_inds = max_inds.reshape(B, K, N).long() # [B, K, N]
        obj_ind = max_inds % N # [B, K*N,]
        # ssim = guide_sim(X=anchors.reshape(-1, 3, 16, 16), Y=obj_diffs.reshape(-1, 3, 16, 16), as_loss=False)
        # # ssim = ssim.reshape(B, K, N, 1)

        pseudo_z_occ = []
        batch_idx = torch.arange(B, device=z_occ.device).long()
        for ind in obj_ind.permute(1, 0): # iterate over index dim {0, 1, ... ,N-1}, total K*N times
            # inds has Batch shape
            pseudo_z_occ.append(z_occ[batch_idx, ind]) # append of shape [B, 1] 

        # max_b, max_k, max_n = torch.argmax(ssim, dim=-1)[1].long() # these are z_occ
        # anchors[max_b, max_k, max_n].reshape(-1, 3, 16, 16)
        # z_occ = z_occ[:, None].repeat(1, K, 1, 1)[max_b, max_k, max_n] anchors.reshape(-1, 3, 16, 16).mean(0)
        # where_sim = torch.where(ssim > 0.3)
        # where_sim_ex = torch.where(guide_sim(X=anchors.reshape(-1, 3, 16, 16), Y=obj_diffs.reshape(-1, 3, 16, 16), as_loss=False) <= 0.3)
        
        # anchors.reshape(-1, 3, 16, 16)[where_sim]
        # obj_diffs.reshape(-1, 3, 16, 16)[where_sim]
        # obj_diffs.reshape(-1, 3, 16, 16)[where_sim_ex]

        return None



class ProposalEncoder(nn.Module):
    """Same as glimpse encoder, but for encoding the proposal area"""

    def __init__(self):
        nn.Module.__init__(self)
        embed_size = ARCH.GLIMPSE_SIZE // 16
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(1, 16),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(2, 32),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(4, 64),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 128),
        )
        # TODO (cheolhui): robot action should be conditioned independently
        self.enc_what = nn.Linear(128 * embed_size ** 2, ARCH.PROPOSAL_ENC_DIM)

    def forward(self, x):
        """
        Args:
            x: (B, N, 3, H, W)

        Returns:
            enc: (B, N, D)
        """
        x_size = x.size()
        H, W = x_size[-2], x_size[-1]
        x = x.reshape(-1, 3, H, W)
        x = self.enc(x) # [B*N, D, 1, 1]
        x = x.flatten(start_dim=1) # [B*N, D]

        new_size = x_size[:-3] + (ARCH.PROPOSAL_ENC_DIM, )
        return self.enc_what(x).view(*new_size)


# for debugging purpose
def generate_heatmaps(keypoints, sigma=2.0, heatmap_width=64):
    """Turns (x, y, scale)-tuples into pixel maps with a Gaussian blob at (x, y).
    Args:
    keypoints: [batch_size, K=num_keypoints, 3] tensor of keypoints where the 1st
        dimension contains (x, y, scale) triplets.
    sigma: Std. dev. of the Gaussian blob, in units of heatmap pixels.
    heatmap_width: Width of output heatmaps in pixels.
    Returns:
    A [batch_size, num_keypoints, heatmap_width, heatmap_width] tensor.
    """

    coordinates, map_scales = torch.split(keypoints, [2, 1], dim=-1) # [B, K, 2], [B, K, 1]
    # split into two dim and one dim
    def get_grid(axis):
        grid = _get_pixel_grid(axis, heatmap_width)
        shape = [1, 1, 1, 1]
        shape[axis.value] = -1
        return torch.reshape(grid, shape)

    # Expand to [batch_size, num_keypoints, 1, 1] for broadcasting later:
    # TODO(cheolhui): determine the axes for # keypoints and the features (x,y,scale) 
    x_coordinates = coordinates[:, :, None, None, 0] # [B, K, 1, 1]
    y_coordinates = coordinates[:, :, None, None, 1] # [B, K, 1, 1]

    # Create two 1-D Gaussian vectors (marginals) and multiply to get a 2-d map:
    sigma = torch.tensor(sigma).float()
    keypoint_width = 2.0 * (sigma / heatmap_width) ** 2.0
    x_vec =  (- (get_grid(Axis.x).cuda(device=keypoints.device) - x_coordinates).pow(2) / keypoint_width).exp() # [B, K, 1, W]
    y_vec =  (- (get_grid(Axis.y).cuda(device=keypoints.device) - y_coordinates).pow(2) / keypoint_width).exp() # [B, K, H, 1]
    maps =  x_vec * y_vec # [B, K, H, W]
    return maps * map_scales[:, :, 0, None, None] # [B, K, H, W]

def _get_pixel_grid(axis, width):
    """Returns an array of length `width` containing pixel coordinates."""
    if axis == Axis.x: # pixel width
        return torch.linspace(-1.0, 1.0, width)  # Left is negative, right is positive.
    elif axis == Axis.y: # pixel height
        return torch.linspace(1.0, -1.0, width)  # Top is positive, bottom is negative.

class Axis(enum.Enum):
    """Maps axes to image indices, assuming that 0th dimension is the batch,
      and the 1st dimension is the channel."""
    y = 2
    x = 3
