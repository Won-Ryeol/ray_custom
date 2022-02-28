import torch
from torch import nn
from .utils import spatial_transform
from .module import kl_divergence_bern_bern
from gatsbi_rl.gatsbi.arch import ARCH

def preprocess_attention_input(obs, z_size, z_goal_size, with_n_frames=None):
    n_objects = obs[:, :1]
    latent_obs = obs[:, :-(z_goal_size + 1)]
    g = obs[:, -z_goal_size:][:, None, :]

    if with_n_frames is not None:
        # obs layout:
        # 0 => n_total_objects
        # 1 : 1 + n_frames => n_objects_per_frame
        zs = latent_obs[:, 1 + with_n_frames:]

        # Add zero for z_depth and one-hot 1 for z_time_id
        g_add = torch.tensor([0, 1] + [0] * with_n_frames,
                             dtype=g.dtype, device=g.device)[None, None]
        g = torch.cat((g, g_add.expand(g.size(0), 1, -1)), dim=-1)
    else:
        # obs layout:
        # 0 => n_objects
        # 1:max_objects * z_size + 1 => objects
        # max_objects * z_size + 1 : max_objects * z_size + 2 => goal idx
        # max_objects * z_size + 2 : => goal
        zs = latent_obs[:, 1:]

        # Add zero for z_depth
        g = torch.cat((g, torch.zeros((g.size(0), 1, 1),
                                      dtype=g.dtype, device=g.device)), dim=-1)

    zs = rearrange(zs, 'b (objects z) -> b objects z', z=z_size)

    return zs, g, n_objects

class Attention(nn.Module):
    """
    Attention borrowed from https://github.com/martius-lab/SMORL.
    
    # TODO (chmin): figure out how attention works.

    #! initialization of attention.
    attention = Attention(embed_dim, z_goal_size, z_size,
                            **attention_kwargs)

    attention_key_query_size=8
    #? we have explicit positional parameters. Do we really need to
    #? embed 2-dim position into another dimension?

    def forward(self, obs,
                reparameterize=True,
                deterministic=False,
                return_log_prob=False):
        x, g, n_objects = preprocess_attention_input(obs,
                                                     self.z_size,
                                                     self.z_goal_size,
                                                     self.n_frames)
        goal_embedding = self.attention.embed(g)
        state_embedding = self.attention.embed(x)

        h = self.attention.forward(state_embedding, goal_embedding, n_objects)

    """

    def __init__(self, embed_dim, obj_pos_dim, agent_kypt_dim,
                 num_heads=3):
        super().__init__()

        self._embed_dim = max(num_heads, 1) * embed_dim

        self.embedding = nn.Linear(obj_pos_dim, self._embed_dim)
        torch.nn.init.zeros_(self.embedding.bias)

        attention_dim = self._embed_dim

        if self._embed_dim != attention_dim:
            self.query_proj = nn.Linear(self._embed_dim, attention_dim)
            torch.nn.init.zeros_(self.query_proj.bias)

        # torch's innate attention module.
        self.attention = nn.MultiheadAttention(attention_dim,
                                                num_heads,
                                                dropout=0.0,
                                                bias=True,
                                                add_bias_kv=False,
                                                add_zero_attn=False,
                                                kdim=self._embed_dim,
                                                vdim=self._embed_dim)
        nn.init.xavier_uniform_(self.attention.out_proj.weight)

        self.occ_attention = nn.MultiheadAttention(attention_dim,
                                                num_heads,
                                                dropout=0.0,
                                                bias=True,
                                                add_bias_kv=False,
                                                add_zero_attn=False,
                                                kdim=self._embed_dim,
                                                vdim=self._embed_dim)
        nn.init.xavier_uniform_(self.occ_attention.out_proj.weight)

        self.uncertain_pose_from_attention = nn.Sequential(
            nn.Linear(self._embed_dim + ARCH.Z_SHIFT_DIM + ARCH.ACTION_ENHANCE + 1 + 2, 128),
            nn.CELU(),
            nn.Linear(128, 3 * ARCH.Z_SHIFT_DIM)
            )

        self.z_occ_prior_feature_from_attention = nn.Sequential(
            nn.Linear(ARCH.ACTION_ENHANCE + self._embed_dim, 128),
            nn.CELU(),
            nn.Linear(128, ARCH.PROPOSAL_ENC_DIM),
            )

        # posterior and prior share these layers
        self.cat_occ_feature = nn.Sequential(
            nn.Linear(ARCH.PROPOSAL_ENC_DIM + ARCH.Z_SHIFT_DIM + 
                ARCH.Z_DEPTH_DIM, 64),
            nn.CELU(),
            nn.Linear(64, 64)
            )
        self.z_occ_from_feat = nn.Sequential(
            nn.CELU(),
            nn.Linear(64, 1)
        )


    @property
    def output_dim(self):
        dim = 0
        if self.attention is not None:
            dim += self.attention.embed_dim
        return dim

    @property
    def embed_dim(self):
        return self._embed_dim

    def embed(self, x):
        """
        In : x [B, N, 2]
        Out : embedding (batch_size, N, embed_dim)
        """
        return self.embedding(x)

    def forward(self, agent_kypt_embedding, obj_embedding, obj_where, agent_kypt_diff,
        z_occ, enhanced_act):
        """
        Input : state_embedding (batch_size, max_objects, embed_dim)
                g (batch_size, 1, embed_dim),
                n_objects (batch_size, 1)
        Output: value with shape (batch_size, embed_dim))
        """
        B, K, D = agent_kypt_embedding.size()
        N = obj_embedding.size(1)
        assert D == self._embed_dim
        assert obj_embedding.size(-1) == self._embed_dim

        # TODO (chmin): find out proper axes.
        agent_kypt_embedding = agent_kypt_embedding.transpose(1, 0) # [K, B, D]
        obj_embedding = obj_embedding.transpose(1, 0) # [N, B, D]

        if self.attention is not None:
            if self.attention.embed_dim != obj_embedding.size(-1):
                query = self.query_proj(obj_embedding)
            else:
                query = obj_embedding

            attn_output, _ = self.attention(query=query,
                                    key=agent_kypt_embedding,
                                    value=agent_kypt_embedding,
                                    key_padding_mask=None,
                                    need_weights=False) # out [N, B, D]
        # TODO (chmin): figure out the desired input for the attention.

        attn_output = attn_output.transpose(1, 0) # [B, N, D]
        enhanced_act = enhanced_act[:, None].repeat(1, N, 1)


        uncertain_out = self.uncertain_pose_from_attention(torch.cat([attn_output, 
        obj_where, enhanced_act, z_occ, agent_kypt_diff], dim=-1)) # [B, N, 3 * 2]
        return uncertain_out

    def occ_feature_post(self, obj_to_agent_depth, obj_to_agent_where, proposal_enc):
        """
        Input : 
                proposal_enc [B, N, D]
                obj_to_agent_depth [B, N, 1]
                obj_to_agent_where [B, N, 2]

        Output: value with shape (batch_size, embed_dim))
        """
        feat = torch.cat([proposal_enc, obj_to_agent_depth, 
            obj_to_agent_where], dim=-1) # [B, N, D + 3]

        occ_feat_cat = self.cat_occ_feature(feat) # [B, N, 1] 
        z_occ_prob = self.z_occ_from_feat(occ_feat_cat)
        return z_occ_prob, occ_feat_cat

    def occ_feature_prior(self, agent_kypt_embedding, obj_embedding, obj_to_agent_depth, 
        obj_to_agent_where, enhanced_act):
        """
        Input : state_embedding (batch_size, max_objects, embed_dim)
                g (batch_size, 1, embed_dim),
                n_objects (batch_size, 1)
        Output: value with shape (batch_size, embed_dim))
        """


        B, K, D = agent_kypt_embedding.size()
        N = obj_embedding.size(1)
        assert D == self._embed_dim
        assert obj_embedding.size(-1) == self._embed_dim

        # TODO (chmin): find out proper axes.
        agent_kypt_embedding = agent_kypt_embedding.transpose(1, 0) # [K, B, D]
        obj_embedding = obj_embedding.transpose(1, 0) # [N, B, D]

        if self.attention is not None:
            if self.attention.embed_dim != obj_embedding.size(-1):
                query = self.query_proj(obj_embedding)
            else:
                query = obj_embedding

            attn_output, _ = self.occ_attention(query=query,
                                    key=agent_kypt_embedding,
                                    value=agent_kypt_embedding,
                                    key_padding_mask=None,
                                    need_weights=False) # out [N, B, D]
        # TODO (chmin): figure out the desired input for the attention.

        attn_output = attn_output.transpose(1, 0) # [B, N, D]

        enhanced_act = enhanced_act[:, None].repeat(1, N, 1) # [B, N, A]

        z_occ_prior_feat = self.z_occ_prior_feature_from_attention(torch.cat([
            attn_output, enhanced_act], dim=-1)) # [B, N, 3 * 2]
        # order matters: depth -where
        z_occ_prior_feat = torch.cat([z_occ_prior_feat, 
            obj_to_agent_depth, obj_to_agent_where], dim=-1)

        occ_feat_cat = self.cat_occ_feature(z_occ_prior_feat) # [B, N, 1] 
        z_occ_prob = self.z_occ_from_feat(occ_feat_cat)
        return z_occ_prob

class AttentionMlp(nn.Module):
    def __init__(self, embed_dim, z_goal_size, z_size,
                 action_size, max_objects, hidden_sizes, output_size,
                 n_frames=None, attention_kwargs=None, **kwargs):
        super().__init__()
        self.z_goal_size = z_goal_size
        self.z_size = z_size
        self.n_frames = n_frames

        if attention_kwargs is None:
            attention_kwargs = {}

        self.attention = Attention(embed_dim,
                                   z_goal_size,
                                   z_size,
                                   **attention_kwargs)
        inp_dim = (self.attention.output_dim
                   + self.attention.embed_dim
                   + action_size)
        self.mlp = FlattenMlp(hidden_sizes, output_size, inp_dim, **kwargs)

    def forward(self, obs, actions):
        x, g, n_objects = preprocess_attention_input(obs,
                                                     self.z_size,
                                                     self.z_goal_size,
                                                     self.n_frames)
        goal_embedding = self.attention.embed(g)
        state_embedding = self.attention.embed(x)

        h = self.attention.forward(state_embedding, goal_embedding, n_objects)

        output = self.mlp(h, goal_embedding.squeeze(1), actions)

        return output