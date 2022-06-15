from yacs.config import CfgNode
import os
ARCH = CfgNode()

# scalor related parameters
ARCH.PHASE_NLL = False
ARCH.NO_DISCOVERY = False
ARCH.NUM_CELL_H = 4
ARCH.NUM_CELL_W = 4
ARCH.N_ITR = 6001
ARCH.BATCH_SIZE = 8
ARCH.LR = 1e-4
ARCH.MAX_NUM_OBJ = 5
ARCH.EXPLAINED_RATIO_THRESHOLD = 0.1
ARCH.SIGMA = 0.1
ARCH.RATIO_ANC = 1.0
ARCH.VAR_ANC = 0.3
ARCH.SIZE_ANC = 0.22
ARCH.VAR_S = 0.12
ARCH.Z_PRES_ANNEAL_END_VALUE = 1e-4
ARCH.Z_WHAT_DIM = 8
ARCH.Z_WHERE_DIM = 4
ARCH.Z_WHERE_SCALE_DIM = 2
ARCH.Z_WHERE_SHIFT_DIM = 2
ARCH.Z_DEPTH_DIM = 1
ARCH.IMG_H = 64
ARCH.IMG_W = 64
ARCH.COLOR_NUM = 500
ARCH.CP = 1.0
ARCH.EPOCHS = 4000
ARCH.GEN_DISC_PRES_PROBS = 0.1
ARCH.GENERATE_FREQ = 5000
ARCH.OBSERVE_FRAMES = 5
ARCH.PHASE_CONV_LSTM = True
ARCH.PHASE_DO_REMOVE_DETACH = True
ARCH.PHASE_EVAL = True
ARCH.PHASE_GENERATE = False
ARCH.PHASE_NO_BACKGROUND = False
ARCH.PHASE_PARALLEL = False

ARCH.GLIMPSE_SIZE = 32
ARCH.IMG_ENCODE_DIM = 64
ARCH.BG_WHAT_DIM = 1
ARCH.TEMPORAL_RNN_HID_DIM = 128
ARCH.TEMPORAL_RNN_OUT_DIM = 128
ARCH.PROPAGATE_ENCODE_DIM = 32

ARCH.Z_WHERE_TRANSIT_BIAS_NET_HID_DIM = 128
ARCH.Z_DEPTH_TRANSIT_NET_HID_DIM = 128
ARCH.Z_PRES_HID_DIM = 64

ARCH.Z_WHAT_FROM_TEMPORAL_HID_DIM = 64
ARCH.Z_WHAT_ENC_DIM = 128

ARCH.Z_WHAT_ENC_DIM = 128
ARCH.TAU = 1.0

ARCH.REOVE_DETACH_STEP = 30000

ARCH.PHASE_SIMPLIFY_SUMMARY = True
ARCH.PRING_FREQ = 100
ARCH.REMOVE_DETACH_STEP = 30000
ARCH.SAVE_EPOCH_FREQ = 1000
ARCH.SEED = 777
ARCH.GLOBAL_STATE = 0
ARCH.TAU_END = 0.3
ARCH.TAU_EP = 20000.0
ARCH.TAU_IMP = 0.25























prior_rnn_hid_dim = 64
prior_rnn_out_dim = prior_rnn_hid_dim

seq_len = 10
phase_obj_num_contrain = True
phase_rejection = True

temporal_img_enc_hid_dim = 64
temporal_img_enc_dim = 128
z_where_bias_dim = 4
temporal_rnn_inp_dim = 128
prior_rnn_inp_dim = 128
bg_prior_rnn_hid_dim = 32
where_update_scale = .2

pres_logit_factor = 8.8

conv_lstm_hid_dim = 64


# TODO (chmin): wrap it up.
                cp=1.0, 
                epochs=4000, 
                gen_disc_pres_probs=0.1, 
                generate_freq=5000, 
                last_ckpt='', 
                nocuda=False, 
                num_img_summary=3, 
                observe_frames=5, 
                phase_conv_lstm=True, 
                phase_do_remove_detach=True, 
                phase_eval=True,  
                phase_generate=False, 
                phase_nll=False, 
                phase_no_background=False, 
                phase_parallel=False, 
                phase_simplify_summary=True, 
                print_freq=100, 
                remove_detach_step=30000, 
                save_epoch_freq=1000, 
                seed=666, 
                global_state=0, 
                summary_dir='./summary', 
                tau_end=0.3, 
                tau_ep=20000.0, 
                tau_imp=0.25


# Whether to use deterministic version
ARCH.DETER = False
ARCH.ORIGINAL_GATSBI = False
ARCH.G = 8
ARCH.GENERATION_SEQ_LEN = 70
# T: sequence length

# Maximum number of objects
ARCH.MAX = 5
# Enable background modeling
ARCH.BG_ON = False
# This will be useful only when BG_ON is True. Before this step, we learn background only.
ARCH.BG_ONLY_STEP = 1000
# For gaussian
# If we use gaussian
ARCH.SIGMA = 0.2
ARCH.SIGMA_ANNEAL = False
ARCH.SIGMA_START_VALUE = 0.2
ARCH.SIGMA_END_VALUE = 0.2
ARCH.SIGMA_START_STEP = 0
ARCH.SIGMA_END_STEP = 10
# Latent dimensions
ARCH.Z_DYNA_DIM = 160 # 64
ARCH.Z_PRES_DIM = 1
ARCH.Z_SCALE_DIM = 2
ARCH.Z_SHIFT_DIM = 2
ARCH.Z_WHERE_DIM = 4
ARCH.Z_DEPTH_DIM = 1
ARCH.Z_WHAT_DIM = 16

ARCH.Z_DIM = ARCH.Z_PRES_DIM + ARCH.Z_DEPTH_DIM + ARCH.Z_WHERE_DIM + ARCH.Z_WHAT_DIM + ARCH.Z_DYNA_DIM
# 1 + 1 + 4 + 128 + 128
# ARCH.Z_DIM = ARCH.Z_PRES_DIM + ARCH.Z_WHERE_DIM + ARCH.Z_DEPTH_DIM + ARCH.Z_WHAT_DIM + ARCH.Z_DYNA_DIM
# Image shape and size
ARCH.IMG_SIZE = 64
ARCH.IMG_SHAPE = (ARCH.IMG_SIZE,) * 2
# Glimpse shape and size
ARCH.GLIMPSE_SIZE = 16
ARCH.GLIMPSE_SHAPE = (ARCH.GLIMPSE_SIZE,) * 2
# Img encoding dimension
ARCH.IMG_ENC_DIM = 128
# Glimpse encoding dimension
ARCH.GLIMPSE_ENC_DIM = 128
# Proposal encoding dimension
ARCH.PROPOSAL_ENC_DIM = 128
# Temporal rnn input dimension
ARCH.RNN_INPUT_DIM = 128
# Temporal rnn latent dimensions
ARCH.RNN_HIDDEN_DIM = 128
# Latent dimensions, for each entity k in K
ARCH.Z_MASK_DIM = 16
ARCH.Z_COMP_DIM = 32
#! --- Important hyper params ---
ARCH.FIX_ALPHA_STEPS = 0
ARCH.FIX_ALPHA_VALUES = 0.4
#! --- Background related ---
# Background rnn hidden dimension
ARCH.RNN_CTX_COMP_HIDDEN_DIM = 64
ARCH.RNN_CTX_COMP_PRIOR_HIDDEN_DIM = 32 # same as above - for kl
ARCH.RNN_CTX_MASK_HIDDEN_DIM = 32
ARCH.RNN_CTX_MASK_PRIOR_HIDDEN_DIM = 32 # same as above - for kl
# Hidden layer dim for the network that computes q(z_c|z_m, x)
ARCH.PREDICT_COMP_HIDDEN_DIM = 32

# Add conditional steps.
ARCH.COND_STEPS = 5
ARCH.FG_SAMPLE = True
ARCH.BG_SAMPLE = True
ARCH.NUM_GEN = 1
ARCH.INDICES = [0]

# Action dimension (robot variables)
ARCH.ACTION_DIM = 8
ARCH.Z_MASK_UPDATE_SCALE = 0.3
ARCH.Z_COMP_UPDATE_SCALE = 0.2
#! Robot layer related
ARCH.USE_AGENT_RNN_TRANSITION = False
ARCH.RESIDUAL_BACKGROUND = False
#! keypoint detection related
ARCH.KEYPOINTS = False
ARCH.NUM_KEYPOINTS = 32 # keypoint should be set near the gripper
ARCH.NUM_SAMPLES_FOR_BOM = 10
ARCH.USE_DETERMINISTIC_BELIEF = False
ARCH.KYPT_RNN_UNITS = 512
ARCH.KYPT_PRIOR_HIDDEN_DIM = 128
ARCH.KYPT_POST_HIDDEN_DIM = 128
ARCH.KYPT_Z_DIM = 16
ARCH.KYPT_SEP_LOSS_SCALE = 10.0
ARCH.KYPT_SEP_LOSS_SIGMA = 0.1
ARCH.KYPT_SIZE_SCALE = 1.0
ARCH.KYPT_KL_LOSS_SCALE = 1.0
ARCH.LR_MILESTONES = [20000]
ARCH.MODULE_TRAINING_SCHEME = [5000, 5500, 7000] # kypt_only, kypt+bg only, change input as bg
ARCH.EXP_NAME = 'test'

# ARCH.MODULE_TRAINING_SCHEME = [100, 200, 300] # kypt_only, kypt+bg only, change input as bg
ARCH.KYPT_ONLY_STEPS = 5000
ARCH.KYPT_BG_ONLY_STEPS = 5200
ARCH.HEATMAP_REG = 5.0
ARCH.ENTROPY_SCALE = 0.0

# Temperature for gumbel-softmax
ARCH.TAU_START_STEP = 0
ARCH.TAU_END_STEP = 10000
ARCH.TAU_START_VALUE = 1.0
ARCH.TAU_END_VALUE = 1.0
# Prior for scale in discovery
ARCH.Z_SCALE_MEAN_START_STEP = 0
ARCH.Z_SCALE_MEAN_END_STEP = 10000
ARCH.Z_SCALE_MEAN_START_VALUE = -1.5
ARCH.Z_SCALE_MEAN_END_VALUE = -1.5
ARCH.Z_SCALE_STD = 0.3
# Prior for z_shift
ARCH.Z_SHIFT_MEAN = 0.0
ARCH.Z_SHIFT_STD = 1.0
# Prior for presence in discovery
ARCH.Z_PRES_PROB_START_STEP = 0
ARCH.Z_PRES_PROB_END_STEP = 1500
ARCH.Z_PRES_PROB_START_VALUE = 1e-1
ARCH.Z_PRES_PROB_END_VALUE = 1e-8
# Update z_where and z_what
ARCH.PROPOSAL_UPDATE_MIN = 0.0
ARCH.PROPOSAL_UPDATE_MAX = 0.3
ARCH.Z_SHIFT_UPDATE_SCALE = 0.1
ARCH.Z_SCALE_UPDATE_SCALE = 0.3
ARCH.Z_WHAT_UPDATE_SCALE = 0.2
ARCH.Z_DEPTH_UPDATE_SCALE = 1.0
# Mlp layer sizesa in the mlp that compute the propagation map
ARCH.PROP_MAP_MLP_LAYERS = [128, 128]
# Propagation map depth/channels/dimensions
ARCH.PROP_MAP_DIM = 128
# This is used for the gaussian kernel
ARCH.PROP_MAP_SIGMA = 0.1
# Mlp layer sizesa in the mlp that compute the propagation map
ARCH.PROP_COND_MLP_LAYERS = [128, 128]
# Propagation conditioning vector depth/channels/dimensions
ARCH.PROP_COND_DIM = 128
# This is used in the gaussian kernel
ARCH.PROP_COND_SIGMA = 0.1
# Rejection
ARCH.REJECTION = True
ARCH.REJECTION_THRESHOLD = 0.6
# AOE
ARCH.BG_CONDITIONED = True
ARCH.BG_ATTENTION = True
ARCH.BG_PROPOSAL_DIM = 128
ARCH.BG_PROPOSAL_SIZE = 0.25
# Discovery Dropout
ARCH.DISCOVERY_DROPOUT = 0.5
# Auxiliary presence loss in propagation
ARCH.AUX_PRES_KL = True
# Action conditioning for robot agent.
ARCH.ACTION_COND = 'bg' # 'fg' / 'seperate' - where the action conditoined on.
# Params from Image shape
ARCH.IMAGE_SHAPE = (64, 64),
# Grid size. There will be G*G slots
ARCH.G = 4
# Background configurations
# ==== START ====
# Number of background components. If you set this to one, you should use a strong decoder instead.
ARCH.K =  3
# Background likelihood sigma
ARCH.BG_SIGMA = 0.15
ARCH.AGENT_SIGMA = 0.15

# Image encoding dimension
ARCH.BG_DECOMP = 'sbp' # 'sbp' / 'softmax'
ARCH.BG_UPDATE = 'full' # 'full' / 'residual'
ARCH.BG_TEMPORAL = 'step-wise'
ARCH.BG_COND_COMP = True
ARCH.SPATIO_TEMPORAL = 'cond_vrnn'

ARCH.USE_KNN = True
ARCH.PROP_COND_FEAT_DIM = ARCH.Z_PRES_DIM + ARCH.Z_DEPTH_DIM + ARCH.Z_WHERE_DIM + ARCH.Z_WHAT_DIM + ARCH.Z_DYNA_DIM + ARCH.RNN_HIDDEN_DIM
ARCH.KNN = 4
ARCH.BG_REG_MODE = 0
ARCH.PTH_START = 0
ARCH.KYPT_MASK_LOSS = True
ARCH.KYPT_MASK_LOSS_METHOD = 'MSE' # 'CE'
ARCH.KYPT_MASK_JOINT_LOSS = [False, False]
ARCH.KYPT_MASK_LOSS_LONG = True
ARCH.ENLARGE_KL_BG = False
ARCH.KL_BG_WEIGHT = 1.0
ARCH.KL_BG_WEIGHT_STEP = 0
ARCH.MASK_SCHEDULE = 17000

ARCH.SBP_INV_HEURISTIC = False

ARCH.ACTION_ENHANCE = 64

ARCH.AGENT_INTERACTION_MODE = 2
# 0 : Nothing
# 1 : bg_wo_agent -> bg as fg input (not a real action-centric interaction)
# 2 : Spatial Transform of raw sequence -> robot glimpse -> cat(glimpse, z_mask_rob, h_mask_rob) -> r-o enc

ARCH.AO_INTERACTION_MLP_LAYERS = [128, 128, 128]
ARCH.AO_INTERACTION_DIM = 64
ARCH.BG_ENCODER = 'vanilla'
ARCH.SOFTMAX_SCALE = 1.0
ARCH.AUTOREG_PRIORS = True

ARCH.KYPT_MIX_JOINT_UNTIL = 10000
ARCH.USING_SEQ = False
ARCH.RES_HIDDEN_DIM = 128
ARCH.IMG_ENC_HIDDEN_DIM = 128
ARCH.RESIDUAL_SCALE = 1.0
ARCH.ACTION_HIDDEN_DIM = 64
ARCH.LOCAL_AGENT_MLP_LAYERS = [128, 128, 128]
ARCH.AO_ATTENTION_MLP_LAYERS = [128, 128, 32]
ARCH.GLOBAL_AGENT_MLP_LAYERS = [128]
ARCH.DETACH_BG = False
ARCH.ACTION_ENHANCE_MODE = True
ARCH.IMG_ENC_HIDDEN_DIM = 128
ARCH.MASK_COND_HIDDEN_DIM = 128
ARCH.COMP_COND_HIDDEN_DIM = 128
ARCH.ACTION_HIDDEN_DIM = 64
ARCH.RES_HIDDEN_DIM = 128


#! Visualization related params
ARCH.IS_ON = True
ARCH.RESUME = False
ARCH.CKPT_START = 0
ARCH.TRAIN_PER_ITER = 200
ARCH.TOTAL_ITER = 10000

# hidden size of reward and action pred models
ARCH.DENSE_HIDDEN_DIM = 256

# memory management related
ARCH.PRETRAIN_DEMO = False
ARCH.PRETRAIN_GATSBI_UNTIL = 100000
ARCH.SLICE_DEMO_FROM = 100 # slice demo from...
# curriculum learning of gatsbi

ARCH.T = [2, 4, 6, 8, 10]
# When to change T, in terms of global steps
ARCH.T_MILESTONES = [10000, 12500, 15000, 17500]
ARCH.VIS_EVERY = 500
ARCH.TASK_NAME = 'clear_objects'
ARCH.SIM_TIMESTEP = 0.15
ARCH.EVAL_EPISODES = 50
ARCH.BATCH_SIZE = 4
ARCH.BATCH_LENGTH = 50
ARCH.HORIZON = 50
ARCH.IMAGINE_HORIZON = 5
ARCH.PREFILL_TIMESTEPS = 10000
ARCH.MAX_BUFFER_EPISODES = 500
ARCH.MAX_DEMO_EPISODES = 500
ARCH.DEBUG = False
ARCH.NGPU = 0
ARCH.PSEUDO_MASK_UNTIL = 40000
ARCH.LR = 3e-4
ARCH.HEADLESS = False
ARCH.AGENT_COMP_EMBED_SCALE = 0.0
ARCH.EMBED_SCALE = 1.0
ARCH.SSIM_SCALE = 1.0
ARCH.BCE = 'mean'
ARCH.MAX_PIX_THRESHOLD = 1e-4
ARCH.ALPHAMAP_REG = 100.0
ARCH.REJECT_ALPHA_START = 18000
ARCH.REJECT_ALPHA_UNTIL = 20000
ARCH.BG_KL_SCALE = 1.0
ARCH.INV_DYNA_HIDDEN_DIM = 128
ARCH.INV_ACT_SIGMA = 0.5
ARCH.INV_DYNA_SCALE = 1.0
ARCH.GRAD_CLIP = 1.0
ARCH.RESIDUAL_RES = 1.0
#! Residual related
ARCH.RESIDUAL_MASK_RES_START = 1.0
ARCH.RESIDUAL_MASK_RES_END = 0.5
ARCH.RESIDUAL_COMP_RES_START = 1.0    
ARCH.RESIDUAL_COMP_RES_END = 0.1    
ARCH.AGENT_DETECT_THRESHOLD = 0.1
ARCH.FREEZE_BATCHNORM_FROM = 50000
ARCH.LEARN_BATCH_NORM = True
ARCH.GATSBI = False
ARCH.SCENE_FILE = 'clear_objects.ttt'

ARCH.FILTER_GATSBI_AGENT = True

ARCH.AGENT_MASK_LATENT = True
ARCH.AGENT_COMP_LATENT = True
ARCH.AGENT_DEPTH_LATENT = True
ARCH.AGENT_POS_LATENT = True

ARCH.BG_MASK_LATENT = True
ARCH.BG_COMP_LATENT = True

ARCH.AGENT_MASK_HISTORY = True
ARCH.AGENT_COMP_HISTORY = True 

ARCH.BG_MASK_HISTORY = True
ARCH.BG_COMP_HISTORY = True 
# z_pres, z_depth, z_where, z_what, z_dyna
ARCH.OBJ_LATENT = [True, True, True, True, True]
ARCH.OBJ_OCC = True
ARCH.OBJ_HISTORY = True
ARCH.PRE_ACT_NORM_WEIGHT = 1.0
ARCH.FINETUNE_GATSBI_UNTIL = 120000

# train from long sequence
ARCH.TRAIN_LONG_TERM = False
ARCH.TRAIN_LONG_TERM_FROM = 15000

ARCH.DETACHED_T = [10, 15, 25, 35]
# When to change T, in terms of global steps
ARCH.DETACHED_T_MILESTONES = [30000, 50000, 70000]
ARCH.COMP_INPUT_REG_SCALE = 1.0
ARCH.TUNE_AGENT_MASK_FROM = 20000
ARCH.ATTENTION_DIM = 8 # 16, 32
ARCH.LR_DECAY_SCALE = 1.0 # 16, 32
ARCH.ALPHA_MAP_SIGMA = 32.0
ARCH.ARC_MARGIN = 0.5
ARCH.OCC_BERN_TEMP = 1.0
ARCH.SEED = 10000
ARCH.IMAGINE_TIMESTEP_CUTOFF = 5
ARCH.VIS_TIMESTEP_TRUNC = 200
ARCH.JOINT_TRAIN_GATSBI_START = 75000
ARCH.FIG_DIR = os.path.expanduser("~/rss22figs/")
ARCH.VISUALIZE = False
ARCH.AC_LR = 3e-4
ARCH.LAMBDA = 0.95
ARCH.VALUE_SIGMA = 1.0
ARCH.EXPLORATION_SIGMA = 0.3
ARCH.AC_GRAD_CLIP = 10.0
ARCH.POLYAK = 0.995
ARCH.UPDATE_TARGET_EVERY = 1
ARCH.REAL_WORLD = 'false'
ARCH.HIGH_LEVEL_HORIZON = 4
ARCH.BERN_DIFF = 1.0
ARCH.SUB_REWARD_REG_STEPS = 500
ARCH.MASK_DIFF_SCALE = 0.01
ARCH.HIGH_LEVEL_ACTOR_GRAD = 'dynamics' # 'reinforce'
ARCH.ENTROPY_DECAY_STEPS = 10000 # 'reinforce'
ARCH.EXP_ENTROPY = 1e-4

ARCH.Z_CTX_DIM = 128

cfg = Namespace(batch_size=20, 
                lr=1e-04, 
                num_cell_h=4, 
                num_cell_w=4, 
                max_num_obj=10,
                explained_ratio_threshold=0.2, 
                ratio_anc=1.0, 
                sigma=0.1, 
                size_anc=0.25, 
                var_anc=0.2, 
                var_s=0.15,  
                z_pres_anneal_end_value=0.0001,
                ckpt_dir='./model/', 
                color_num=500,
                cp=1.0, 
                epochs=4000, 
                gen_disc_pres_probs=0.1, 
                generate_freq=5000, 
                last_ckpt='', 
                nocuda=False, 
                num_img_summary=3, 
                observe_frames=5, 
                phase_conv_lstm=True, 
                phase_do_remove_detach=True, 
                phase_eval=True,  
                phase_generate=False, 
                phase_nll=False, 
                phase_no_background=False, 
                phase_parallel=False, 
                phase_simplify_summary=True, 
                print_freq=100, 
                remove_detach_step=30000, 
                save_epoch_freq=1000, 
                seed=666, 
                global_state=0, 
                summary_dir='./summary', 
                tau_end=0.3, 
                tau_ep=20000.0, 
                tau_imp=0.25)
