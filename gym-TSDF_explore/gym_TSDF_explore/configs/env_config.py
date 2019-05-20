from yacs.config import CfgNode as CN
import numpy as np

# pose_multi_resoluton_net related params
ENV_CONFIG = CN()
ENV_CONFIG.SCALING_FACTORS = [10.0, 10.0, np.pi]