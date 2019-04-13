import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='TSDF_explore-v0',
    entry_point='gym_TSDF_explore.environment:RobotEnv'
)