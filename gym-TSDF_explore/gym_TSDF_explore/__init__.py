"""
 *!
 * @author    Ossama Ahmed
 * @email     oahmed@ethz.ch
 *
 * Copyright (C) 2019 Autonomous Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.asl.ethz.ch/
 *
 """
import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='TSDF_explore-v0',
    entry_point='gym_TSDF_explore.environment:RobotEnv'
)