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
class Robot(object):
    def __init__(self, name):
        self.name = name
        self.size_x = 10
        self.size_y = 10
        self.current_position = None
        self.current_velocity = None
        self.current_observed_voxels = None
        self.is_colliding = False
        self.accumulated_observed_voxels = 0
        self.default_action = [1e-5, 0, 0, 0, 0, 0]
        self.stuck_counter = 0
        self.current_TSDF_map = None
        self.observations = None
        return

    def initialize(self):
        pass

    def reset(self):
        self.accumulated_observed_voxels = 0
        self.observations = None
        self.current_position = None
        self.current_velocity = None
        self.is_colliding = False
        self.stuck_counter = 0
        self.current_TSDF_map = None
        return

    def move(self):
        pass

    def set_dynamics(self):
        pass

    def get_dynamics(self):
        pass

    def set_observations(self, observations):
        self.accumulated_observed_voxels += observations['newly_observed_voxels']
        if observations['collsion']:
            self.stuck_counter += 1
            self.is_colliding = True
        else:
            self.stuck_counter = 0
            self.is_colliding = False
        self.current_observed_voxels = observations['newly_observed_voxels']
        self.current_TSDF_map = observations['submap']
        self.observations = observations
        return

    def get_observations(self):
        return self.observations

    def is_stuck(self):
        if self.stuck_counter >= 1:
            return True
        return False