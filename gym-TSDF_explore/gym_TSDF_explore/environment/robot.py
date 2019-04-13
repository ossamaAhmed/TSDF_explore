
class Robot(object):
    def __init__(self, name):
        self.name = name
        self.size_x = 10
        self.size_y = 10
        self.current_position = None
        self.current_velocity = None
        self.motor_off = None
        self.current_observed_voxels = None
        self.discovered_voxels = None
        self.is_colliding = False
        return

    def initialize(self):
        pass

    def reset(self):
        pass

    def move(self):

        pass

    def set_dynamics(self):
        pass

    def get_dynamics(self):
        pass

    def set_observations(self):
        pass

    def get_observations(self):
        pass

    def stop_movement(self):
        pass

    def is_stationary(self):
        pass