import numpy as np

class RandomPolicy(object):
    def __init__(self):
        self.high = np.array([10.0, 10.0, np.pi])
        self.low = np.array([-10.0, -10.0, -np.pi])
        self.randomization = 'unifrom'

    def predict(self, obs):
        if self.randomization == 'unifrom':
            return [np.random.uniform(low=self.low, high=self.high)], None
        else:
            raise Exception("Not implemented yet")
