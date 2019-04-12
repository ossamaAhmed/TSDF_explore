from algorithms.encoding_state import encodingState
from datasets_processing.SDF import SDF
import rospy
from voxblox_rl_simulator.srv import *
from geometry_msgs.msg import Vector3, Quaternion, Transform, TransformStamped
import numpy as np


def _send_transform():
    t = TransformStamped()
    t.header.stamp = rospy.Time().now()
    t.header.frame_id = "velodyne"
    t.child_frame_id = 'obs%d' % 1
    t.transform.translation.x = 0
    t.transform.translation.y = 0
    t.transform.translation.z = 0
    t.transform.rotation.x = 0
    t.transform.rotation.y = 0
    t.transform.rotation.z = 0
    t.transform.rotation.w = 0
    return t

# Transform(
#             Vector3(0,
#                     0,
#                     0),
#             Quaternion(0,
#                        0,
#                        0,
#                        0)
#         )


def client_stuff():
    t = _send_transform
    rospy.wait_for_service('/voxblox_rl_simulator/simulation/move')
    try:
        random_walk_result = rospy.ServiceProxy('/voxblox_rl_simulator/simulation/move', Move)
        resp = random_walk_result(T_C_old_C_new=Transform(
                                    Vector3(0,
                                            0,
                                            0.5),
                                    Quaternion(1.0,
                                               0,
                                               0,
                                               0)
        ))
        map = np.array(resp.submap.data)
        print(map.shape)
        #resp.submap.layout.dim[0].size
        return resp
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
    return


def main():
    client_stuff()
    # state_model = encodingState()
    # state_model.train()
    # dataset = SDF()

if __name__ == "__main__":
    main()