#!/usr/bin/env python
import inspect

import rospy
import numpy as np

from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, PoseStamped, PoseArray, Quaternion, PolygonStamped,Polygon, Point32, PoseWithCovarianceStamped, PointStamped
import tf.transformations
import tf
import matplotlib.pyplot as plt

def angle_to_quaternion(angle):
    """Convert an angle in radians into a quaternion _message_."""
    return Quaternion(*tf.transformations.quaternion_from_euler(0, 0, angle))

def quaternion_to_angle(q):
    """Convert a quaternion _message_ into an angle in radians.
    The angle represents the yaw.
    This is not just the z component of the quaternion."""
    x, y, z, w = q.x, q.y, q.z, q.w
    roll, pitch, yaw = tf.transformations.euler_from_quaternion((x, y, z, w))
    return yaw

def rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.matrix([[c, -s], [s, c]])

def particle_to_posestamped(particle, frame_id):
    pose = PoseStamped()
    pose.header = make_header(frame_id)
    pose.pose.position.x = particle[0]
    pose.pose.position.y = particle[1]
    pose.pose.orientation = angle_to_quaternion(particle[2])
    return pose

def particle_to_pose(particle):
    pose = Pose()
    pose.position.x = particle[0]
    pose.position.y = particle[1]
    pose.orientation = angle_to_quaternion(particle[2])
    return pose

def particles_to_poses(particles):
    return map(particle_to_pose, particles)

def make_header(frame_id, stamp=None):
    if stamp == None:
        stamp = rospy.Time.now()
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    return header

def point(npt):
    pt = Point32()
    pt.x = npt[0]
    pt.y = npt[1]
    return pt

def points(arr):
    return map(point, arr)

def map_to_world(poses,map_info):
    scale = map_info.resolution
    angle = quaternion_to_angle(map_info.origin.orientation)

    # rotate

    # rotation
    c, s = np.cos(angle), np.sin(angle)
    # we need to store the x coordinates since they will be overwritten
    temp = np.copy(poses[:,0])
    poses[:,0] = c*poses[:,0] - s*poses[:,1]
    poses[:,1] = s*temp       + c*poses[:,1]

    # scale
    poses[:,:2] *= float(scale)

    # translate
    poses[:,0] += map_info.origin.position.x
    poses[:,1] += map_info.origin.position.y
    poses[:,2] += angle

def world_to_map(poses, map_info):
    # equivalent to map_to_grid(world_to_map(poses))
    # operates in place
    scale = map_info.resolution
    angle = -quaternion_to_angle(map_info.origin.orientation)

    # translation
    poses[:,0] -= map_info.origin.position.x
    poses[:,1] -= map_info.origin.position.y

    # scale
    poses[:,:2] *= (1.0/float(scale))

    # rotation
    c, s = np.cos(angle), np.sin(angle)
    # we need to store the x coordinates since they will be overwritten
    temp = np.copy(poses[:,0])
    poses[:,0] = c*poses[:,0] - s*poses[:,1]
    poses[:,1] = s*temp       + c*poses[:,1]
    poses[:,2] += angle

def describe(var_list):
    # Name code taken from here: https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string/18425523
    def retrieve_name(var):
        callers_local_vars = inspect.currentframe().f_back.f_locals.items()
        return [var_name for var_name, var_val in callers_local_vars if var_val is var]

    for i in xrange(len(var_list)):
        name = retrieve_name(var_list[i])
        print("{} = {}".format(name, var_list[i]))

# Helper function adapted from: https://stackoverflow.com/questions/9269681/intersection-of-2d-numpy-ndarrays
def multidim_intersect(arr1, arr2):
    arr1_view = arr1.view([('',arr1.dtype)] * arr1.shape[1]).ravel()
    arr2_view = arr2.view([('',arr2.dtype)] * arr2.shape[1]).ravel()
    intersected = np.intersect1d(arr1_view, arr2_view).view(a.dtype).reshape(-1, a.shape[1])
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])
