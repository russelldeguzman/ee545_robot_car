#!/usr/bin/env python
import inspect

import rospy
import numpy as np

from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from nav_msgs.srv import GetMap
from geometry_msgs.msg import Point, Pose, PoseStamped, PoseArray, Quaternion, PolygonStamped,Polygon, Point32, PoseWithCovarianceStamped, PointStamped
import tf.transformations
import tf
import matplotlib.pyplot as plt
import torch

# Note that not all of these functions are necessary

'''
  Convert yaw angle in radians into a quaternion message
    angle: The yaw angle
    Returns: An equivalent geometry_msgs/Quaternion message
'''
def angle_to_quaternion(angle):
    return Quaternion(*tf.transformations.quaternion_from_euler(0, 0, angle))

'''
  Convert a quaternion message into a yaw angle in radians.
    q: A geometry_msgs/Quaternion message
    Returns: The equivalent yaw angle 
'''
def quaternion_to_angle(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    roll, pitch, yaw = tf.transformations.euler_from_quaternion((x, y, z, w))
    return yaw

'''
  Constructs a rotation matrix from a given angle in radians
    theta: The angle in radians
    Returns: The equivalent 2x2 numpy rotation matrix
'''
def rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.matrix([[c, -s], [s, c]])

'''
  Converts a particle to a pose message
    particle: The particle to convert - [x,y,theta]
    Returns: An equivalent geometry_msgs/Pose
'''
def particle_to_pose(particle):
    pose = Pose()
    pose.position.x = particle[0]
    pose.position.y = particle[1]
    pose.orientation = angle_to_quaternion(particle[2])
    return pose

'''
  Converts a list of particles to a list of pose messages
    particles: A list of particles, where each element is itself a list of the form [x,y,theta]
    Returns: A list of equivalent geometry_msgs/Pose messages
'''
def particles_to_poses(particles):
    return map(particle_to_pose, particles)

def particle_to_posestamped(particle, frame_id):
    pose = PoseStamped()
    pose.header = make_header(frame_id)
    pose.pose.position.x = particle[0]
    pose.pose.position.y = particle[1]
    pose.pose.orientation = angle_to_quaternion(particle[2])
    return pose

'''
  Creates a header with the given frame_id and stamp. Default value of stamp is
  None, which results in a stamp denoting the time at which this function was called
    frame_id: The desired coordinate frame
    stamp: The desired stamp
    Returns: The resulting header
'''
def make_header(frame_id, stamp=None):
    if stamp == None:
        stamp = rospy.Time.now()
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    return header

'''
  Converts a list with coordinates into a point message
  npt: A list of length two containing x and y coordinates
  Returns: A geometry_msgs/Point32 message
'''
def point(npt):
    pt = Point32()
    pt.x = npt[0]
    pt.y = npt[1]
    return pt

'''
  Converts a list of coordinates into a list of equivalent point messages
  arr: A list of coordinates, where each element is itself a two dimensional list
  Returns: A list of geometry_msgs/Point32 messages
'''
def points(arr):
    return map(point, arr)

''' Get the map from the map server
In:
  map_topic: The service topic that will provide the map
Out:
  map_img: A numpy array with dimensions (map_info.height, map_info.width). 
           A zero at a particular location indicates that the location is impermissible
           A one at a particular location indicates that the location is permissible
  map_info: Info about the map, see
            http://docs.ros.org/kinetic/api/nav_msgs/html/msg/MapMetaData.html 
            for more info    
'''
def get_map(map_topic):
  rospy.wait_for_service(map_topic)
  map_msg = rospy.ServiceProxy(map_topic, GetMap)().map
  array_255 = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))
  map_img = np.zeros_like(array_255, dtype=bool)
  map_img[array_255==0] = 1 
  
  return map_img, map_msg.info

''' 
Convert an array of pixel locations in the map to poses in the world. Does computations
in-place
  poses: Pixel poses in the map. Should be a nx3 numpy array
  map_info: Info about the map (returned by get_map)
'''
def map_to_world(poses,map_info):
    scale = map_info.resolution
    angle = quaternion_to_angle(map_info.origin.orientation)

    # Rotation
    c, s = np.cos(angle), np.sin(angle)
    
    # Store the x coordinates since they will be overwritten
    temp = np.copy(poses[:,0])
    poses[:,0] = c*poses[:,0] - s*poses[:,1]
    poses[:,1] = s*temp      + c*poses[:,1]

    # Scale
    poses[:,:2] *= float(scale)

    # Translate
    poses[:,0] += map_info.origin.position.x
    poses[:,1] += map_info.origin.position.y
    poses[:,2] += angle

''' 
Convert array of poses in the world to pixel locations in the map image 
  pose: The poses in the world to be converted. Should be a nx3 numpy array
  map_info: Info about the map (returned by get_map)
'''    
def world_to_map(poses, map_info):
   
    scale = map_info.resolution
    angle = -quaternion_to_angle(map_info.origin.orientation)

    # Translation
    poses[:,0] -= map_info.origin.position.x
    poses[:,1] -= map_info.origin.position.y

    # Scale
    poses[:,:2] *= (1.0/float(scale))

    # Rotation
    c, s = np.cos(angle), np.sin(angle)
    
    # Store the x coordinates since they will be overwritten
    temp = np.copy(poses[:,0])
    poses[:,0] = c*poses[:,0] - s*poses[:,1]
    poses[:,1] = s*temp       + c*poses[:,1]
    poses[:,2] += angle

''' 
Convert array of poses in the world to pixel locations in the map image 
  pose: The poses in the world to be converted. Should be a nx3 numpy array
  map_info: Info about the map (returned by get_map)
'''    
def world_to_map_torch(poses, map_info, device):
    map_poses = poses[:, :2].clone()
    scale = torch.tensor(map_info.resolution, dtype=torch.float, device=device)
    angle = torch.tensor(-quaternion_to_angle(map_info.origin.orientation), dtype=torch.float, device=device)

    # Translation
    map_poses[:,0] -= torch.tensor(map_info.origin.position.x, dtype=torch.float, device=device)
    map_poses[:,1] -= torch.tensor(map_info.origin.position.y, dtype=torch.float, device=device)

    # Scale
    map_poses /= scale

    if angle == 0:
      map_poses[:, 0] = map_poses[:, 0].clamp(0, map_info.width - 1)
      map_poses[:, 1] = map_poses[:, 1].clamp(0, map_info.height - 1)
      map_poses = map_poses.type(dtype=torch.int32)
      return map_poses

    # Rotation
    c, s = torch.cos(angle), torch.sin(angle)
    
    # Store the x coordinates since they will be overwritten
    temp = map_poses[:,0].clone()
    map_poses[:,0] = c*map_poses[:,0] - s*map_poses[:,1]
    map_poses[:,1] = s*temp + c*map_poses[:,1]
    map_poses[:, 0] = map_poses[:, 0].clamp(0, map_info.height - 1)
    map_poses[:, 1] = map_poses[:, 1].clamp(0, map_info.width - 1)
    map_poses = map_poses.type(dtype=torch.int32)
    return map_poses

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
