#!/usr/bin/env python

import rospy
import numpy as np

from std_msgs.msg import Header
from geometry_msgs.msg import Quaternion 
from nav_msgs.srv import GetMap
import tf.transformations
import tf
import matplotlib.pyplot as plt

''' 
Convert an angle in radians into a quaternion message.
In:
    angle: The yaw angle in radians
Out:
    The Quaternion message
'''
def angle_to_quaternion(angle):
    return Quaternion(*tf.transformations.quaternion_from_euler(0, 0, angle))

''' 
Convert a quaternion message into an angle in radians.
In:
  q: The quaternion message
Out:
  The yaw angle
'''
def quaternion_to_angle(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    roll, pitch, yaw = tf.transformations.euler_from_quaternion((x, y, z, w))
    return yaw

''' 
Returns a rotation matrix that applies the passed angle (in radians)
In:
  theta: The desired rotation angle
Out:
  The corresponding rotation matrix
'''
def rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.matrix([[c, -s], [s, c]])

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
Convert a pose in the world to a pixel location in the map image 
In:
  pose: The pose in the world to be converted. Should be a list or tuple of the
        form [x,y,theta]
  map_info: Info about the map (returned by get_map)
Out:
  The corresponding pose in the pixel map - has the form [x,y,theta]
  where x and y are integers
'''    
def world_to_map(pose, map_info):
    scale = map_info.resolution
    angle = -quaternion_to_angle(map_info.origin.orientation)
    config = [0.0,0.0,0.0]
    # translation
    config[0] = (1.0/float(scale))*(pose[0] - map_info.origin.position.x)
    config[1] = (1.0/float(scale))*(pose[1] - map_info.origin.position.y)
    config[2] = pose[2]
   

    # rotation
    c, s = np.cos(angle), np.sin(angle)
    # we need to store the x coordinates since they will be overwritten
    temp = np.copy(config[0])
    config[0] = int(c*config[0] - s*config[1])
    config[1] = int(s*temp       + c*config[1])
    config[2] += angle
    
    return config
      
''' 
Convert a pixel location in the map to a pose in the world
In: 
  pose: The pixel pose in the map. Should be a list or tuple of the form [x,y,theta]
  map_info: Info about the map (returned by get_map)
Out:
  The corresponding pose in the world - has the form [x,y,theta]
'''
def map_to_world(pose,map_info):
    scale = map_info.resolution
    angle = quaternion_to_angle(map_info.origin.orientation)

    # rotate
    config = np.array([pose[0],map_info.height-pose[1],pose[2]])
    # rotation
    c, s = np.cos(angle), np.sin(angle)
    # we need to store the x coordinates since they will be overwritten
    temp = np.copy(config[0])
    config[0] = c*config[0] - s*config[1]
    config[1] = s*temp       + c*config[1]

    # scale
    config[:2] *= float(scale)

    # translate
    config[0] += map_info.origin.position.x
    config[1] += map_info.origin.position.y
    config[2] += angle

    return config
