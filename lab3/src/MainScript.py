#!/usr/bin/env python

from __future__ import division

import rospy
import numpy as np
import time
import utils as Utils
import tf.transformations
import tf
from threading import Lock

from nav_msgs.srv import GetMap
from geometry_msgs.msg import PoseStamped, PoseArray, PoseWithCovarianceStamped, PointStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from ReSample import ReSampler
from SensorModel import SensorModel
from MotionModel import KinematicMotionModel

MAP_TOPIC = "static_map"
PUBLISH_PREFIX = '/pf/viz'
PUBLISH_TF = True