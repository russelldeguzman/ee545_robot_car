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

'''
  Implements particle filtering for estimating the state of the robot car
'''
class ParticleFilter():


  '''
  Initializes the particle filter
    n_particles: The number of particles
    n_viz_particles: The number of particles to visualize
    motor_state_topic: The topic containing motor state information
    servo_state_topic: The topic containing servo state information
    scan_topic: The topic containing laser scans
    laser_ray_step: Step for downsampling laser scans
    exclude_max_range_rays: Whether to exclude rays that are beyond the max range
    max_range_meters: The max range of the laser
    resample_type: Whether to use naiive or low variance sampling
    speed_to_erpm_offset: Offset conversion param from rpm to speed
    speed_to_erpm_gain: Gain conversion param from rpm to speed
    steering_angle_to_servo_offset: Offset conversion param from servo position to steering angle
    steering_angle_to_servo_gain: Gain conversion param from servo position to steering angle
    car_length: The length of the car
  '''
  def __init__(self, n_particles, n_viz_particles,
               motor_state_topic, servo_state_topic, scan_topic, laser_ray_step,
               exclude_max_range_rays, max_range_meters, resample_type,
               speed_to_erpm_offset, speed_to_erpm_gain, steering_angle_to_servo_offset,
               steering_angle_to_servo_gain, car_length):
    self.N_PARTICLES = n_particles # The number of particles
                                   # In this implementation, the total number of
                                   # particles is constant
    self.N_VIZ_PARTICLES = n_viz_particles # The number of particles to visualize

    self.particle_indices = np.arange(self.N_PARTICLES) # Cached list of particle indices
    self.particles = np.zeros((self.N_PARTICLES,3)) # Numpy matrix of dimension N_PARTICLES x 3
    self.weights = np.ones(self.N_PARTICLES) / float(self.N_PARTICLES) # Numpy matrix containig weight for each particle

    self.state_lock = Lock() # A lock used to prevent concurrency issues. You do not need to worry about this

    self.tfl = tf.TransformListener() # Transforms points between coordinate frames
    self.permit_coords = None
    self.N_PARTICLE_ANGLES = 8  # Number of variations in angle that will be instantiated as different particles for each sampled [x, y] coord during intialize_global

    # Get the map
    print("Getting map from service: ", MAP_TOPIC)
    rospy.wait_for_service(MAP_TOPIC)
    map_msg = rospy.ServiceProxy(MAP_TOPIC, GetMap)().map # The map, will get passed to init of sensor model
    self.map_info = map_msg.info # Save info about map for later use


    # Create numpy array representing map for later use
    array_255 = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))
    self.permissible_region = np.zeros_like(array_255, dtype=bool)
    self.permissible_region[array_255 == 0] = 1 # Numpy array of dimension (map_msg.info.height, map_msg.info.width),
                                              # With values 0: not permissible, 1: permissible
    x_perm, y_perm = np.where(self.permissible_region == 1)
    self.permit_coords = np.vstack((x_perm, y_perm)).T
    # Globally initialize the particles
    self.initialize_global()

    # Publish particle filter state
    self.pub_tf = tf.TransformBroadcaster() # Used to create a tf between the map and the laser for visualization
    self.pose_pub      = rospy.Publisher(PUBLISH_PREFIX + "/inferred_pose", PoseStamped, queue_size = 1) # Publishes the expected pose
    self.particle_pub  = rospy.Publisher(PUBLISH_PREFIX + "/particles", PoseArray, queue_size = 1) # Publishes a subsample of the particles
    self.pub_laser     = rospy.Publisher(PUBLISH_PREFIX + "/scan", LaserScan, queue_size = 1) # Publishes the most recent laser scan
    self.pub_odom      = rospy.Publisher(PUBLISH_PREFIX + "/odom", Odometry, queue_size = 1) # Publishes the path of the car

    self.RESAMPLE_TYPE = resample_type # Whether to use naiive or low variance sampling
    self.resampler = ReSampler(self.particles, self.weights, self.state_lock)  # An object used for resampling

    # An object used for applying sensor model
    self.sensor_model = SensorModel(scan_topic, laser_ray_step, exclude_max_range_rays,
                                    max_range_meters, map_msg, self.particles, self.weights,
                                    self.state_lock)

    # An object used for applying kinematic motion model
    self.motion_model = KinematicMotionModel(motor_state_topic, servo_state_topic,
                                             speed_to_erpm_offset, speed_to_erpm_gain,
                                             steering_angle_to_servo_offset, steering_angle_to_servo_gain,
                                             car_length, self.particles, self.state_lock)

    # Subscribe to the '/initialpose' topic. Publised by RVIZ. See clicked_pose_cb function in this file for more info
    self.pose_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.clicked_pose_cb, queue_size=1)
    print('Initialization complete')

  '''
    Initialize the particles as uniform samples across the in-bounds regions of
    the map
  '''
  def initialize_global(self):
    self.state_lock.acquire()

    # Use self.permissible_region to get in-bounds states
    # Uniformally sample from in-bounds regions
    # Convert map samples (which are in pixels) to world samples (in meters/radians)
    #   Take a look at utils.py
    # Update particles in place
    # Update weights in place so that all particles have the same weight and the
    # sum of the weights is one.
    # YOUR CODE HERE
    n_pts = int(self.N_PARTICLES / self.N_PARTICLE_ANGLES)
    permit_inds = np.random.choice(self.permit_coords.shape[0], size=n_pts, replace=False)  # Uniformly samples from all permitted coordinates
    angles, step = np.linspace(0, 2*np.pi, self.N_PARTICLE_ANGLES, retstep=True)
    rand_offset = np.random.uniform(low=0.0, high=step)
    angles += rand_offset
    angles = ((angles + np.pi) % (2*np.pi)) - np.pi # Maps angles to interval [-pi : pi)
    # Build an array of particles that creates all combinations of [x,y] and the thetas in angles[:]
    self.particles[0:n_pts*self.N_PARTICLE_ANGLES, :2] = np.tile(self.permit_coords[permit_inds], (self.N_PARTICLE_ANGLES, 1))
    self.particles[0:n_pts*self.N_PARTICLE_ANGLES, 2] = np.repeat(angles, n_pts)
    Utils.map_to_world(self.particles, self.map_info)
    self.weights[:] = [1 / float(self.N_PARTICLES)]
    # Utils.map_to_world(self.particles, self.map_info)
    # self.weights = np.ones(self.particles.shape[0]) / float(self.particles.shape[0])
    self.state_lock.release()

  '''
    Publish a tf between the laser and the map
    This is necessary in order to visualize the laser scan within the map
      pose: The pose of the laser w.r.t the map
      stamp: The time at which this pose was calculated, defaults to None - resulting
             in using the time at which this function was called as the stamp
  '''
  def publish_tf(self, pose, stamp=None):
    if stamp is None:
        stamp = rospy.Time.now()
    try:
      # Lookup the offset between laser and odom
      delta_off, delta_rot = self.tfl.lookupTransform("/laser","/odom",rospy.Time(0))

      # Transform offset to be w.r.t the map
      off_x = delta_off[0]*np.cos(pose[2]) - delta_off[1]*np.sin(pose[2])
      off_y = delta_off[0]*np.sin(pose[2]) + delta_off[1]*np.cos(pose[2])

      # Broadcast the tf
      self.pub_tf.sendTransform((pose[0]+off_x,pose[1]+off_y,0.0),tf.transformations.quaternion_from_euler(0,0,pose[2]+tf.transformations.euler_from_quaternion(delta_rot)[2]),stamp,"/odom","/map")

    except (tf.LookupException): # Will occur if odom frame does not exist
      self.pub_tf.sendTransform((pose[0],pose[1],0),tf.transformations.quaternion_from_euler(0,0,pose[2]), stamp , "/laser", "/map")

    """ From MIT Racecar Github
    Our particle filter provides estimates for the "laser" frame
    since that is where our laser range estimates are measure from. Thus,
    We want to publish a "map" -> "laser" transform.
    However, the car's position is measured with respect to the "base_link"
    frame (it is the root of the TF tree). Thus, we should actually define
    a "map" -> "base_link" transform as to not break the TF tree.
    """

    # # also publish odometry to facilitate getting the localization pose
    # if self.PUBLISH_ODOM:
    #     odom = Odometry()
    #     odom.header = Utils.make_header("/map", stamp)
    #     odom.pose.pose.position.x = pose[0]
    #     odom.pose.pose.position.y = pose[1]
    #     odom.pose.pose.orientation = Utils.angle_to_quaternion(pose[2])
    #     self.odom_pub.publish(odom)
    
    # return # below this line is disabled

    # """
    # Our particle filter provides estimates for the "laser" frame
    # since that is where our laser range estimates are measure from. Thus,
    # We want to publish a "map" -> "laser" transform.
    # However, the car's position is measured with respect to the "base_link"
    # frame (it is the root of the TF tree). Thus, we should actually define
    # a "map" -> "base_link" transform as to not break the TF tree.
    # """

    # Get map -> laser transform.
    map_laser_pos = np.array( (pose[0],pose[1],0) )
    map_laser_rotation = np.array( tf.transformations.quaternion_from_euler(0, 0, pose[2]) )
    # Apply laser -> base_link transform to map -> laser transform
    # This gives a map -> base_link transform
    laser_base_link_offset = (0.265, 0, 0)
    map_laser_pos -= np.dot(tf.transformations.quaternion_matrix(tf.transformations.unit_vector(map_laser_rotation))[:3,:3], laser_base_link_offset).T

    # Publish transform
    self.pub_tf.sendTransform(map_laser_pos, map_laser_rotation, stamp , "/base_link", "/map")

  '''
    Returns a 3 element numpy array representing the expected pose given the
    current particles and weights
    Uses weighted cosine and sine averaging to more accurately compute average theta
      https://en.wikipedia.org/wiki/Mean_of_circular_quantities
  '''
  def expected_pose(self):
    # YOUR CODE HERE
    assert np.allclose(np.sum(self.weights), 1), "self.weights does not sum to 1"
    x = np.sum([self.weights[i] * self.particles[i][0] for i in range(len(self.particles))])
    y = np.sum([self.weights[i] * self.particles[i][1] for i in range(len(self.particles))])
    theta = np.arctan2(np.sum([np.sin(self.particles[i][2]) for i in range(len(self.particles))]), np.sum([np.cos(self.particles[i][2]) for i in range(len(self.particles))]))
    return np.array([x, y, theta])

  '''
    Callback for '/initialpose' topic. RVIZ publishes a message to this topic when you specify an initial pose
    using the '2D Pose Estimate' button
    Reinitialize particles and weights according to the received initial pose
  '''
  def clicked_pose_cb(self, msg):
    self.state_lock.acquire()
    # Sample particles from a gaussian centered around the received pose
    # Updates the particles in place
    # Updates the weights to all be equal, and sum to one
    # YOUR CODE HERE
    # rospy.loginfo("this is the msg.pose.position")
    # rospy.loginfo(msg.pose.x)
    # rospy.loginfo(msg.pose.x)

    rcvd_pose_x = msg.pose.pose.position.x
    rcvd_pose_y = msg.pose.pose.position.y
    rcvd_pose_theta = Utils.quaternion_to_angle(msg.pose.pose.orientation) 

    map_pose = np.array((rcvd_pose_x, rcvd_pose_y, rcvd_pose_theta), ndmin = 2 )
    Utils.world_to_map(map_pose, self.map_info)

    for i in range(len(self.particles)):
      # rospy.loginfo("This is the beginning of the for loop")
      in_bounds = 0
      x_samp = 0
      y_samp = 0
      while not in_bounds:
        # rospy.loginfo("This is if the position is not in bounds") 
        x_samp = int (np.random.normal(map_pose[:,0], .01) )# good std deviation?
        y_samp = int (np.random.normal(map_pose[:,1], .01) ) # good std deviation?
        in_bounds = self.permissible_region[y_samp][x_samp]
        # rospy.loginfo(x_samp) 
        # rospy.loginfo(y_samp)
        # rospy.loginfo(in_bounds)

      self.particles[i] = [x_samp,y_samp,map_pose[:,2]]
      # rospy.loginfo("self.particles.shape")
      # rospy.loginfo(self.particles.shape)
      # rospy.loginfo("This is the end of the for loop")

    Utils.map_to_world(self.particles,self.map_info)
    self.weights[:] = [1 / float(len(self.particles))]
    self.state_lock.release()

  '''
    Visualize the current state of the filter
   (1) Publishes a tf between the map and the laser. Necessary for visualizing the laser scan in the map
   (2) Publishes the most recent laser measurement. Note that the frame_id of this message should be '/laser'
   (3) Publishes a PoseStamped message indicating the expected pose of the car
   (4) Publishes a subsample of the particles (use self.N_VIZ_PARTICLES).
       Sample so that particles with higher weights are more likely to be sampled.
  '''
  def visualize(self):
    #print 'Visualizing...'
    self.state_lock.acquire()
    self.inferred_pose = self.expected_pose()

    if isinstance(self.inferred_pose, np.ndarray):
      if PUBLISH_TF:
        self.publish_tf(self.inferred_pose)
      ps = PoseStamped()
      ps.header = Utils.make_header("map")
      ps.pose.position.x = self.inferred_pose[0]
      ps.pose.position.y = self.inferred_pose[1]
      ps.pose.orientation = Utils.angle_to_quaternion(self.inferred_pose[2])
      if(self.pose_pub.get_num_connections() > 0):
        self.pose_pub.publish(ps)
      if(self.pub_odom.get_num_connections() > 0):
        odom = Odometry()
        odom.header = ps.header
        odom.pose.pose = ps.pose
        self.pub_odom.publish(odom)

    if self.particle_pub.get_num_connections() > 0:
      if self.particles.shape[0] > self.N_VIZ_PARTICLES:
        # randomly downsample particles
        proposal_indices = np.random.choice(self.particle_indices, size=self.N_VIZ_PARTICLES, replace=False, p=self.weights)
        # proposal_indices = np.random.choice(self.particle_indices, self.N_VIZ_PARTICLES)
        self.publish_particles(self.particles[proposal_indices,:])
      else:
        self.publish_particles(self.particles)

    if self.pub_laser.get_num_connections() > 0 and isinstance(self.sensor_model.last_laser, LaserScan):
      self.sensor_model.last_laser.header.frame_id = "/laser"
      self.sensor_model.last_laser.header.stamp = rospy.Time.now()
      self.pub_laser.publish(self.sensor_model.last_laser)
    self.state_lock.release()

  '''
  Helper function for publishing a pose array of particles
    particles: To particles to publish
  '''
  def publish_particles(self, particles):
    pa = PoseArray()
    pa.header = Utils.make_header("map")
    pa.poses = Utils.particles_to_poses(particles)
    self.particle_pub.publish(pa)

# Suggested main
if __name__ == '__main__':
  rospy.init_node("particle_filter", anonymous=True) # Initialize the node

  n_particles = int(rospy.get_param("~n_particles")) # The number of particles
  n_viz_particles = int(rospy.get_param("~n_viz_particles")) # The number of particles to visualize
  motor_state_topic = rospy.get_param("~motor_state_topic", "/vesc/sensors/core") # The topic containing motor state information
  servo_state_topic = rospy.get_param("~servo_state_topic", "/vesc/sensors/servo_position_command") # The topic containing servo state information
  scan_topic = rospy.get_param("~scan_topic", "/scan") # The topic containing laser scans
  laser_ray_step = int(rospy.get_param("~laser_ray_step")) # Step for downsampling laser scans
  exclude_max_range_rays = bool(rospy.get_param("~exclude_max_range_rays")) # Whether to exclude rays that are beyond the max range
  max_range_meters = float(rospy.get_param("~max_range_meters")) # The max range of the laser
  resample_type = rospy.get_param("~resample_type", "naiive") # Whether to use naiive or low variance sampling

  speed_to_erpm_offset = float(rospy.get_param("/vesc/speed_to_erpm_offset", 0.0)) # Offset conversion param from rpm to speed
  speed_to_erpm_gain = float(rospy.get_param("/vesc/speed_to_erpm_gain", 4350))   # Gain conversion param from rpm to speed
  steering_angle_to_servo_offset = float(rospy.get_param("/vesc/steering_angle_to_servo_offset", 0.5)) # Offset conversion param from servo position to steering angle
  steering_angle_to_servo_gain = float(rospy.get_param("/vesc/steering_angle_to_servo_gain", -1.2135)) # Gain conversion param from servo position to steering angle
  car_length = float(rospy.get_param("/car_kinematics/car_length", 0.33)) # The length of the car

  # Create the particle filter
  pf = ParticleFilter(n_particles, n_viz_particles,
                      motor_state_topic, servo_state_topic, scan_topic, laser_ray_step,
                      exclude_max_range_rays, max_range_meters, resample_type,
                      speed_to_erpm_offset, speed_to_erpm_gain, steering_angle_to_servo_offset,
                      steering_angle_to_servo_gain, car_length)

  while not rospy.is_shutdown(): # Keep going until we kill it
    # Callbacks are running in separate threads
    if pf.sensor_model.do_resample: # Check if the sensor model says it's time to resample
      pf.sensor_model.do_resample = False # Reset so that we don't keep resampling

      # Resample
      if pf.RESAMPLE_TYPE == "naiive":
        pf.resampler.resample_naiive()
      elif pf.RESAMPLE_TYPE == "low_variance":
        pf.resampler.resample_low_variance()
      else:
        print "Unrecognized resampling method: "+ pf.RESAMPLE_TYPE

      pf.visualize() # Perform visualization
