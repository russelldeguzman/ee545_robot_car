#!/usr/bin/env python

import rospy
import numpy as np
import math
import sys
import rosbag
import utils
from collections import deque
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped, PoseArray, Pose

SCAN_TOPIC = '/scan' # The topic to subscribe to for laser scans
CMD_TOPIC = '/vesc/high_level/ackermann_cmd_mux/input/nav_0' # The topic to publish controls to
POSE_TOPIC = 'pf/viz/inferred_pose' # The topic to subscribe to for current pose of the car (From the particle filter)
                                  # NOTE THAT THIS IS ONLY NECESSARY FOR VIZUALIZATION
# POSE_TOPIC = '/sim_car_pose/pose'
VIZ_TOPIC = '/mpc_controller/rollouts' # The topic to publish to for vizualizing
                                       # the computed rollouts. Publish a PoseArray.

PLAN_PUB_TOPIC = "/planner_node/full_car_plan"

MAX_PENALTY = 10000 # The penalty to apply when a configuration in a rollout
                    # goes beyond the corresponding laser scan
'''
Wanders around using minimum (steering angle) control effort while avoiding crashing
based off of laser scans.
'''
class MPCController:

  '''
  Initializes the MPCController
    rollouts: An NxTx3 numpy array that contains N rolled out trajectories, each
              containing T poses. For each trajectory, the t-th element represents
              the [x,y,theta] pose of the car at time t+1
    deltas: An N dimensional array containing the possible steering angles. The n-th
            element of this array is the steering angle that would result in the
            n-th trajectory in rollouts
    speed: The speed at which the car should travel
    compute_time: The amount of time (in seconds) we can spend computing the cost
    laser_offset: How much to shorten the laser measurements
  '''
  def __init__(self, rollouts, deltas, speed, compute_time, laser_offset, laser_window, delta_incr, lookahead_distance, plan):
    # Store the params for later
    self.rollouts = rollouts
    self.deltas = deltas
    self.speed = speed
    self.compute_time = compute_time
    self.laser_offset = laser_offset
    self.laser_window = laser_window
    self.delta_incr = delta_incr
    self.lookahead_distance = lookahead_distance
    self.plan = plan
    self.prev_angle = None
    self.current_pose = None
    self.flag = True
    # YOUR CODE HERE
    self.cmd_pub = rospy.Publisher(CMD_TOPIC, AckermannDriveStamped, queue_size = 1)
    self.laser_sub = rospy.Subscriber(SCAN_TOPIC, LaserScan, self.wander_cb)
    self.viz_pub = rospy.Publisher(VIZ_TOPIC, PoseArray, queue_size = 1) # Create a publisher for vizualizing trajectories. Will publish PoseArrays
    self.viz_sub = rospy.Subscriber(POSE_TOPIC, PoseStamped, self.vizsub_cb) # Create a subscriber to the current position of the car
    self.plan_pub = rospy.Publisher(PLAN_PUB_TOPIC, PoseArray, queue_size=1)
    self.goal_pub = rospy.Publisher('/mpc_controller/current_goal', PoseStamped, queue_size=1)
    # NOTE THAT THIS VIZUALIZATION WILL ONLY WORK IN SIMULATION. Why?


# This should always return the current goal we're navigating to
# it should only update once we've achieved that goal

# Having a fixed look ahead - of distance
# Find the closest pose in plan to the current_pose -
# Calculate a fixed distance ahead of this pose


  def publish_full_car_plan(self, msg):
    self.plan_pub.publish(msg)

  '''
  Vizualize the rollouts. Transforms the rollouts to be in the frame of the world.
  Only display the last pose of each rollout to prevent lagginess
    msg: A PoseStamped representing the current pose of the car
  '''
  def vizsub_cb(self, msg):
    # Create the PoseArray to publish. Will contain N poses, where the n-th pose
    # represents the last pose in the n-th trajectory
    pa = PoseArray()
    pa.header.frame_id = '/map'
    pa.header.stamp = rospy.Time.now()
    self.current_pose = [msg.pose.position.x,msg.pose.position.y,utils.quaternion_to_angle(msg.pose.orientation)]
    # Transform the last pose of each trajectory to be w.r.t the world and insert into
    # the pose array
    # YOUR CODE HERE
    for n in range(self.rollouts.shape[0]):
        pose = Pose()
        pose.orientation = utils.angle_to_quaternion(self.rollouts[n][-1][2])
        pose.position.x = self.rollouts[n][-1][0] + self.current_pose[0]
        pose.position.y = self.rollouts[n][-1][1] + self.current_pose[1]
        pose.position.z = 0
        pa.poses.append(pose)

    self.viz_pub.publish(pa)

  """
  current pose: [x,y,theta]
  rollout pose: [x,y,theta]
  return angle between current pose's x-axis and the rollout pose
  """
  def _compute_pose_angle(self, current_pose, rollout_pose):
    delta_x = rollout_pose[0] - current_pose[0]
    delta_y = rollout_pose[1] - current_pose[1]
    return np.arctan(delta_y/delta_x)

  def idx_at_dist(self, lookahead_distance):
    dist = 0
    if(len(self.plan)==1):
      return 0
    for i in xrange(len(self.plan)-1) :
      dist  += np.linalg.norm( np.array(self.plan[i+1][:-1]) - np.array(self.plan[i][:-1]))  
      if (dist>lookahead_distance):
        break
    return i

  def get_next_pose(self):
   while True:
      try:

        [target_x, target_y, target_th] = self.plan.popleft()

        # Vector between ith pose and the current car pse
        vectorC2P = [target_x - self.current_pose[0], target_y - self.current_pose[1]]

        # Unit vector in the direction of the car pose 
        carPoseVector = [np.cos(self.current_pose[2]), np.sin(self.current_pose[2])]
        dotProduct = np.dot(vectorC2P, carPoseVector)

        # If dot product is positive value, then the target node is in front 
        # Break out of the while loop if dot product > -0.2 (i.e. target node is in front)
        if dotProduct > -0.2:   # Counting a point as "in front" when the dot product is slightly negative helps make sure the robot doesn't stop slightly short of the last pose in the array
          self.plan.appendleft(np.array([target_x, target_y, target_th])) # If it turns out the point was in front, put it back in the deque in case it's still in front at the next timestep
          rospy.loginfo("Quiting While Loop - Point forward of car reached")
          rospy.loginfo("self.plan length") 
          rospy.loginfo(len(self.plan))
          break
      except IndexError:
        rospy.loginfo("THIS SHOULD BE THE END OF MSGs")
        return [540, 835, 4.1 ]

    # Plain Lookahead
    # goal_idx = int(min(0+self.lookahead_distance, len(self.plan)-1))

    # Distance Lookahead
   goal_idx = int (self.idx_at_dist(self.lookahead_distance) )   

   if self.plan != None:
    rospy.loginfo("Goal pose from get_next_pose")
    rospy.loginfo(self.plan[goal_idx])
    return self.plan[goal_idx]

   return [540, 835, 4.1]

  '''
  Compute the cost of one step in the trajectory. It should penalize the magnitude
  of the steering angle. It should also heavily penalize crashing into an object
  (as determined by the laser scans)
    delta: The steering angle that corresponds to this trajectory
    rollout_pose: The pose in the trajectory
    laser_msg: The most recent laser scan
  '''
  def compute_cost(self, delta, rollout_pose, laser_msg, plan_pose):

    # Initialize the cost to be the euclid distance from the objective
    # Consider the line that goes from the robot to the rollout pose
    # Compute the angle of this line with respect to the robot's x axis
    # Find the laser ray that corresponds to this angle
    # Add MAX_PENALTY to the cost if the distance from the robot to the rollout_pose
    # is greater than the laser ray measurement - np.abs(self.laser_offset)
    # Return the resulting cost
    # Things to think about:
    #   What if the angle of the pose is less (or greater) than the angle of the
    #   minimum (or maximum) laser scan angle
    #   What if the corresponding laser measurement is NAN?
    # NOTE THAT NO COORDINATE TRANSFORMS ARE NECESSARY INSIDE OF THIS FUNCTION
    # YOUR CODE HERE

    #intialize the cost to be the euclid distance from the objective
    if self.flag:
        self.flag = False
        ANGLE_FACTOR  = 1000
        direction_vec = [plan_pose[0]-self.current_pose[0], plan_pose[1]-self.current_pose[1]]
        direction_unit_vec = direction_vec/np.linalg.norm(direction_vec)
        angle_goal = np.arctan(direction_unit_vec[0], direction_unit_vec[1])
        cost = ANGLE_FACTOR * np.abs(angle_goal - rollout_pose[2])

    else:
        cost = np.absolute(delta) #FIXME: Is this the right thing to do?
    current_pose = [0,0,0]
    rollout_pose_angle = self._compute_pose_angle(current_pose, rollout_pose)
    rollout_pose_distance = np.linalg.norm(np.array(current_pose[:-1]) - np.array(rollout_pose[:-1]))

    # if the laser can't see this position, assign max penalty, (e.g. assume that there's something there)
    # if rollout_pose_angle < laser_msg.angle_min or rollout_pose_angle > laser_msg.angle_max:
    #     cost += MAX_PENALTY
    #     return cost

    for i in range(0, self.laser_window):
        angle_index = (int) ((rollout_pose_angle - laser_msg.angle_min) / laser_msg.angle_increment)
        if (angle_index + i) < len(laser_msg.ranges):
            angle_index += i
            laser_ray_dist = laser_msg.ranges[angle_index]

            # we want to ignore NaN and 0 values (as per Patrick's note)
            # these are non-converged sensor values which will result in udefnined(bad) behavior
            if np.isfinite(laser_ray_dist) and laser_ray_dist > 0 and rollout_pose_distance > (laser_ray_dist - np.abs(self.laser_offset)):
                cost += MAX_PENALTY
                return cost

    # if self.prev_angle != None :
    #     prev_angle = self.prev_angle
    #     if np.abs( prev_angle - delta ) > ( 2 * self.delta_incr ):
    #         cost += MAX_PENALTY

    return cost

  '''
  Controls the steering angle in response to the received laser scan. Uses approximately
  self.compute_time amount of time to compute the control
    msg: A LaserScan
  '''
  def wander_cb(self, msg):
    start = rospy.Time.now().to_sec() # Get the time at which this function started

    # A N dimensional matrix that should be populated with the costs of each
    # trajectory up to time t <= T
    drive_msg = AckermannDriveStamped()
    delta_costs = np.zeros(self.deltas.shape[0], dtype=np.float)
    traj_depth = 0

    # Evaluate the cost of each trajectory. Each iteration of the loop should calculate
    # the cost of each trajectory at time t = traj_depth and add those costs to delta_costs
    # as appropriate

    # Pseudo code
    # while(you haven't run out of time AND traj_depth < T):
    #   for each trajectory n:
    #       delta_costs[n] += cost of the t=traj_depth step of trajectory n
    #   traj_depth += 1
    # YOUR CODE HERE

    #Need to call get_next_pose only if current_pose is near plan_pose
    plan_pose = self.get_next_pose()
    p = PoseStamped()
    p.header.frame_id = '/map'
    p.header.stamp = rospy.Time.now()
    p.pose.position.x = plan_pose[0]
    p.pose.position.y = plan_pose[1]
    p.pose.position.z = 0
    p.pose.orientation = utils.angle_to_quaternion(plan_pose[2])
    self.goal_pub.publish(p)
    T = self.rollouts.shape[1]
    while (rospy.Time.now().to_sec() - start < self.compute_time and traj_depth < 2):
        for n in range(self.rollouts.shape[0]):

            delta_costs[n] += self.compute_cost(self.deltas[n], self.rollouts[n][traj_depth], msg, plan_pose)
        traj_depth += 1
    rospy.loginfo(" Cost array , angle chosen")
    rospy.loginfo(delta_costs)
    # Find the delta that has the smallest cost and execute it by publishing
    # YOUR CODE HERE

    #if there is no good path, pick the last direction and then turn
    # if ( min(delta_costs) >= MAX_PENALTY ):
    #     if( self.prev_angle > 0):
    #         min_delta_index = len(self.deltas) - 1
    #     else:
    #         min_delta_index = 0
    # else:
    # delta_costs = - delta_costs
    min_delta_index = np.argmin(delta_costs)
    rospy.loginfo("MIN INDEX MIN ANGLE")
    rospy.loginfo(min_delta_index)    
    min_delta = self.deltas[min_delta_index]
    rospy.loginfo(min_delta)
    self.prev_angle = min_delta #keep track of the angle we chose
    drive_msg.header.stamp = rospy.Time.now()
    drive_msg.header.frame_id = '/map'
    drive_msg.drive.steering_angle = min_delta
    drive_msg.drive.speed = self.speed
    self.cmd_pub.publish(drive_msg)

###CLASS DEFINITION ENDS

'''
Apply the kinematic model to the passed pose and control
  pose: The current state of the robot [x, y, theta]
  control: The controls to be applied [v, delta, dt]
  car_length: The length of the car
Returns the resulting pose of the robot
'''
def kinematic_model_step(pose, control, car_length):
  # Apply the kinematic model
  # Make sure your resulting theta is between 0 and 2*pi
  # Consider the case where delta == 0.0

  #Calculating Beta

  # if(np.isfinite(tan(control[1])) ) == False):
  #   control[1]+= 0.1
  B = math.atan(  0.5*math.tan(control[1]) )
  theta_next = pose[2] + (control[0]/car_length)*(math.sin(2*B))*control[2]

  if(theta_next<0):
    theta_next = 2*math.pi + theta_next

  elif(theta_next > 2*math.pi ):
    theta_next = theta_next - math.pi

 # If delta = 0.0, the car is aligned in the required direction, so angle should remain same.
 # delta=0 > tanB = 0 > sinB =0 > theta next = theta
 # x(dot) = x_next - x_prev => x_next = x_prev + speed*cos(theta_next), similarly for y
  if(B==0):
    x_next = pose[0] + control[0]*math.cos(theta_next)*control[2]
    y_next = pose[1] + control[0]*math.sin(theta_next)*control[2]

  else:
    x_next = pose[0] + car_length/math.sin(2*B) * (math.sin(theta_next) - math.sin(pose[2]))
    y_next = pose[1] - car_length/math.sin(2*B) * (math.cos(theta_next) - math.cos(pose[2]))


  resulting_pose = [x_next, y_next, theta_next]
  return resulting_pose

  # YOUR CODE HERE
  # pass

'''
Repeatedly apply the kinematic model to produce a trajectory for the car
  init_pose: The initial pose of the robot [x,y,theta]
  controls: A Tx3 numpy matrix where each row is of the form [v,delta,dt]
  car_length: The length of the car
Returns a Tx3 matrix where the t-th row corresponds to the robot's pose at time t+1
'''
def generate_rollout(init_pose, controls, car_length):
  # YOUR CODE HERE
  # pass
  N = controls.shape[0]
  rollout_list = []
  pose = init_pose
  for i in xrange(N):
    pose = kinematic_model_step(pose, controls[i], car_length)
    rollout_list.append(pose)

  rollout = np.asarray(rollout_list)
  #rospy.loginfo('%s' % rollout)
  return rollout



'''
Helper function to generate a number of kinematic car rollouts
    speed: The speed at which the car should travel
    min_delta: The minimum allowed steering angle (radians)
    max_delta: The maximum allowed steering angle (radians)
    delta_incr: The difference (in radians) between subsequent possible steering angles
    dt: The amount of time to apply a control for
    T: The number of time steps to rollout for
    car_length: The length of the car
Returns a NxTx3 numpy array that contains N rolled out trajectories, each
containing T poses. For each trajectory, the t-th element represents the [x,y,theta]
pose of the car at time t+1
'''
def generate_mpc_rollouts(speed, min_delta, max_delta, delta_incr, dt, T, car_length):

  deltas = np.arange(min_delta, max_delta, delta_incr)
  N = deltas.shape[0]

  init_pose = np.array([0.0,0.0,0.0], dtype=np.float)

  rollouts = np.zeros((N,T,3), dtype=np.float)
  for i in xrange(N):
    controls = np.zeros((T,3), dtype=np.float)
    controls[:,0] = speed
    controls[:,1] = deltas[i]
    controls[:,2] = dt
    rollouts[i,:,:] = generate_rollout(init_pose, controls, car_length)

  return rollouts, deltas



def main():

  rospy.init_node('mpc_controller', anonymous=True)

  # Load these parameters from launch file
  # We provide suggested starting values of params, but you should
  # tune them to get the best performance for your system
  # Look at constructor of MPCControllerer class for description of each var
  # 'Default' values are ones that probably don't need to be changed (but you could for fun)
  # 'Starting' values are ones you should consider tuning for your system
  # YOUR CODE HERE
  speed = rospy.get_param('~speed')# Default val: 1.0
  speed = speed
  min_delta = rospy.get_param('~min_delta')# Default val: -0.34
  max_delta = rospy.get_param('~max_delta')# Default val: 0.341
  delta_incr = rospy.get_param('~delta_incr')# Starting val: 0.34/3 (consider changing the denominator)
  laser_window = rospy.get_param('~laser_window')# to account for car width we search a window
  #in the laser scan for possible collision objects Default val = 1
  delta_incr = delta_incr / 3
  dt = rospy.get_param('~dt')# Default val: 0.01
  T = rospy.get_param('~T')# Starting val: 300
  compute_time = rospy.get_param('~compute_time') # Default val: 0.09
  laser_offset = rospy.get_param('~laser_offset')# Starting val: 1.0
  lookahead_distance = rospy.get_param('~lookahead_dist', 2)# Default Val: 2m
  # DO NOT ADD THIS TO YOUR LAUNCH FILE, car_length is already provided by teleop.launch
  car_length = rospy.get_param("car_kinematics/car_length", 0.33)
  bag_path = rospy.get_param('~bag_path')
  # Generate the rollouts
  rollouts, deltas = generate_mpc_rollouts(speed, min_delta, max_delta, delta_incr, dt, T, car_length)

  
  bag = rosbag.Bag(bag_path)
  plan = deque()

  # for topic, msg, t in bag.read_messages(topics=BAG_TOPIC):
  #   msgList.append(msg)

  for _, plan_msg, _ in bag.read_messages(topics=PLAN_PUB_TOPIC):
    for i, msg in enumerate(plan_msg.poses):
      theta = utils.quaternion_to_angle(msg.orientation)
      pose = np.array([msg.position.x,msg.position.y,theta])
      plan.append(pose)
  
  rospy.loginfo("BAG RECEIVED AND STORED")
  # Create the MPCControllerer
  lw = MPCController(rollouts, deltas, speed, compute_time, laser_offset, laser_window, delta_incr, lookahead_distance, plan)
  
  rospy.sleep(1)
  lw.publish_full_car_plan(plan_msg)

  # Keep the node alive
  rospy.spin()


if __name__ == '__main__':
  main()
