#!/usr/bin/env python
from __future__ import division

from threading import Lock

import sys
import time

import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable

import rosbag
import rospy
import utils as Utils
from utils import describe
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import (PointStamped, PoseArray, PoseStamped,
                               PoseWithCovarianceStamped)
from nav_msgs.msg import Path
from nav_msgs.srv import GetMap
from vesc_msgs.msg import VescStateStamped

class MPPIController:

  def __init__(self, T, K, sigma=(0.5 * torch.eye(2)), _lambda=0.5):
    self.dtype = torch.float
    if torch.cuda.is_available():
      print('Running PyTorch on GPU')
      self.device = torch.device("cuda")
    else:
      print('Running PyTorch on CPU')
      self.device = torch.device("cpu")

    self.SPEED_TO_ERPM_OFFSET = float(rospy.get_param("/vesc/speed_to_erpm_offset", 0.0))
    self.SPEED_TO_ERPM_GAIN   = float(rospy.get_param("/vesc/speed_to_erpm_gain", 4614.0))
    self.STEERING_TO_SERVO_OFFSET = float(rospy.get_param("/vesc/steering_angle_to_servo_offset", 0.5304))
    self.STEERING_TO_SERVO_GAIN   = float(rospy.get_param("/vesc/steering_angle_to_servo_gain", -1.2135))
    self.CAR_LENGTH = 0.33 
    self.OOB_COST = 100000 # Cost associated with an out-of-bounds pose
    self.MAX_SPEED = 5.0 # TODO NEED TO FIGURE OUT ACTUAL FIGURE FOR THIS
    self.DIST_COST_GAIN = 1000.0

    self.last_pose = None
    # MPPI params
    self.T = T # Length of rollout horizon
    self.K = K # Number of sample rollouts
    self.sigma = torch.tensor([[0.001, 0.0],[0.0, 0.001]], dtype=self.dtype, device=self.device)
    # self.sigma = 0.05 * torch.eye(2)  # NOTE: DEBUG
    self._lambda = torch.tensor(_lambda, dtype=self.dtype, device=self.device)
    self.dt = None

    self.goal = None # Lets keep track of the goal pose (world frame) over time
    self.lasttime = None

    self.device = None
    self.state_lock = Lock()

    self.missed_msgs = 0
    self.max_angle = -10000.0
    self.min_angle = 10000.0

    # PyTorch / GPU data configuration
    # TODO
    # you should pre-allocate GPU memory when you can, and re-use it when
    # possible for arrays storing your controls or calculated MPPI costs, etc

    # Load model
    # model_name = rospy.get_param("~nn_model", "myneuralnetisbestneuralnet.pt")
    # self.model = torch.load(model_name)
    # self.model.cuda()  # Tell Torch to run model on GPU

    # print("Loading:", model_name)
    # print("Model:\n",self.model)
    print("Torch Datatype:", self.dtype)

    self.rollouts = torch.empty(self.K, self.T + 1, 3, dtype=self.dtype, device=self.device)  # KxTx[x, y, theta]_map
    self.controls = torch.zeros(self.K, self.T, 2, dtype=self.dtype, device=self.device)  # Tx[delta, speed] array of controls
    self.nominal_control = torch.zeros(self.T, 2, dtype=self.dtype, device=self.device)
    self.noise_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), self.sigma)
    self.noise = torch.zeros(self.K, self.T, 2, dtype=self.dtype, device=self.device)
    self.cost = None
    self.nominal_rollout = torch.zeros(self.T, 3, dtype=self.dtype, device=self.device)
    # control outputs
    self.msgid = 0

    # visualization paramters
    self.num_viz_paths = 40
    if self.K < self.num_viz_paths:
        self.num_viz_paths = self.K

    # We will publish control messages and a way to visualize a subset of our
    # rollouts, much like the particle filter
    self.ctrl_pub = rospy.Publisher('/vesc/high_level/ackermann_cmd_mux/input/nav_0',
            AckermannDriveStamped, queue_size=2)
    self.path_pub = rospy.Publisher("/mppi/paths", Path, queue_size = self.num_viz_paths)
    self.nom_path_pub = rospy.Publisher("/mppi/nominal", Path, queue_size = 1)

    # Use the 'static_map' service (launched by MapServer.launch) to get the map
    map_service_name = rospy.get_param("~static_map", "static_map")
    print("Getting map from service: ", map_service_name)
    rospy.wait_for_service(map_service_name)
    map_msg = rospy.ServiceProxy(map_service_name, GetMap)().map # The map, will get passed to init of sensor model
    self.map_info = map_msg.info # Save info about map for later use    
    print("Map Information:\n",self.map_info)

    # Create numpy array representing map for later use
    self.map_height = map_msg.info.height
    self.map_width = map_msg.info.width
    array_255 = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))
    self.permissible_region = np.zeros_like(array_255, dtype=bool)
    self.permissible_region[array_255==0] = 1 # Numpy array of dimension (map_msg.info.height, map_msg.info.width),
                                              # With values 0: not permissible, 1: permissible
    x_perm, y_perm = np.where(self.permissible_region == 1)
    self.permit_coords = np.vstack((x_perm, y_perm)).T                  
                                              
    print("Making callbacks")
    self.goal_sub = rospy.Subscriber("/move_base_simple/goal",
            PoseStamped, self.clicked_goal_cb, queue_size=1)
    self.pose_sub  = rospy.Subscriber("/pf/viz/inferred_pose",
            PoseStamped, self.mppi_cb, queue_size=1)
    print("Done Initializing")

  # TODO
    # You may want to debug your bounds checking code here, by clicking on a part
    # of the map and convincing yourself that you are correctly mapping the
    # click, and thus the goal pose, to accessible places in the map
  def clicked_goal_cb(self, msg):
    self.goal = np.array([msg.pose.position.x,
                          msg.pose.position.y,
                          Utils.quaternion_to_angle(msg.pose.orientation)])
    print("Current Pose: ", self.last_pose)
    print("SETTING Goal: ", self.goal)

  def compute_costs(self):
    pose_cost = torch.zeros(self.K, dtype=self.dtype, device=self.device)
    bounds_cost = torch.zeros(self.K, dtype=self.dtype, device=self.device)
    ctrl_cost = torch.zeros(self.K, dtype=self.dtype, device=self.device)

    ### COMMENTED OUT FOR TESTING ###
    pose_cost = torch.sum((torch.abs(self.controls[:,:,1]) - self.MAX_SPEED)**2, dim=1)  # TODO: this will be much better if we can output speed as a predicted state parameter from MPC
    
    ctrl_cost = self._lambda * torch.sum(torch.matmul(torch.abs(self.nominal_control), torch.inverse(self.sigma)) * torch.abs(self.noise), dim=(1, 2))  # This is verified to give same result as the looped code shown in the spec

    total_in_bounds = 0
    # TODO: Probably need a vectorized way of doing this
    for k in xrange(self.K):
      # map_poses = self.rollouts[k, :, :].clone().numpy()
      map_poses = self.rollouts[k, :, :].numpy()
      map_poses = map_poses.copy()
      Utils.world_to_map(map_poses, self.map_info)  #NOTE: Clone required here since world_to_map operates in place
      map_poses = np.round(map_poses).astype(int)
      try:
        first_oob = np.argmin(self.permissible_region[map_poses[:,1], map_poses[:,0]])
        # print("first_oob:")
        # print(first_oob)
        # n_in_bounds = np.sum(self.permissible_region[map_poses[:,1], map_poses[:,0]])
      except IndexError:
        bounds_cost[k] = self.OOB_COST
        continue
      # total_in_bounds += n_in_bounds  # TODO: Can potentially use this later to live-alter the value of self.sigma to prevent more than a certain fraction of total rolled out poses from going out of bounds
      # if n_in_bounds < map_poses.shape[0]:
      #   bounds_cost[k] = self.OOB_COST
      if first_oob > 0:
        bounds_cost[k] = self.OOB_COST * (self.T - first_oob)
    cart_off = self.rollouts[:, -1, :] - torch.tensor(self.goal, dtype=self.dtype, device=self.device)  # Cartesian offset between [X, Y, theta]_rollout[k] and [X, Y, theta]_goal
    dist_cost = torch.tensor(self.DIST_COST_GAIN) * torch.sqrt(cart_off[:, 0]**2 + cart_off[:, 1]**2)  # Calculates magnitude of distance from goal

    assert np.all(np.equal([pose_cost.shape, ctrl_cost.shape, bounds_cost.shape], self.K)), "Shape of cost components != (self.K)\n pose_cost.shape = {}\n ctrl_cost.shape = {}\n bounds_cost.shape = {}\n".format(pose_cost.shape, ctrl_cost.shape, bounds_cost.shape)
    # describe([pose_cost, ctrl_cost, bounds_cost, dist_cost])
    # print("Costs:")
    for cost, name in zip([pose_cost, ctrl_cost, bounds_cost, dist_cost], ["pose_cost", "ctrl_cost", "bounds_cost", "dist_cost"]):
      if np.any(np.isnan(cost)):
        assert False, "NaNs in output: {}".format(name)
      
      if (name == "bounds_cost") and (torch.max(cost) > 0):
        cost = cost / torch.min(cost[torch.nonzero(cost)])
        assert torch.min(cost[torch.nonzero(cost)]) >= 1.0, "bounds_cost is < 1: bounds_cost = {}".format(torch.min(cost))
        # print(name)
        # print(cost)
        continue

      cost = cost - torch.min(cost)
      cost_max = torch.max(cost)
      if cost_max > 0:
        cost = cost / cost_max
      # print(name)
      # print(cost)
    self.cost = (pose_cost) + (ctrl_cost) + (bounds_cost * 1000) + (2*dist_cost)
# JPH mm code
  # def mm_step(self, states, controls):
  #   # self.state_lock.acquire()
  #   # if self.last_servo_cmd is None:
  #   #   self.state_lock.release()
  #   #   return

  #   # if self.last_vesc_stamp is None:
  #   #   self.last_vesc_stamp = msg.header.stamp
  #   #   self.state_lock.release()
  #   #   return

  #   # #Convert the current speed and delta
  #   # curr_speed = (msg.state.speed - self.SPEED_TO_ERPM_OFFSET)/self.SPEED_TO_ERPM_GAIN
  #   # curr_delta = (self.last_servo_cmd - self.STEERING_TO_SERVO_OFFSET)/self.STEERING_TO_SERVO_GAIN

  #   deltas = controls[:, 0]
  #   speeds = controls[:, 1]
  #   states_next = torch.zeros_like(states)

  #   #Calculate the Kinematic Model additions
  #   beta = torch.atan(torch.tan(deltas) * 0.5 )
  #   KM_theta = speeds/self.CAR_LENGTH*torch.sin(2*beta)*self.dt
  #   # KM_theta = ((KM_theta + np.pi) % (2*np.pi)) - np.pi # FIXME: Maybe this is wrong?

  #   if(theta_next<0):
  #     theta_next = 2*math.pi + theta_next

  #   elif(theta_next > 2*math.pi ):
  #     theta_next = theta_next - math.pi

  #   # assert torch.any((KM_theta <= np.pi) | (KM_theta >= -np.pi)), "KM_theta = {} (not within the range [-pi, pi])".format(KM_theta)
  #   KM_X = self.CAR_LENGTH/torch.sin(2*beta)*(torch.sin(states[:,2] + KM_theta)-torch.sin(states[:,2]))
  #   KM_Y = self.CAR_LENGTH/torch.sin(2*beta)*(-torch.cos(states[:,2] + KM_theta)+torch.cos(states[:,2]))

  #   #Propogate the model forward and add noise
  #   states_next[:,0] = states[:,0] + KM_X
  #   states_next[:,1] = states[:,1] + KM_Y
  #   states_next[:,2] = ((states[:,2] + KM_theta + np.pi) % (2*np.pi)) - np.pi
  #   assert torch.any((states_next[:,2] <= np.pi) | (states_next[:,2] >= -np.pi)), "states_next[:,2] = {} (not within the range [-pi, pi])".format(KM_theta)

  #   return states_next
  #   # self.last_vesc_stamp = msg.header.stamp
  #   # self.state_lock.release()

''' This is currently ported from TA's lab2 solutions for checking '''

  def mm_step(self, states, controls):
  
    # Update the proposal distribution by applying the control to each particle
      
    delta, v, dt = control
    # v_mag = torch.abs(v)
    # delta_mag = torch.abs(delta)

    # # Sample control noise and add to nominal control
    # v += np.random.normal(loc=0.0, scale=KM_V_NOISE, size=proposal_dist.shape[0])
    # delta += np.random.normal(loc=0.0, scale=KM_DELTA_NOISE, size=proposal_dist.shape[0])    

    # Compute change in pose based on controls
    if delta_mag < 1e-2:
        dx = v * torch.cos(proposal_dist[:, 2]) * dt
        dy = v * torch.sin(proposal_dist[:, 2]) * dt
        dtheta = 0
    else:
        beta = torch.atan(0.5 * torch.tan(delta))
        sin2beta = torch.sin(2 * beta)
        dtheta = ((v / self.CAR_LENGTH) * sin2beta) * dt
        dx = (self.CAR_LENGTH/sin2beta)*(np.sin(proposal_dist[:,2]+dtheta)-np.sin(proposal_dist[:,2]))        
        dy = (self.CAR_LENGTH/sin2beta)*(-1*np.cos(proposal_dist[:,2]+dtheta)+np.cos(proposal_dist[:,2]))

    # Propagate particles forward, and add sampled model noise
    # proposal_dist[:, 0] += dx + np.random.normal(loc=0.0, scale=KM_X_FIX_NOISE+KM_X_SCALE_NOISE*v_mag, size=proposal_dist.shape[0])
    proposal_dist[:, 0] += dx
    # proposal_dist[:, 1] += dy + np.random.normal(loc=0.0, scale=KM_Y_FIX_NOISE+KM_Y_SCALE_NOISE*v_mag, size=proposal_dist.shape[0])
    proposal_dist[:, 1] += dy
    # proposal_dist[:, 2] += dtheta + np.random.normal(loc=0.0, scale=KM_THETA_FIX_NOISE, size=proposal_dist.shape[0])
    proposal_dist[:, 2] += dtheta + np.random.normal(loc=0.0, scale=KM_THETA_FIX_NOISE, size=proposal_dist.shape[0])
    # Limit particle totation to be between -pi and pi
    proposal_dist[proposal_dist[:,2] < -1*np.pi,2] += 2*np.pi  
    proposal_dist[proposal_dist[:,2] > np.pi,2] -= 2*np.pi  


  def do_rollouts(self):
    print("Making Rollouts...")
    if not isinstance(self.last_pose, np.ndarray):
      print("self.last_pose not yet defined!")
      return
    else:
      self.rollouts[:, 0, :] = torch.tensor(self.last_pose, dtype=self.dtype, device=self.device)

    for t in xrange(1, self.T + 1):
      self.rollouts[:, t, :] = self.mm_step(self.rollouts[:, t - 1, :], self.controls[:, t - 1, :])
    print("Done")

  def mppi(self, init_pose, init_input):
    t0 = time.time()
    # NOTE:
      # Network input can be:
      #   0    1       2          3           4        5      6   7
      # xdot, ydot, thetadot, sin(theta), cos(theta), vel, delta, dt

      # MPPI should
      # generate noise according to sigma
      # combine that noise with your central control sequence
      # Perform rollouts with those controls from your current pose
      # Calculate costs for each of K trajectories
      # Perform the MPPI weighting on your calculated costs
      # Scale the added noise by the weighting and add to your control sequence
      # Apply the first control values, and shift your control trajectory
      
      # Notes:
      # MPPI can be assisted by carefully choosing lambda, and sigma
      # It is advisable to clamp the control values to be within the feasible range
      # of controls sent to the Vesc
      # Your code should account for theta being between -pi and pi. This is
      # important.
      # The more code that uses pytorch's cuda abilities, the better; every line in
      # python will slow down the control calculations. You should be able to keep a
      # reasonable amount of calculations done (T = 40, K = 2000) within the 100ms
      # between inferred-poses from the particle filter.
    self.noise = self.noise_dist.rsample((self.K, self.T))  # Generates a self.K x self.T x2 matrix of noise sampled from self.noise_dist
    # self.noise[0, :, :] = torch.zeros(self.T, 2) # Make sure the current nominal trajectory is considered as one of the possible rollouts
    self.controls = self.nominal_control.repeat(self.K, 1, 1) + self.noise
    assert self.nominal_control.repeat(self.K,1,1).shape == (self.K, self.T, 2)
    self.cost = torch.zeros(K, dtype=self.dtype, device=self.device)
    self.do_rollouts()  # Perform rollouts from current state, update self.rollouts in place

    self.compute_costs()

    # Perform the MPPI weighting on your calculated costs
    beta = torch.min(self.cost)
    assert beta >= 0, "Minimum cost is < 0. beta = {}".format(beta)

    self.cost = self.cost - beta
    self.weights = torch.zeros_like(self.cost)
    self.weights = torch.exp((-1.0/self._lambda)*(self.cost))
    self.weights = self.weights / torch.sum(self.weights)

    assert torch.abs(torch.sum(self.weights) - 1) < 1e-5, "self.weights sums to {} (not == 1)".format(torch.sum(self.weights))
    # Generate the new nominal control
    for t in xrange(self.T):
      self.nominal_control[t] = self.nominal_control[t] + torch.sum(self.noise[:, t, :] * self.weights.view(self.K, 1).repeat(1,2), dim=0)
    
    for t in xrange(self.T):
      self.nominal_rollout[t] = torch.sum(self.rollouts[:, t, :] * self.weights.view(self.K, 1).repeat(1,3), dim=0)

    run_ctrl = self.nominal_control[0].clone()
    self.nominal_control = torch.cat((self.nominal_control[1:, :], self.nominal_control[-1, :].view(1,2)))  # Rolls the array forward in time, then duplicates the last row to maintain size

    print("MPPI: %4.5f ms" % ((time.time()-t0)*1000.0))

    return run_ctrl

  def mppi_cb(self, msg):
    self.state_lock.acquire()
    print("mppi_callback")
    if self.last_pose is None:
      print("No self.last_pose. Initializing with goal at present car location.")
      self.last_pose = np.array([msg.pose.position.x,
                                 msg.pose.position.y,
                                 Utils.quaternion_to_angle(msg.pose.orientation)])
      # Default: initial goal to be where the car is when MPPI node is
      # initialized
      self.goal = self.last_pose
      self.lasttime = msg.header.stamp.to_sec()
      self.state_lock.release()
      return

    theta = Utils.quaternion_to_angle(msg.pose.orientation)
    self.min_angle = min(theta, self.min_angle)
    self.max_angle = max(theta, self.max_angle)
    curr_pose = np.array([msg.pose.position.x,
                          msg.pose.position.y,
                          theta])

    pose_dot = curr_pose - self.last_pose # get state
    self.last_pose = curr_pose

    timenow = msg.header.stamp.to_sec()
    self.dt = timenow - self.lasttime
    self.lasttime = timenow
    nn_input = np.array([pose_dot[0], pose_dot[1], pose_dot[2],
                         np.sin(theta),
                         np.cos(theta), 0.0, 0.0, self.dt])

    run_ctrl = self.mppi(curr_pose, nn_input)

    self.send_controls(run_ctrl[0], run_ctrl[1])
    self.state_lock.release()
    self.visualize()
    print(self.min_angle)
    print(self.max_angle)

  def send_controls(self, steer, speed):
    print("Speed:", speed, "Steering:", steer)
    ctrlmsg = AckermannDriveStamped()
    ctrlmsg.header = Utils.make_header('map')
    ctrlmsg.header.seq = self.msgid
    ctrlmsg.drive.steering_angle = steer 
    ctrlmsg.drive.speed = speed
    self.ctrl_pub.publish(ctrlmsg)
    self.msgid += 1

  # Publish some paths to RVIZ to visualize rollouts
  def visualize(self):
    print("Running viz")
    if self.path_pub.get_num_connections() > 0:
      frame_id = 'map'
      pa = Path()
      pa.header = Utils.make_header(frame_id)
      for i in range(0, self.num_viz_paths):
        pa.poses = [Utils.particle_to_posestamped(pose, frame_id) for pose in self.rollouts[i,:,:]]
        self.path_pub.publish(pa)
    if self.nom_path_pub.get_num_connections() > 0:
      frame_id = 'map'
      pa = Path()
      pa.header = Utils.make_header(frame_id)
      pa.poses = [Utils.particle_to_posestamped(pose, frame_id) for pose in self.nominal_rollout]
      self.nom_path_pub.publish(pa)

def test_MPPI(mp, N, goal=np.array([0.,0.,0.])):
  init_input = np.array([0.,0.,0.,0.,1.,0.,0.,0.])
  pose = np.array([0.,0.,0.])
  mp.goal = goal
  print("Start:", pose)
  mp.ctrl.zero_()
  last_pose = np.array([0.,0.,0.])
  for i in range(0,N):
    # ROLLOUT your MPPI function to go from a known location to a specified
    # goal pose. Convince yourself that it works.

    print("Now:", pose)
  print("End:", pose)
     
if __name__ == '__main__':
  rospy.init_node("mppi", anonymous=True) # Initialize the node

  T = 20
  K = 512
  sigma = 0.05 # These values will need to be tuned
  _lambda = 1.0

  mppi = MPPIController(T, K, sigma, _lambda)
  while not rospy.is_shutdown(): # Keep going until we kill it
  # Callbacks are running in separate threads
    rospy.sleep(1)
