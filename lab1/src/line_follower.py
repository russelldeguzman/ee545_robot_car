#!/usr/bin/env python

import collections
import sys
import csv

import rospy
import numpy as np
from geometry_msgs.msg import PoseArray, PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from collections import deque

import utils

# The topic to publish control commands to
PUB_TOPIC = "/vesc/high_level/ackermann_cmd_mux/input/nav_0"
# SUB_TOPIC = '/sim_car_pose/pose' # The topic that provides the simulated car pose
MAP_TOPIC = "static_map"  # The service topic that will provide the map
CSV_PATH = "/home/car-user/racecar_ws/src/ee545_robot_car/lab1/src/"
CSV_FILE = "pid-error.csv"

"""
Follows a given plan using constant velocity and PID control of the steering angle
"""


class LineFollower:

    """
  Initializes the line follower
    plan: A list of length T that represents the path that the robot should follow
          Each element of the list is a 3-element numpy array of the form [x,y,theta]
    pose_topic: The topic that provides the current pose of the robot as a PoseStamped msg
    plan_lookahead: If the robot is currently closest to the i-th pose in the plan,
                    then it should navigate towards the (i+plan_lookahead)-th pose in the plan
    translation_weight: How much the error in translation should be weighted in relation
                        to the error in rotation
    rotation_weight: How much the error in rotation should be weighted in relation
                     to the error in translation
    kp: The proportional PID parameter
    ki: The integral PID parameter
    kd: The derivative PID parameter
    error_buff_length: The length of the buffer that is storing past error values
    speed: The speed at which the robot should travel
  """

    def __init__(
        self,
        plan,
        pose_topic,
        plan_lookahead,
        translation_weight,
        rotation_weight,
        kp,
        ki,
        kd,
        error_buff_length,
        speed,
    ):
        # Store the passed parameters
        self.plan = plan
        self.plan_lookahead = plan_lookahead
        # Normalize translation and rotation weights
        self.translation_weight = translation_weight / (
            translation_weight + rotation_weight
        )
        self.rotation_weight = rotation_weight / (translation_weight + rotation_weight)
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error_log = deque()

        # The error buff stores the error_buff_length most recent errors and the
        # times at which they were received. That is, each element is of the form
        # [time_stamp (seconds), error]. For more info about the data struct itself, visit
        # https://docs.python.org/2/library/collections.html#collections.deque
        self.error_buff = deque(maxlen=error_buff_length)
        self.speed = speed

        # Publisher
        self.cmd_pub = rospy.Publisher(PUB_TOPIC, AckermannDriveStamped, queue_size=10)

        # Create a subscriber to pose_topic, with callback 'self.pose_cb'
        self.pose_sub = rospy.Subscriber(
            pose_topic, PoseStamped, self.pose_cb, queue_size=1
        )

    """
  Computes the error based on the current pose of the car
    cur_pose: The current pose of the car, represented as a numpy array [x,y,theta]
  Returns: (False, 0.0) if the end of the plan has been reached. Otherwise, returns
           (True, E) - where E is the computed error
  """

    def compute_error(self, cur_pose):

        # Find the first element of the plan that is in front of the robot, and remove
        # any elements that are behind the robot. To do this:
        # Loop over the plan (starting at the beginning) For each configuration in the plan
        # If the configuration is behind the robot, remove it from the plan
        #   Will want to perform a coordinate transformation to determine if
        #   the configuration is in front or behind the robot
        # If the configuration is in front of the robot, break out of the loop

        # Current position of car
        cur_pose_x = cur_pose[0]
        cur_pose_y = cur_pose[1]
        cur_pose_th = cur_pose[2]

        # This code starts at the beginning of self.plan and marches forward until
        # it encounters a pose that is in front of the car
        # To do this, simply we will make use of the dot product instead of coordinate transformations.
        # First calculate vector between carPose and planPose, vectorC2P
        # If dot product between carPose unit vector and vectorC2P is positive, then the point is in front
        # If dot product between carPose unit vector and vectorC2P is negative, then the point is behind
        while True:
            try:
                [target_x, target_y, target_th] = self.plan.popleft()

                # Vector between ith pose and the current car pse
                vectorC2P = [target_x - cur_pose_x, target_y - cur_pose_y]
                unit_vectorC2P = vectorC2P / np.linalg.norm(vectorC2P)

                # Unit vector in the direction of the car pose
                carPoseVector = [np.cos(cur_pose_th), np.sin(cur_pose_th)]

                dotProduct = np.dot(vectorC2P, carPoseVector)

                # If dot product is positive value, then the target node is in front
                # Break out of the while loop if dot product > -0.2 (i.e. target node is in front)
                if (
                    dotProduct > -0.2
                ):  # Counting a point as "in front" when the dot product is slightly negative helps make sure the robot doesn't stop slightly short of the last pose in the array
                    self.plan.appendleft(
                        np.array([target_x, target_y, target_th])
                    )  # If it turns out the point was in front, put it back in the deque in case it's still in front at the next timestep
                    rospy.loginfo("Quiting While Loop - Point forward of car reached")
                    # Prints the length of the plan.
                    rospy.loginfo("self.plan length")
                    rospy.loginfo(len(self.plan))
                    break
            except IndexError:
                # self.plan is empty, so return (False, 0.0)
                rospy.loginfo("THIS SHOULD BE THE END OF MSGs")
                return (False, 0.0)

        # At this point, we have removed configurations from the plan that are behind
        # the robot. Therefore, element 0 is the first configuration in the plan that is in
        # front of the robot. To allow the robot to have some amount of 'look ahead',
        # we choose to have the robot head towards the configuration at index 0 + self.plan_lookahead
        # We call this index the goal_index

        goal_idx = int(round(min(0 + self.plan_lookahead, len(self.plan) - 1)))
        [goal_idx_x, goal_idx_y, goal_idx_th] = self.plan[goal_idx]

        ### Compute the translation error between the robot and the configuration at goal_idx in the plan

        # Unit vector in the same orientation as the goal pose
        goal_unit_vector = [np.cos(goal_idx_th), np.sin(goal_idx_th)]

        # Vector between the current pose and the goal pose
        distance_vector = [cur_pose_x - goal_idx_x, cur_pose_y - goal_idx_y]

        # Use the cross product to calculate the distance error to capture positive or negative wrt to goal_unit_vector
        translation_error = np.cross(distance_vector, goal_unit_vector)
        rotation_error = (
            (goal_idx_th - cur_pose_th + np.pi) % (2 * np.pi)
        ) - np.pi  # Needed in order to constrain this difference to the interval [-pi, pi] as opposed to [-2*pi, 2*pi]

        rospy.loginfo("goal_th = ")
        rospy.loginfo(goal_idx_th)
        rospy.loginfo("pose_th = ")
        rospy.loginfo(cur_pose_th)
        rospy.loginfo("translation error ")
        rospy.loginfo(translation_error)
        rospy.loginfo("rotation error")
        rospy.loginfo(rotation_error)

        # Compute the total error
        # Translation error was computed above
        # Rotation error is the difference in yaw between the robot and goal configuration
        #   Be carefult about the sign of the rotation error

        error = (
            self.translation_weight * translation_error
            + self.rotation_weight * rotation_error
        )
        self.error_log.append(error)
        return (True, error)

    """
  Uses a PID control policy to generate a steering angle from the passed error
    error: The current error
  Returns: The steering angle that should be executed
  """

    def compute_steering_angle(self, error):
        now = rospy.Time.now().to_sec()  # Get the current time

        # Compute the derivative error using the passed error, the current time,
        # the most recent error stored in self.error_buff, and the most recent time
        # stored in self.error_buff
        deriv_error = 0
        if len(self.error_buff) > 0:

            [last_time, last_error] = self.error_buff[-1]
            deriv_error = (error - last_error) / (now - last_time)

        # Compute the integral error by applying rectangular integration to the elements
        integ_error = 0
        next_time = now
        next_error = error
        for elem in reversed(self.error_buff):

            error_element = (elem[1] + next_error) / 2 * (next_time - elem[0])
            integ_error = integ_error + error_element

            # Store the next time
            next_time = elem[0]

        # Add the current error to the buffer
        self.error_buff.append((now, error))

        # Compute the steering angle as the sum of the pid errors
        steering_angle = self.kp * error + self.ki * integ_error + self.kd * deriv_error
        rospy.loginfo("Steering Angle")
        rospy.loginfo(steering_angle)
        steering_angle = np.sign(steering_angle) * min(abs(steering_angle), np.pi / 2)
        return steering_angle

    """
  Callback for the current pose of the car
    msg: A PoseStamped representing the current pose of the car
    This is the exact callback that we used in our solution, but feel free to change it
  """

    def pose_cb(self, msg):
        cur_pose = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                utils.quaternion_to_angle(msg.pose.orientation),
            ]
        )

        success, error = self.compute_error(cur_pose)

        if not success:
            # We have reached our goal (or at least popped all points in the plan)
            self.pose_sub.unregister()  # Kill the subscriber
            rospy.loginfo("self.pose_sub has been shutdown")
            self.speed = 0.0  # Set speed to zero so car stops
            with open(CSV_PATH + CSV_FILE, mode="w") as err_file:
                err_writer = csv.writer(err_file, delimiter=" ")
                for step, err in enumerate(self.error_log):
                    try:
                        err_writer.writerow((step, err))
                    except IndexError:
                        break

        rospy.loginfo("error = ")
        rospy.loginfo(error)

        delta = self.compute_steering_angle(error)

        rospy.loginfo("Steer Angle")

        rospy.loginfo(delta)
        print("\n")

        # Setup the control message
        ads = AckermannDriveStamped()
        ads.header.frame_id = "/map"
        ads.header.stamp = rospy.Time.now()
        ads.drive.steering_angle = delta
        ads.drive.speed = self.speed

        # Send the control message
        self.cmd_pub.publish(ads)


def main():

    rospy.init_node("line_follower", anonymous=True)  # Initialize the node

    # Load these parameters from launch file
    # We provide suggested starting values of params, but you should
    # tune them to get the best performance for your system
    # Look at constructor of LineFollower class for description of each var
    # 'Default' values are ones that probably don't need to be changed (but you could for fun)
    # 'Starting' values are ones you should consider tuning for your system
    # YOUR CODE HERE

    # Default values
    plan_topic = "/planner_node/car_plan"
    pose_topic = "/sim_car_pose/pose"
    plan_lookahead = 5
    translation_weight = 1.0
    rotation_weight = 0.0
    kp = 1.0
    ki = 0.0
    kd = 0.0
    error_buff_length = 10
    speed = 1.0

    # Values from Launch file
    plan_topic = rospy.get_param("~plan_topic", None)
    pose_topic = rospy.get_param("~pose_topic", None)
    plan_lookahead = rospy.get_param("~plan_lookahead", None)
    translation_weight = rospy.get_param("~translation_weight", None)
    rotation_weight = rospy.get_param("~rotation_weight", None)
    kp = rospy.get_param("~kp", None)
    ki = rospy.get_param("~ki", None)
    kd = rospy.get_param("~kd", None)
    error_buff_length = rospy.get_param("~error_buff_length", None)
    speed = rospy.get_param("~speed", None)

    # Waits for ENTER key press
    raw_input("Press Enter to when plan available...")

    # Use rospy.wait_for_message to get the plan msg
    # Convert the plan msg to a list of 3-element numpy arrays
    #     Each array is of the form [x,y,theta]
    # Create a LineFollower object
    plan_msg = rospy.wait_for_message(plan_topic, PoseArray)
    plan = deque()
    for msg in plan_msg.poses:
        x = msg.position.x
        y = msg.position.y
        theta = utils.quaternion_to_angle(msg.orientation)
        pose = np.array([x, y, theta])
        plan.append(pose)

    # Create a clone follower
    lf = LineFollower(
        plan,
        pose_topic,
        plan_lookahead,
        translation_weight,
        rotation_weight,
        kp,
        ki,
        kd,
        error_buff_length,
        speed,
    )

    rospy.spin()  # Prevents node from shutting down


if __name__ == "__main__":
    main()

