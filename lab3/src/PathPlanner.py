#!/usr/bin/env python

import collections
import sys

import rospy
import numpy as np
from geometry_msgs.msg import PoseArray, PoseStamped, PoseWithCovarianceStamped
import utils

# The topic to publish the full car plan to 
PLAN_PUB_TOPIC = "/planner_node/full_car_plan"

# Topics to set end points of pose arrays 
INITIAL_PUB_TOPIC = "/initialpose"
GOAL_PUB_TOPIC = "/move_base_simple/goal"

MAP_TOPIC = "static_map"  # The service topic that will provide the map

class PathPlanner:

    def __init__(self, plan, plan_targets):
        # Store the passed parameters
        self.plan = plan
        self.plan_targets = plan_targets

        # Publisher
        self.plan_pub = rospy.Publisher(PLAN_PUB_TOPIC, PoseArray, queue_size=10)
        self.initial_pub = rospy.Publisher(INITIAL_PUB_TOPIC, PoseWithCovarianceStamped, queue_size=1)
        self.goal_pub = rospy.Publisher(GOAL_PUB_TOPIC, PoseStamped, queue_size=1)

    def publish_next_positions(self, initialPose, goalPose, time_stamp):        

        rospy.loginfo("Initial Pose")
        rospy.loginfo(initialPose)
        rospy.loginfo("Goal Pose")
        rospy.loginfo(goalPose)
        InitialPoseMsg = PoseWithCovarianceStamped()
        quaternion = utils.angle_to_quaternion(initialPose[2])
        InitialPoseMsg.header.stamp = time_stamp
        InitialPoseMsg.header.frame_id = "/map"
        InitialPoseMsg.pose.pose.position.x = initialPose[0]
        InitialPoseMsg.pose.pose.position.y = initialPose[1]
        InitialPoseMsg.pose.pose.orientation = quaternion

        GoalPoseMsg = PoseStamped()
        quaternion = utils.angle_to_quaternion(goalPose[2])
        GoalPoseMsg.header.stamp = time_stamp 
        GoalPoseMsg.header.frame_id = "/map"
        GoalPoseMsg.pose.position.x = goalPose[0]
        GoalPoseMsg.pose.position.y = goalPose[1]
        GoalPoseMsg.pose.orientation = quaternion

        # rospy.loginfo(InitialPoseMsg)
        self.initial_pub.publish(InitialPoseMsg) 
        self.goal_pub.publish(GoalPoseMsg)

if __name__ == "__main__":

    rospy.init_node("PathPlanner", anonymous=True)  # Initialize the node

    # Default values
    plan_topic = "/planner_node/car_plan"

    # # Start Pose
    start_pose = np.array([[2500, 640, 4.0]])

    # Start and Blue targets from the CSV files 
    plan_targets = np.array([[2600, 660, 1.8],
                            [1880, 440, 3.55],
                            [1435, 545, 2.7], 
                            [1250, 460, 3.0], 
                            [540, 835, 4.1 ]])

    #flip about X axis 
    start_pose[:,1]=2*618-start_pose[:,1]
    plan_targets[:,1]=2*618-plan_targets[:,1]

    plan=[]
    map_img, map_info = utils.get_map(MAP_TOPIC)

    utils.map_to_world(start_pose,map_info)
    utils.map_to_world(plan_targets,map_info)

    # Create a PathPlanner
    pp = PathPlanner(
        plan,
        plan_targets
    )

    # Loop over self.plan_targets 
    start_pose = start_pose[0,:]
    for elem in plan_targets: 

        raw_input("Initiate Publish")

        pp.publish_next_positions(start_pose, elem, rospy.get_rostime())

        # Waits for ENTER key press
        raw_input("Press Enter to when new plan is available")

        # Append the new plan into self.plan
        plan_msg = rospy.wait_for_message(plan_topic, PoseArray)
        plan.extend(plan_msg.poses)

        # Set new start positions 
        start_pose = elem

        # rospy.loginfo(plan_msg)

    #Pause to make sure the plan publishing works 
    rospy.sleep(5)
    raw_input("Press Enter to Publish the Plan")
    poseArrayMsg = PoseArray()
    poseArrayMsg.header.frame_id = "/map"
    poseArrayMsg.poses = plan
    pp.plan_pub.publish(poseArrayMsg)
    # Publish the pose Array 

    rospy.spin()  # Prevents node from shutting down

