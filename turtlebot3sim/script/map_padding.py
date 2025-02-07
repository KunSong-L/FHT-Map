#!/usr/bin/python3.8
from numpy.lib.function_base import _median_dispatcher
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
import numpy as np
import csv
import tf
import os
import glob
import re


class MapPadding:
    def __init__(self, robot_name) -> None:
        self.self_robot_name = robot_name
        self.map_pub = rospy.Publisher(
            robot_name+"/map", OccupancyGrid, queue_size=10)
        # self.pose_pub = rospy.Publisher(
        #     robot_name+"/testpose", PoseStamped, queue_size=10)
        self.map_timestamps = []
        self.zeros_counts = []
        print("robot number =", robot_num)
        if robot_num == 1:
            self.single_robot = 1
        else:
            self.single_robot = 0
        self.tf_listener = tf.TransformListener()


        rospy.Subscriber(
            robot_name+"/map_origin", OccupancyGrid, self.map_callback, queue_size=1)
    
    def map_callback(self, map):
        # print(map.info.origin.position)
        map_message = OccupancyGrid()
        map_message.header = map.header
        map_message.info = map.info
        # print("map orientation::", map.info.origin)
        padding = 100 #跑topoexplore需要给200padding，正常代码给10就行
        shape = (map.info.height, map.info.width)
        mapdata = np.asarray(map.data).reshape(shape)

        localMap = np.full((shape[0]+padding*2, shape[1]+padding*2), -1).astype(np.int8)
        localMap[padding:shape[0]+padding, padding:shape[1]+padding] = mapdata

        map_message.data = tuple(localMap.flatten())
        map_message.info.height += padding*2
        map_message.info.width += padding*2
        map_message.info.origin.position.x -= padding*map.info.resolution
        map_message.info.origin.position.y -= padding*map.info.resolution
        self.map_pub.publish(map_message)
        # before_send = np.asarray(map_message.data).reshape((map_message.info.height, map_message.info.width))
        # pose = PoseStamped()
        # pose.header.frame_id = map_message.header.frame_id
        # pose.pose.position = map_message.info.origin.position
        # pose.pose.orientation.z = 0
        # pose.pose.orientation.w = 1
        # self.pose_pub.publish(pose)
    
    def update_robot_pose(self):
        # ----get now pose----  
        #tracking map->base_footprint
        tmptimenow = rospy.Time.now()
        pose = [0,0]
        try:
            self.tf_listener.waitForTransform(robot_name+"/map", robot_name+"/base_footprint", tmptimenow, rospy.Duration(0.5))
            tf_transform, rotation = self.tf_listener.lookupTransform(robot_name+"/map", robot_name+"/base_footprint", tmptimenow)
            pose[0] = tf_transform[0]
            pose[1] = tf_transform[1]

        except:
            pass

        return pose


if __name__ == '__main__':
    
    rospy.init_node("map_padding")
    robot_name = rospy.get_param("~robot_name")
    robot_num = int(rospy.get_param("~robot_num"))
    node = MapPadding(robot_name)
    rospy.spin()