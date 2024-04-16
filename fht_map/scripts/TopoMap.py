import numpy as np
import cv2
import math
from networkx.generators import social
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN
import copy
import rospy
from robot_function import calculate_entropy,calculate_infor

class Vertex:

    def __init__(self, robot_name=None, id=None, pose=None, descriptor=None, localMap=None, local_image=None, local_laserscan_angle=None) -> None:
        self.robot_name = robot_name
        self.id = id
        self.pose = pose
        self.descriptor = descriptor
        self.localMap = localMap
        self.local_laserscan_angle = local_laserscan_angle
        # self.local_laserscan = local_laserscan #2*n array
        self.local_image = local_image
        self.descriptor_infor = 0
        self.local_free_space_rect = [0,0,0,0] #x1,y1,x2,y2 in map frame  ; x1< x2,y1 <y2
        
        if descriptor is not None:
            self.descriptor_infor = calculate_infor(local_image)

class Support_Vertex:
    def __init__(self, robot_name=None, id=None, pose=None) -> None:
        self.robot_name = robot_name
        self.id = id
        self.pose = pose
        self.local_free_space_rect = [0,0,0,0]

class Edge:
    
    def __init__(self, id, link) -> None:
        self.id = id
        self.link = link # [[last_robot_name, last_robot_id], [now_robot_name, now_vertex_id]]


class TopologicalMap:
    
    def __init__(self, robot_name='1', threshold=0.8) -> None:
        self.robot_name = robot_name
        self.vertex = list()#保存了所有节点
        self.edge = list()
        self.threshold = threshold
        self.offset_angle = 0 #Not used
        self.vertex_id = -1
        self.edge_id = 0
        self.x = np.array([])
        self.y = np.array([])
        self.map_resolution = float(rospy.get_param('map_resolution', 0.05))
    
    def insert(self, vertex=None, edge=None) -> None:
        self.vertex.append(vertex)
        self.edge.append(edge)

    def add(self, vertex=None) -> None:
        self.vertex_id += 1
        vertex.id = self.vertex_id
        current_node = vertex #add a new vertex
        self.vertex.append(vertex)

        return self.vertex_id, current_node
    
            