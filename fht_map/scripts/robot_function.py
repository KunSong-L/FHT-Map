import rospy
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Quaternion, PoseStamped, Point
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
import copy
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from collections import Counter
from sklearn.neighbors import KDTree
from scipy.optimize import minimize


height_vertex = 0.5
def set_marker(robot_name, id, pose, color=(0.5, 0, 0.5), action=Marker.ADD, scale = 0.3,frame_name = "/map"):
    now = rospy.Time.now()
    marker_message = Marker()
    marker_message.header.frame_id = robot_name + frame_name
    marker_message.header.stamp = now
    marker_message.ns = "topological_map"
    marker_message.id = id
    marker_message.type = Marker.SPHERE
    marker_message.action = action

    now_vertex_pose = Point()
    now_vertex_pose.x = pose[0]
    now_vertex_pose.y = pose[1]
    now_vertex_pose.z = height_vertex
    now_vertex_ori = Quaternion()
    orientation = R.from_euler('z', pose[2], degrees=True).as_quat()
    now_vertex_ori.x = orientation[0]
    now_vertex_ori.y = orientation[1]
    now_vertex_ori.z = orientation[2]
    now_vertex_ori.w = orientation[3]

    marker_message.pose.position = now_vertex_pose
    marker_message.pose.orientation = now_vertex_ori
    marker_message.scale.x = scale
    marker_message.scale.y = scale
    marker_message.scale.z = scale
    marker_message.color.a = 1.0
    marker_message.color.r = color[0]
    marker_message.color.g = color[1]
    marker_message.color.b = color[2]

    return marker_message

def set_edge(robot_name, id, poses, type="edge", color = (0,1,0), scale = 0.05, frame_name = "/map"):
    now = rospy.Time.now()
    path_message = Marker()
    path_message.header.frame_id = robot_name + frame_name
    path_message.header.stamp = now
    path_message.ns = "topological_map"
    path_message.id = id
    if type=="edge":
        path_message.type = Marker.LINE_STRIP
        path_message.color.a = 1.0
        path_message.color.r = color[0]
        path_message.color.g = color[1]
        path_message.color.b = color[2]
    elif type=="arrow":
        path_message.type = Marker.ARROW
        path_message.color.a = 1.0
        path_message.color.r = 1.0
        path_message.color.g = 0.0
        path_message.color.b = 0.0
    path_message.action = Marker.ADD
    path_message.scale.x = scale
    path_message.scale.y = scale
    path_message.scale.z = scale

    point_1 = Point()
    point_1.x = poses[0][0]
    point_1.y = poses[0][1]
    point_1.z = height_vertex
    path_message.points.append(point_1)

    point_2 = Point()
    point_2.x = poses[1][0]
    point_2.y = poses[1][1]
    point_2.z = height_vertex
    path_message.points.append(point_2)

    path_message.pose.orientation.x=0.0
    path_message.pose.orientation.y=0.0
    path_message.pose.orientation.z=0.0
    path_message.pose.orientation.w=1.0

    return path_message

PRETRAINED = {
    'retrievalSfM120k-vgg16-gem'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-vgg16-gem-b4dcdc6.pth',
    'retrievalSfM120k-resnet101-gem'    : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
    # new networks with whitening learned end-to-end
    'rSfM120k-tl-resnet50-gem-w'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet50-gem-w-97bf910.pth',
    'rSfM120k-tl-resnet101-gem-w'       : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
    'rSfM120k-tl-resnet152-gem-w'       : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet152-gem-w-f39cada.pth',
    'gl18-tl-resnet50-gem-w'            : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-83fdc30.pth',
    'gl18-tl-resnet101-gem-w'           : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth',
    'gl18-tl-resnet152-gem-w'           : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet152-gem-w-21278d5.pth',
}

def get_net_param(state):
    net_params = {}
    net_params['architecture'] = state['meta']['architecture']
    net_params['pooling'] = state['meta']['pooling']
    net_params['local_whitening'] = state['meta'].get('local_whitening', False)
    net_params['regional'] = state['meta'].get('regional', False)
    net_params['whitening'] = state['meta'].get('whitening', False)
    net_params['mean'] = state['meta']['mean']
    net_params['std'] = state['meta']['std']
    net_params['pretrained'] = False

    return net_params

def if_frontier(window):
    if 100 in window: # 障碍物
        return False
    if 0 not in window: # 可通过
        return False
    if 255 not in window: # 未知
        return False
    return True

def detect_frontier_old(image):
    kernel_size = 1
    step_size = 3
    frontier_points = []
    shape = image.shape
    for i in range(0,shape[0]-kernel_size,step_size):
        for j in range(0,shape[1]-kernel_size,step_size):
            if if_frontier(image[i - kernel_size : i+kernel_size+1, j - kernel_size : j+kernel_size+1]): #找到已知和未知的边界
                frontier_points.append([i, j])
    
    return np.fliplr(np.array(frontier_points))

def detect_frontier(image):
    kernel = np.ones((3, 3), np.uint8)
    free_space = image ==0
    unknown = (image == 255).astype(np.uint8)
    obs = (image == 100).astype(np.uint8)
    expanded_unknown = cv2.dilate(unknown, kernel).astype(bool)
    expanded_obs = cv2.dilate(obs, kernel).astype(bool)
    near = free_space & expanded_unknown & (~expanded_obs)
    return np.fliplr(np.column_stack(np.where(near)))

def calculate_entropy(array):
    num_bins = 20
    hist, bins = np.histogram(array, bins=num_bins)
    probabilities = hist / len(array)
    probabilities = probabilities[np.where(probabilities != 0)] 
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy

def calculate_infor(image):
    #input an image; output: feature point number of a image
    method = 0
    if method == 0:
        # Initiate ORB detector
        orb = cv2.ORB_create(nfeatures=100000)
        # find the keypoints and descriptors with ORB
        keypoints, des = orb.detectAndCompute(image,None)
    else:
        # Initiate SIFT detector
        sift = cv2.SIFT_create()
        # find the keypoints and descriptors with SIFT
        keypoints, des = sift.detectAndCompute(image,None)

    fp_num = len(keypoints)

    return fp_num
def sparse_point_cloud(data,delta):
    data_num = len(data)
    choose_index = np.ones(data_num,dtype=bool)
    check_dick = dict()
    for index, now_point in enumerate(data):
        x = now_point[0]//delta
        y = now_point[1]//delta

        if (x,y) not in check_dick.keys():
            check_dick[(x,y)] = 1
        else:
            choose_index[index] = False
    
    return data[choose_index]

def expand_obstacles(map_data, expand_distance=2):
    map_binary = (map_data == 100).astype(np.uint8)
    kernel_size = 2 * expand_distance + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    expanded_map_binary = cv2.dilate(map_binary, kernel)
    extended_map = copy.deepcopy(map_data)
    extended_map[expanded_map_binary == 1] = 100

    return extended_map

def find_local_max_rect(image, seed_point, map_origin, map_reso):
    #input: image and (x,y) in pixel frame format start point
    #output rect position of (x1, y1, x2, y2) in pixel frame
    def find_nearest_obstacle_position(image, x, y):
        obstacles = np.argwhere((image == 100) | (image == 255))
        point = np.array([[y, x]])
        distances = cdist(point, obstacles)
        min_distance_idx = np.argmin(distances)
        nearest_obstacle_position = obstacles[min_distance_idx]
        nearest_obstacle_position = np.array([nearest_obstacle_position[1],nearest_obstacle_position[0]])
        return nearest_obstacle_position
    
    vertex_pose_pixel = (np.array(seed_point) - np.array(map_origin))/map_reso
    x = int(vertex_pose_pixel[0])
    y = int(vertex_pose_pixel[1])
    if image[y,x] != 0:
        return [0,0,0,0]
    height, width = image.shape
    nearest_obs_index = find_nearest_obstacle_position(image, x, y)
    

    # 定义左上角和右下角初始值
    if nearest_obs_index[0] < x:
        x1 = nearest_obs_index[0]
        x2 = min(2*x - x1,width)
    else:
        x1 = max(2*x - nearest_obs_index[0],0)
        x2 = nearest_obs_index[0]
    
    if nearest_obs_index[1] < y:
        y1 = nearest_obs_index[1]
        y2 = min(2*y - y1,height)
    else:
        y1 = max(2*y - nearest_obs_index[1],0)
        y2 = nearest_obs_index[1]
    
    if x1 == x:
        y1 += 1
        y2 -= 1
    elif y1 ==y:
        x1 += 1
        x2 -= 1
    else:
        x1 += 1
        y1 += 1
        x1 -= 1
        y2 -= 1
        
    free_space_flag = [True, True, True, True] #up,left,down,right
    while True in free_space_flag:
        if free_space_flag[0]:
            if y1 < 1 or np.any(image[y1-1, x1:x2+1]):
                free_space_flag[0] = False
            else:
                y1 -= 1
        if free_space_flag[1]:
            if x1 < 1 or np.any(image[y1:y2+1, x1-1]):   
                free_space_flag[1] = False
            else:
                x1 -= 1
        if free_space_flag[2]:
            if y2 > height -2 or np.any(image[y2+1, x1:x2+1]):
                free_space_flag[2] = False
            else:
                y2 += 1
        if free_space_flag[3]:
            if x2 > width -2 or np.any(image[y1:y2+1, x2+1]):
                free_space_flag[3] = False
            else:
                x2 += 1
    x1 = x1 * map_reso + map_origin[0]
    x2 = x2 * map_reso + map_origin[0]
    y1 = y1 * map_reso + map_origin[1]
    y2 = y2 * map_reso + map_origin[1]
    return [x1,y1,x2,y2]

def calculate_vertex_info(frontiers, cluser_eps=1, cluster_min_samples=7):
    # input: frontier; DBSCAN eps; DBSCAN min samples
    # output: how many vertex in this cluster
    dbscan = DBSCAN(eps=cluser_eps, min_samples=cluster_min_samples)
    labels = dbscan.fit_predict(frontiers)
    label_counts = Counter(labels)
    label_counts[-1] = 0
    vertex_infor = [label_counts[now_label] for now_label in labels]

    return vertex_infor

def outlier_rejection(input,dis_th = 0.1):
    #input: a list of estimation
    if len(input) < 4:
        return input

    estimated_center = []
    for now_input in input:
        R_map_i = R.from_euler('z', now_input[3][2], degrees=True).as_matrix()
        t_map_i = np.array([now_input[3][0],now_input[3][1],0]).reshape(-1,1)
        T_map_i = np.block([[R_map_i,t_map_i],[np.zeros((1,4))]])
        T_map_i[-1,-1] = 1

        R_nav_i1 = R.from_euler('z', now_input[2][2], degrees=True).as_matrix()
        t_nav_i1 = np.array([now_input[2][0],now_input[2][1],0]).reshape(-1,1)
        T_nav_i1 = np.block([[R_nav_i1,t_nav_i1],[np.zeros((1,4))]])
        T_nav_i1[-1,-1] = 1

        R_i1_i = R.from_euler('z', now_input[4][2], degrees=True).as_matrix()
        t_i1_i = np.array([now_input[4][0],now_input[4][1],0]).reshape(-1,1)
        T_i1_i = np.block([[R_i1_i,t_i1_i],[np.zeros((1,4))]])
        T_i1_i[-1,-1] = 1

        T_nav_map = T_nav_i1 @ T_i1_i @  np.linalg.inv(T_map_i) 
        rot = R.from_matrix(T_nav_map[0:3,0:3]).as_euler('xyz',degrees=True)[2]
        estimated_center.append([T_nav_map[0,-1], T_nav_map[1,-1], rot])

    estimated_center = np.array(estimated_center)
    estimated_center[:,2] /= 4
    # 创建KD树
    kdtree = KDTree(estimated_center)
    # 选择K值
    K = 2
    maintan_ratio = 0.5
    #至少保留百分之五十点
    # 计算每个点的K近邻
    distances, indices = kdtree.query(estimated_center, k=K)
    nearest_dis = distances[:,1]
    while len (np.where(nearest_dis<dis_th)[0] ) < maintan_ratio * len(input):
        dis_th = dis_th*1.1
    
    good_index = np.where(nearest_dis<dis_th)[0]
    new_input = [input[i] for i in good_index]
    # print("origin number:",len(input), "  final number:", len(new_input))

    return new_input

def pose_gragh_opt(input, angle_cost = 0):

    def pose_gragh_opt_cost(x):
        R_nav_map = R.from_euler('z', x[2], degrees=True).as_matrix()
        t_nav_map = np.array([x[0],x[1],0]).reshape(-1,1)
        T_nav_map = np.block([[R_nav_map,t_nav_map],[np.zeros((1,4))]])
        T_nav_map[-1,-1] = 1

        cost = 0
        for now_input in input:
            R_map_i = R.from_euler('z', now_input[3][2], degrees=True).as_matrix()
            t_map_i = np.array([now_input[3][0],now_input[3][1],0]).reshape(-1,1)
            T_map_i = np.block([[R_map_i,t_map_i],[np.zeros((1,4))]])
            T_map_i[-1,-1] = 1

            R_i1_i = R.from_euler('z', now_input[4][2], degrees=True).as_matrix()
            t_i1_i = np.array([now_input[4][0],now_input[4][1],0]).reshape(-1,1)
            T_i1_i = np.block([[R_i1_i,t_i1_i],[np.zeros((1,4))]])
            T_i1_i[-1,-1] = 1

            T_nav_i1_est = T_nav_map @ T_map_i @ np.linalg.inv(T_i1_i)
            rot_nav_i1_est = R.from_matrix(T_nav_i1_est[0:3,0:3]).as_euler('xyz',degrees=True)[2]

            R_nav_i1 = R.from_euler('z', now_input[2][2], degrees=True).as_matrix()
            t_nav_i1 = np.array([now_input[2][0],now_input[2][1],0]).reshape(-1,1)
            T_nav_i1 = np.block([[R_nav_i1,t_nav_i1],[np.zeros((1,4))]])
            T_nav_i1[-1,-1] = 1
            rot_nav_i1 = R.from_matrix(T_nav_i1[0:3,0:3]).as_euler('xyz',degrees=True)[2]
            
            cost += np.linalg.norm(T_nav_i1_est[0:2,-1] - T_nav_i1[0:2,-1]) + angle_cost * abs(rot_nav_i1_est -rot_nav_i1 )

        return cost

    x0 = np.array([1, 1,50])

    result = minimize(pose_gragh_opt_cost, x0,tol=1e-9)

    x_optimal = result.x
    f_optimal = result.fun
    return x_optimal

def change_frame(point_1, T_1_2):
    #T_1_2：[x,y,yaw]
    #point_1: [x,y,yaw] or [x,y]
    # return: the position of point in frame 2
    input_length = len(point_1)
    if input_length==2:
        point_1 = [point_1[0],point_1[1],0]

    R_1_2 = R.from_euler('z', T_1_2[2], degrees=False).as_matrix()
    t_1_2 = np.array([T_1_2[0],T_1_2[1],0]).reshape(-1,1)
    T_1_2 = np.block([[R_1_2,t_1_2],[np.zeros((1,4))]])
    T_1_2[-1,-1] = 1
    
    R_1_point = R.from_euler('z', point_1[2], degrees=False).as_matrix()
    t_1_point = np.array([point_1[0],point_1[1],0]).reshape(-1,1)
    T_1_point = np.block([[R_1_point,t_1_point],[np.zeros((1,4))]])
    T_1_point[-1,-1] = 1

    T_2_point =  np.linalg.inv(T_1_2) @ T_1_point
    rot = R.from_matrix(T_2_point[0:3,0:3]).as_euler('xyz',degrees=False)[2]

    result = [T_2_point[0,-1], T_2_point[1,-1], rot]
    return result[0:input_length]