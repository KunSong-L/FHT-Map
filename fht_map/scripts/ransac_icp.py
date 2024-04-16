import open3d as o3d
import numpy as np
import random
import sys
import copy
from sklearn.decomposition import PCA
import math


def Distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def notTooClose(p,disThreshold):
    return np.linalg.norm(p[0] - p[1]) > disThreshold

def farAway(l1, l2,lengthThreshold):
    ss = np.linalg.norm(l1)
    tt = np.linalg.norm(l2)
    return abs(ss - tt) > lengthThreshold * (ss + tt)

def display_inlier_outlier(pcd, ind):
    inlier_cloud = pcd.select_by_index(ind)
    outlier_cloud = pcd.select_by_index(ind, invert=True)
    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def estimateAvgDis(points):
    sample = random.sample(list(points), 10)
    dis =  []
    for i in range(10):
        for j in range(i,10):
            p1 = sample[i]
            p2 = sample[j]
            dis.append(Distance(p1,p2))
    disThreshold = np.mean(dis)/2
    return disThreshold

def prepare_icp(pcd, color, volSize, downSave=False, outlier=False, draw=False, pcaTag=False):
    pcd.paint_uniform_color(color)
    oldPcd = copy.deepcopy(pcd)
    
    oldNum = np.asarray(oldPcd.points).shape[0]

    if downSave:
        while True:
            volSize *= 1.1
            pcd = oldPcd.voxel_down_sample(voxel_size=volSize)
            tmp = np.asarray(pcd.points).shape[0]
            if  tmp <= min(10000, oldNum-1):
                break
    else:
        pcd = oldPcd.voxel_down_sample(voxel_size=volSize)
    if outlier:
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.95)
        if draw:
            display_inlier_outlier(oldPcd, ind)

    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=10))
    # o3d.visualization.draw_geometries([pcd],
    #                               zoom=0.3412,
    #                               front=[0.4257, -0.2125, -0.8795],
    #                               lookat=[2.6172, 2.0475, 1.532],
    #                               up=[-0.0694, -0.9768, 0.2024],
    #                               point_show_normal=True)

    KDT = o3d.geometry.KDTreeFlann(pcd)
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamKNN(knn=20))
    if pcaTag:
        pca = PCA(n_components=pcaTag)
        pca.fit(fpfh.data.transpose())
        fpfh.data = pca.transform(fpfh.data.T).T
    fpfhKDT = o3d.geometry.KDTreeFlann(fpfh)

    return KDT, fpfhKDT, oldPcd, pcd, fpfh.data.T

def calculateTrans(src, tgt):
    assert src.shape == tgt.shape
    src = np.array(src)[:,0:2]
    tgt = np.array(tgt)[:,0:2]
    num = src.shape[0]
    srcAvg = np.mean(src, axis=0).reshape(1,-1)
    tgtAvg = np.mean(tgt, axis=0).reshape(1,-1)
    src -= np.tile(srcAvg, (num, 1))
    tgt -= np.tile(tgtAvg, (num, 1))
    H = np.transpose(src) @ tgt
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    T = -R @ srcAvg.T + tgtAvg.T

    tmp_R = np.eye(3)
    tmp_R[0:2,0:2] = R
    T = np.array([T[0][0],T[1][0],0]).reshape((3,1))
    return tmp_R, T

def ICP(src, tgt,tgtKDT,fitThreshold):
    # print("ICPing...")
    limit = fitThreshold
    retR = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    retT = np.array([[0], [0], [0]])
    trace = []
    srcNum = src.shape[0]
    for icp_num in range(500):
        tgtCorr = []
        srcCorr = []
        for point in src:
            k, idx, dis2 = tgtKDT.search_knn_vector_3d(point, knn=1)
            if dis2[0] < (limit)**2:
                srcCorr.append(point)
                tgtCorr.append(tgt[idx[0]])
        trace.append([limit, len(srcCorr)])
        R, T = calculateTrans(np.array(srcCorr), np.array(tgtCorr))
        retR = R @ retR
        retT = R @ retT + T
        src = np.transpose((R @ src.T) + np.tile(T, (1, srcNum)))
        limit = (limit - fitThreshold/1.5) * 0.95 + fitThreshold/1.5
        if len(trace) > 100 and len(set([x[1] for x in trace[-20:]])) == 1:
            break
    # print("ICP trace is:", trace[::5])
    return retR, retT


def ransac_icp(source_pc, target_pc,init_yaw_guess, vis = False):
    volSize = 0.05
    disThreshold = 0.2
    lengthThreshold = 0.1
    fitThreshold = 3 * volSize
    srcKDT, srcFpfhKDT, oldSrc, src, srcFpfh = prepare_icp(source_pc, [1, 0, 0],volSize, downSave=False, outlier=False)
    tgtKDT, tgtFpfhKDT, oldTgt, tgt, tgtFpfh = prepare_icp(target_pc, [0, 1, 0], volSize, downSave=False ,outlier=False)

    # o3d.visualization.draw_geometries([src, tgt])

    srcPoints = np.array(src.points)
    tgtPoints = np.array(tgt.points)
    srcNum = np.asarray(srcPoints).shape[0]
    tgtNum = np.asarray(tgtPoints).shape[0]
    disThreshold = estimateAvgDis(srcPoints)

    # do ransac

    maxCount = 0
    fit_times=0
    total_time = 0
    max_fit_time = 0
    # print("RANSACing...")
    failed_reason = [0,0,0]
    max_intertation = 20000 #原来是20000
    while total_time < max_intertation:
        total_time += 1
        srcCorr = random.sample(range(srcNum), 2)
        if not notTooClose([srcPoints[x] for x in srcCorr],disThreshold):
            failed_reason[0] += 1
            continue
        tgtCorr = []
        for id in srcCorr:
            k, idx, dis2 = tgtFpfhKDT.search_knn_vector_xd(srcFpfh[id], knn=1)
            tgtCorr.append(idx[0])
        if farAway(srcPoints[srcCorr[1]] - srcPoints[srcCorr[0]], tgtPoints[tgtCorr[1]] - tgtPoints[tgtCorr[0]],lengthThreshold):
            failed_reason[1] += 1
            continue
        
        # if True in [farAway(srcPoints[i[0]] - srcPoints[j[0]], 
        #                     tgtPoints[i[1]] - tgtPoints[j[1]],lengthThreshold) 
        #             for i in zip(srcCorr, tgtCorr) 
        #             for j in zip(srcCorr, tgtCorr)]:
        #     continue

        R, T = calculateTrans(np.array([srcPoints[i] for i in srcCorr]), 
                              np.array([tgtPoints[i] for i in tgtCorr]))
        # print("estimated is: ", math.atan2(R[1,0],R[0,0])/math.pi*180, "error is: ", math.sin(init_yaw_guess - math.atan2(R[1,0],R[0,0])))
        
        #在这里引入角度约束！！！！！！！！！！！
        if init_yaw_guess:
            if not abs(math.fmod(init_yaw_guess - math.atan2(R[1,0],R[0,0])+ math.pi, 2*math.pi)- math.pi)  < 0.3:
                failed_reason[2] += 1
                continue
        
        A = np.transpose((R @ srcPoints.T) + np.tile(T, (1, srcNum)))
        fit_times += 1
        count = 0
        for point in range(0, srcNum, 1):           
            k, idx, dis2 = tgtKDT.search_hybrid_vector_3d(A[point], 
                                                          radius=fitThreshold, max_nn=1)
            count += k
        if count > maxCount:
            maxCount = count
            bestR, bestT = R, T
            max_fit_time = fit_times     

        if fit_times > 300:#原来是300
            break
    ransac_match_ratio = maxCount/min(srcNum,tgtNum)
    
    # print("RANSAC calculated %d times, maximum matched number: %d, max match find in time: %d" % (fit_times, maxCount, max_fit_time))
    # print("RANSAC MATCHED Ratio: %f"%(ransac_match_ratio))

    if ransac_match_ratio < 0.5:
        print("RANSCA not matched")
        if total_time == max_intertation:
            print("failed Reason:\n",failed_reason)
        return None, None
    
    R1 = bestR
    T1 = bestT

    # R1 = np.eye(3,dtype=float)
    # T1 = np.array([0,-0.5,0]).reshape(3,1)

    # print("RANSAC RESULT IS: \n", R1,"\n",T1)
    # print("RANSAC YAW ANGLE IS: \n", math.atan2(R1[1,0],R1[0,0])/math.pi*180)
    A = np.transpose((R1 @ srcPoints.T) + np.tile(T1, (1, srcNum)))
    A=o3d.utility.Vector3dVector(A)
    src.points = A

    R2, T2 = ICP(np.array(A), tgtPoints,tgtKDT,0.2)
    R = R2 @ R1
    T = R2 @ T1 + T2
    if vis:
        A = np.array(source_pc.points)
        A = np.transpose((R @ np.array(A).T) + np.tile(T, (1, A.shape[0])))
        A=o3d.utility.Vector3dVector(A)
        source_pc.points = A
        o3d.visualization.draw_geometries([source_pc, target_pc])
    # print('\nrotation:\n', R)
    # print('transition:\n', T)

    return R, T



if __name__ == "__main__":
    import time
    srcPath = "/home/master/debug/robot1_local_map.pcd"
    tgtPath = "/home/master/debug/robot2_local_map.pcd"

    source_pc = o3d.io.read_point_cloud(srcPath)
    target_pc = o3d.io.read_point_cloud(tgtPath)

    start = time.time()
    R,T = ransac_icp(source_pc, target_pc,0,0)
    print(R,T)
    end = time.time()

    print(f"time using {end - start}")

    