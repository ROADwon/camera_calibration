import numpy as np
import cv2 as cv
import pandas as pd

video_file_path = "C:/Users/line/Desktop/sam/data/video/test.mp4"
file_save_path = "C:/Users/line/Desktop/sam/out/frames/"
video = cv.VideoCapture(video_file_path)

width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
frame_cnt = int(video.get(cv.CAP_PROP_FRAME_COUNT))
fps = video.get(cv.CAP_PROP_FPS)

focial_length = (1118.12907,1121.49165)
principal_point = (553.850614,659.566864)

cam_mat = np.array([[1118.12907, 0, 553.850614],
            [0, 1121.49165, 659.566864],
            [0, 0 ,1]],dtype= np.float32)

distortion = (0.10541647,-0.4997436,-0.02458957,0.00274952,0.67565388)


lk_params = dict(winSize= (21,21),
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))


vector_3D = np.array([
    (0.0, 0.0, 0.0),
    (13, 0.0, 0.0),
    (0.0, 13, 0.0),
    (13, 13, 0.0)
], dtype="double")

vector_2D =  np.array([
                    (195, 376),
                    (874, 366),
                    (879, 1050),
                    (188, 1050)],dtype=np.float32)

annot = []
for i in range(frame_cnt - 1):
    ret, ref_frame = video.read()
    if not ret :
        print("can not read images")
        break
    ret, cur_frame = video.read()
    print(f"process {i} img")
    if not ret:
        print("can not read images")
        break
    
    detector = cv.SIFT_create()
    
    ref_frame_gray = cv.cvtColor(ref_frame, cv.COLOR_BGR2GRAY)
    cur_frame_gray = cv.cvtColor(cur_frame, cv.COLOR_BGR2GRAY)
    if i < 900:
        kp1, des1 = detector.detectAndCompute(ref_frame_gray, None)
        kp2, des2 = detector.detectAndCompute(cur_frame_gray, None)
    else :
        kp1, des1 = kp1_prev, des1_prev
    
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < .8 * n.distance :
            good_matches.append(m)
            
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        
    E,mask = cv.findEssentialMat(src_pts, dst_pts, cam_mat, method=cv.RANSAC, prob=.999, threshold=1)
    _, R ,t, _  = cv.recoverPose(E,src_pts.copy(),dst_pts.copy(),cam_mat)
     
    
    R = cv.Rodrigues(R)
    
    kp1_prev, des1_prev = kp1, des1

    print(f"successfully append {i} frame truth")
    cv.imwrite(f'{file_save_path}' + str(i).zfill(6) + '.jpg', ref_frame)
  #  rvec, _ = cv.Rodrigues(annot[i][0])
    pose_mat = np.hstack((R,t.reshape(-1,1)))
    annot.append(pose_mat)
    
    
#ground_truth = np.zeros((frame_cnt -1 ,6))


print("Ground_Truth \n")
print(annot)
np.savetxt('ground_truth.txt',np.array(annot).reshape((frame_cnt -1 ,-1)), delimiter=',', fmt='%f')

    
    