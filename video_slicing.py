import numpy as np
import pandas as pd
import cv2 as cv
from cus_vo import PinholeCam, VisualOdometry

file_path = "C:/Users/line/Desktop/sam/data/video/indoor_detect_test.mp4"
video = cv.VideoCapture(file_path)

# video information
lenght = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
frame_cnt = int(video.get(cv.CAP_PROP_FRAME_COUNT))
fps = video.get(cv.CAP_PROP_FPS)

#video height, width, fx, fy, cx, cy
focial_length = (1118.12907,1121.49165)
principal_point = (553.850614,659.566864)
#camera distortion value k1,k2,p1,p2,k3

distortion = (0.10541647,-0.4997436,-0.02458957,0.00274952,0.67565388)

detector = cv.SIFT_create()

pinhole = PinholeCam(height, width,focial_length[0],focial_length[1], principal_point[0], principal_point[1], distortion[0], distortion[1],distortion[2],distortion[3],distortion[4] )

vo = VisualOdometry(pinhole)
traj = np.zeros((1000, 1000,3),dtype=np.uint8)

frames = []
for i in range(frame_cnt):
    ret,frame = video.read()
    if not ret :
        print("error") 
        break
    g_scale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    vo.update(g_scale, i)
    cur_t = vo.cur_t
    if i > 2:
        x, y, z = cur_t[0], cur_t[1], cur_t[2]
        
    else :
        x,y,z = 0, 0, 0
    
    draw_x, draw_y = int(np.squeeze(np.array(x))) + 500, int(np.squeeze(np.array(z))) + 500
    
    cv.circle(traj, (draw_x,draw_y),1, (0,255,0),2)    
    cv.rectangle(traj, (10, 20), (600,60), (0,0,0), -1)
    text = "coords X: %2fm, y:%2fm, z:%2fcm" %(x,y,z) 
    
    cv.putText(traj, text, (20,40), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1 ,8)
    
    cv.imshow('Facing_cam', frame)
    cv.imshow('Traj', traj)
    cv.waitKey(1)
    
