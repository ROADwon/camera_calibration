import numpy as np
import cv2 as cv
import os
import glob
import matplotlib.pyplot as plt
from odometry import PinholeCam, VisualOdometry
import pandas as pd

num = input("00 ~ 10 번까지 데이터셋이 준비되어 있습니다. 번호를 입력해 테스트할 데이터셋 번호를 입력해주세요")


file_path = f"C:/Users/line/Desktop/sam/data/imgs/KITTI{num}/"
pose_file = f"C:/Users/line/Desktop/sam/data/text/{num}.txt"
coords = []

image = glob.glob(f'{file_path}*.png')
image.sort()
pose = np.loadtxt(pose_file)
pinhole = PinholeCam(376.0, 1241.0,718.8560, 718.8560,607.1928, 185.2157)
#pinhole = PinholeCam(375.0,1242.0,718.8560, 718.8560,607.1928, 185.2157)
#pinhole = PinholeCam(370.0,1226.0,718.8560, 718.8560,607.1928, 185.2157)
vo = VisualOdometry(pinhole,pose_file)

traj = np.zeros((1000,1000,3), dtype=np.uint8)

for img_id in range(len(image)) :
    img = cv.imread(f'{file_path}' + str(img_id).zfill(6) + '.png', cv.IMREAD_GRAYSCALE)
    print(f'{img_id}, {img.ndim}, {img.shape}')
    vo.update(img,img_id)
    cur_t = vo.cur_t
    if img_id > 2 :
        x, y, z = cur_t[0], cur_t[1], cur_t[2]
        
    else :
        x, y, z =0, 0, 0
    
    draw_x, draw_y = int(np.squeeze(np.asarray(x))) + 500 , int(np.squeeze(np.asarray(z))) + 500
    true_x, true_y = int(np.squeeze(np.asarray(vo.trueX))) + 500, int(np.squeeze(np.asarray(vo.trueZ))) + 500# +290/ +90
    
    #cv.circle(traj, (draw_x, draw_y), 1, (img_id * 255 / 4540 - img_id * 255 / 4540 ,0), 1)
    cv.circle(traj, (draw_x, draw_y),1, (0,255,0), 2)
    cv.circle(traj, (true_x, true_y),1, (0, 0, 255), 2)
    cv.rectangle(traj, (10,20), (600, 60), (0, 0, 0), -1) 
    text = "coordinates  x :%2fm, y:%2fm, z:%2fm " % (x,y,z)
    
    cv.putText(traj, text, (20,40), cv.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
    
    cv.imshow('Facing CAM', img)
    cv.imshow('Trajectory', traj)
    cv.waitKey(1)
cv.waitKey(0)

cv.imwrite(f'map{num}.png', traj)

cv.destroyAllWindows()

    