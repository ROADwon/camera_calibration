import os
import cv2 as cv
import numpy as np
import glob
import sys


img_path = "C:/Users/line/Desktop/sam/data/imgs/"

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((12*12,3), np.float32)
objp[:,:2] = np.mgrid[0:12,0:12].T.reshape(-1,2)

objpoints = []
imgpoints = []
pts_cnt = 0
pts = np.zeros((4,2),dtype=np.float32)
images = glob.glob(f'{img_path}*.jpg')
image_2D = cv.imread("2D_1.jpg")
image_2D_c = image_2D.copy()

vector_3D = np.array([
                        (0.0,0.0,0.0),
                        (13,0.0,0.0),
                        (0.0,13,0.0),
                        (13,13,0.0)
                    ],dtype="double")

def click_event (event, x,y,flags,params):
    global pts_cnt
    global pts
    global image_2D_c
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(image_2D_c,(x,y),5,(0,0,255),-1)
        cv.imshow("find_vector",image_2D_c)
        pts[pts_cnt] = (x,y)
        pts_cnt += 1
        if pts_cnt == 4:
            print("Select 2D worlds vector Successfully")
            print("press any key....")
cv.imshow("find_vector",image_2D_c)
cv.setMouseCallback("find_vector",click_event)
cv.waitKey(0)
cv.destroyAllWindows()

vector_2D = np.array([tuple(pts[0]),tuple(pts[1]),tuple(pts[2]),tuple(pts[3])])



for file in images:
    img = cv.imread(file)
    if img is None :
        print("Error : unable to read images at", file)
        continue
    g_scale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    ret, corners = cv.findChessboardCorners(g_scale,(12,12),cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
    
    if ret == True:
        objpoints.append(objp)
        
        corners2 = cv.cornerSubPix(g_scale,corners,(5,5),(-1,-1),criteria)
        imgpoints.append(corners2)
        
        img = cv.drawChessboardCorners(img, (12,12),corners2,ret)
        print("Successfully Processed Image at" ,file)
        #cv.imshow("results",img)
        cv.waitKey(0)
    else : 
        print("Processing Failed image at",file)




cv.destroyAllWindows()


h, w= img.shape[:2]

ret, matrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints,imgpoints,g_scale.shape[::-1],None,None)



T = tvecs


# Camera Matrix 카메라 내부 행렬
print("=====================================\n")
print("===Cam matrix : \n",matrix)
print("=====================================\n")
# Lens Distortion Conffidence 렌즈 왜곡 계수
print("=====================================\n")
print("===dist : \n",dist)
print("=====================================\n")

# 회전 벡터
print("=====================================\n")
print("===rvecs : \n",rvecs)
print("=====================================\n")

#이동 벡터
print("=====================================\n")
print("===tvecs : \n",tvecs)
print("=====================================\n")


    
retval, rvec, tvec = cv.solvePnP(vector_3D, vector_2D,matrix,dist,rvec=None,tvec=None, useExtrinsicGuess=None,flags=None)

R = cv.Rodrigues(rvec)
R = R[0]
t = tvec

print("=================================\n")
print("===R value : \n",R)
print("===t value : \n",t)
print("=================================\n")
