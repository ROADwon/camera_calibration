import cv2 as cv
import numpy as np
import glob
import os
import sys

file_path = "data/video/indoor_detect_test.mp4"
save_file_path = "C:/Users/line/Desktop/sam/out/imgs/"
video = cv.VideoCapture(file_path)

length = int(video.get(cv.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv.CAP_PROP_FPS)
frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm= FLANN_INDEX_KDTREE, tree =5)
search_params = dict(check=50)
images = []
good = []
pts1 = []
pts2 = []

for i in range(frame_count) :
    ret, frame = video.read()
    if ret :
        images.append(frame)
        image_path = f"{save_file_path}frame_{i}.jpg"
        print("===== successfully slicing frames =====")
        print(f"===== {images[i].shape} =====")
        
        cv.imwrite(image_path,images[i])
        if i == 20:
            break
        


'''
# fundamental Matrix in video

for i in range(frame_count + 2) :
    ret, frame  = video.read()
    if ret :
        images.append(frame)
        image_path = f"{save_file_path}{i}.jpg"
        print(f"====== successfully slicing {i} imgs ====== ")
        print(f"============ {frame.shape} ================")
    if i > 0 :
        img_1 = images[i]    
        img_2 = images[i-1]

        gray1 = cv.cvtColor(img_1, cv.COLOR_RGB2GRAY)
        gray2 = cv.cvtColor(img_2, cv.COLOR_RGB2GRAY)
            

        detector = cv.SIFT_create()
        kp1, desc1 = detector.detectAndCompute(gray1, None)
        kp2, desc2 = detector.detectAndCompute(gray2, None)

        matcher = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
        #matcher = cv.FlannBasedMatcher(index_params,search_params)
        matches = matcher.match(desc1,desc2)
        res = cv.drawMatches(img_1,kp1,img_2,kp2,matches, None, flags= cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        print("======== successfully finished feature mapping =========")
        # cv.imwrite(f'{save_file_path}{i}.jpg',res)
        
        #cv.imshow(f"res_{i}", res)
        if i == 20 :
            break
        
if matches is not None :
    for i, match in enumerate(matches):
    #    m, n = match
        if match.distance < .8*match.distance:
            good.append(match)
            pts2.append(kp2[match.trainIdx].pt)
            pts1.append(kp1[match.queryIdx].pt)

    pts1 = np.float32(pts1).reshape(-1, 1, 2)
    pts2 = np.float32(pts2).reshape(-1, 1, 2)
    F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
    if mask is not None:
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]
                        
def drawlines(img_1,img_2,lines,pts1,pts2):
    for r, pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int,[0,-r[2]/r[1]])
        x1,y1 = map(int,[c, -(r[2]+r[0]*c)/r[1]])
        img_1 = cv.line(img_1,(x,y0),(x1,y1),color,1)
        img_1 = cv.circle(img_1,tuple(pt1),5,color,-1)
        img_2 = cv.circle(img_2,tuple(pt2),5,color,-1)
    return img_1, img_2

if F is not None:
    # F 행렬이 제대로 반환되었는지 확인
    assert F.shape == (3, 3), "Fundamental Matrix의 크기가 3x3이 아닙니다."

    # 나머지 코드 수행
    lines_1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines_1 = lines_1.reshape(-1, 3)

    img1, img2 = drawlines(img1, img2, lines_1, pts1, pts2)

    lines_2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines_2 = lines_2.reshape(-1, 3)

    img3, img4 = drawlines(img2, img2, lines_2, pts2, pts1)

    cv.imshow('left images with line', img1)
    cv.imshow('left images with points', img2)
    cv.waitKey(0)
    cv.destroyAllWindows()
    video.release()
else:
    print("Fundamental Matrix를 찾을 수 없습니다.")
    
    '''