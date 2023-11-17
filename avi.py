import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import svds
from scipy.linalg import svd

img1 = cv.imread('3.jpg',0)
img1 = cv.resize(img1,(400,400))

img2 = cv.imread('4.jpg',0)
img2 = cv.resize(img2,(400,400))

sift = cv.SIFT_create()

kp1, desc1 = sift.detectAndCompute(img1,None)
kp2, desc2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm= FLANN_INDEX_KDTREE, trees =3)
search_params = dict(checks=10)

flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(desc1,desc2,k=2)

good = []
pts1 = []
pts2 = []

for i,(m,n) in enumerate(matches):
    if m.distance < .8 * n.distance :
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
        
pts1 = np.int32(pts1)   
pts2 = np.int32(pts2)
F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

def drawlines(img1,img2,lines,pts1,pts2):
    r, c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines,pts1,pts2): 
        color = np.random.randint(0,255,3).tolist()
        color = tuple(map(int,color))
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv.line(img1,(x0,y0),(x1,y1),color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2),2,F)
lines1 = lines1.reshape(-1,3)
img_1, img_2 = drawlines(img1,img2, lines1,pts1,pts2)

lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2),1,F)
lines2 = lines2.reshape(-1,3)
img_3, img_4 = drawlines(img2,img1,lines2,pts2,pts1)

print(F)

U, sigma, V = svd(F)
print(U.shape, sigma.shape, V.shape)
print('==========U sigma ============: \n',np.round(U, 3))
print('==========Sigma value=========== :\n',np.round(sigma, 3))
print('==========V transpsoe mat:==========\n',np.round(V,3))

num_component = 4
U_tr, sigma_tr, V_tr = svds(F, k=2)
matrix_tr = np.dot(np.dot(U_tr,np.diag(sigma_tr)),V_tr)
print("==========Truncated Matrix==========")
print(matrix_tr)
tri = cv.triangulatePoints(matrix_tr)

print("============= Triangulate ============")
print(tri)
# cv.imshow('left image with line', img_1)
# cv.imshow('left image with point', img_2)
# cv.imshow('right image with line', img_3)
# cv.imshow('right image with point',img_4)
# cv.imwrite('epipole1.jpg',img_1)
# cv.imwrite('epipole2.jpg',img_3)
cv.waitKey(0)
cv.destroyAllWindows()
    
    