import numpy as np
import cv2 as cv


STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1500

lk_params = dict(winSize  = (21, 21), 
				maxLevel = 2,
             	criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 28, 0.001))

def featureTracking(image_ref, image_cur, px_ref):
	kp2, st, err = cv.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]

	st = st.reshape(st.shape[0])
	kp1 = px_ref[st == 1]
	kp2 = kp2[st == 1]
    

	return kp1, kp2

class PinholeCam :
    def __init__(self,width,height,fx,fy,cx,cy,
                 k1=0.0, k2= 0.0, p1=0.0, p2=0.0, k3=0.0):  # distortion value
        
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.000001)
        self.d = [k1, k2, p1, p2, k3]


class VisualOdometry:
    def __init__(self, cam, annot):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame = None
        self.last_frame = None
        self.cur_R = None
        self.cur_t = None
        self.px_ref = None
        self.px_cur = None
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        #self.detector = cv.FastFeatureDetector_create(threshold=9, nonmaxSuppression=True)
        self.detector = cv.SIFT_create()
        #self.detector = cv.ORB_create() 얘는 아닌듯 함..
        #self.detector = cv.SURF_create()
        with open(annot) as f:
            self.annot = f.readlines()

    def getAbsScale(self, frame_id):
        ss = self.annot[frame_id - 1].strip().split()
        x_prev = float(ss[3])
        y_prev = float(ss[7])
        z_prev = float(ss[11])

        ss = self.annot[frame_id].strip().split()
        x = float(ss[3])
        y = float(ss[7])
        z = float(ss[11])

        self.trueX, self.trueY, self.trueZ = x, y, z
        return np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2 + (z - z_prev) ** 2)

    def processFirstFrame(self):
        self.px_ref = self.detector.detect(self.new_frame, None)
        self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
        self.frame_stage = STAGE_SECOND_FRAME

    def processSecondFrame(self):
        self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
        E, mask = cv.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv.FM_RANSAC,
                                      prob=0.999, threshold=.1)
        if E is not None:
            print("E mat:",E)
            print("shape E: ",E.shape)   
            E = E[:3, :]         
            E = E.reshape((3,3))    
            _, self.cur_R, self.cur_t, mask = cv.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp=self.pp)
            self.px_ref = self.px_ref.reshape(-1,1,2)
            self.px_ref = self.px_cur
            self.frame_stage = STAGE_DEFAULT_FRAME
        else : 
            print("Failed to compute E mat")

    def processFrame(self, frame_id):
        self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
        E, mask = cv.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv.RANSAC,
                                      prob=0.999, threshold=1.0)
        _, R, t, mask = cv.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp=self.pp)
        abs_scale = self.getAbsScale(frame_id)
        if abs_scale > 0.1:
            self.cur_t = self.cur_t + abs_scale * self.cur_R.dot(t)
            self.cur_R = R.dot(self.cur_R)
        if self.px_ref.shape[0] < kMinNumFeature:
            self.px_cur = self.detector.detect(self.new_frame, None)
            self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
        self.px_ref = self.px_cur

    def update(self, img, frame_id):
        assert (img.ndim == 2 and img.shape[1] == int(self.cam.height) and img.shape[0] == int(self.cam.width)),"Image dimensions do not match camera model"
        self.new_frame = img
        if self.frame_stage == STAGE_DEFAULT_FRAME:
            self.processFrame(frame_id)
        elif self.frame_stage == STAGE_SECOND_FRAME:
            self.processSecondFrame()
        elif self.frame_stage == STAGE_FIRST_FRAME:
            self.processFirstFrame()
        self.last_frame = self.new_frame


