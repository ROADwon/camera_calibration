import numpy as np
import pandas as pd
import cv2 as cv

file_path = "C:/Users/line/Desktop/sam/data/video/test.mp4"
video = cv.VideoCapture(file_path)

# video information
length = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
frame_cnt = int(video.get(cv.CAP_PROP_FRAME_COUNT))
fps = video.get(cv.CAP_PROP_FPS)

# video height, width, fx, fy, cx, cy
focal_length = (1118.12907, 1121.49165)
principal_point = (553.850614, 659.566864)
# camera distortion value k1,k2,p1,p2,k3
cam_mat = np.array([[1118.12907, 0, 553.850614],
                    [0, 1121.49165, 659.566864],
                    [0, 0, 1]], dtype=np.float32)

distortion = np.array([0.10541647, -0.4997436, -0.02458957, 0.00274952, 0.67565388], dtype=np.float32)

detector = cv.ORB_create()
traj = np.zeros((1000,1000,3),dtype=np.uint8)
prev_frame = None
prev_kp = None
prev_desc = None

# 저장된 포즈를 저장할 리스트
trajectory = []

while True:
    ret, frame = video.read()

    if not ret:
        print("Error! 비디오 파일을 읽을 수 없습니다.")

    g_scale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    kp, desc = detector.detectAndCompute(g_scale, None)

    if prev_frame is not None:
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        matches = bf.match(prev_desc, desc)

        pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        E, mask = cv.findEssentialMat(pts1, pts2, cam_mat, method=cv.RANSAC, prob=0.999, threshold=1)
        _, rvec, t, mask = cv.recoverPose(E, pts1, pts2, cam_mat)

        # Pose 복원이 성공한지 확인
        if not (np.any(np.isnan(rvec)) or np.any(np.isnan(t))):
            # 현재 프레임의 카메라 위치를 저장
            trajectory.append(t.flatten())

            # 카메라 포즈를 시각화
            frame = cv.drawFrameAxes(frame, cam_mat, distortion, rvec, t, 1)

            x,y = int(t[0,0] + 300), int(t[2,0] + 300)
            cv.circle(traj,(x,y),3,(0,255,0), -1)
    prev_frame = g_scale
    prev_kp = kp
    prev_desc = desc
    cv.imshow("비주얼 오도메트리", frame)

    if cv.waitKey(30) & 0xFF == 27:
        break

# 비디오 종료 후 저장된 포즈를 시각화
trajectory = np.array(trajectory)
df_trajectory = pd.DataFrame(trajectory, columns=['x', 'y', 'z'])
df_trajectory.plot(x='x', y='z', kind='scatter', title='Trajectory')
cv.destroyAllWindows()
