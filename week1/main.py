import cv2
import numpy as np
import matplotlib
import time
from collections import deque

def capture_cam () :
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened() :
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print(f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print('카메라가 정상적으로 연결 되었습니다.')
        return cap

    else :
        print('카메라가 연결 되지 않았습니다.')
        cap.release()
        return None


def grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def g_blur(frame,sigma):
    return cv2.GaussianBlur(frame,(0,0),sigma)

def preprocess(frame, th1, th2, sigma=1):
        gray = grayscale(frame.copy())
        blur = g_blur(gray.copy(),sigma)
        canny = cv2.Canny(blur.copy(), th1, th2, apertureSize = 5)

        canny_c = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

        return np.hstack((frame,canny_c))
        


def show_display (cap):

    if not cap :
        print(f"show_display : Camera가 연결 되지 않았습니다.")
        return None

    cv2.namedWindow('edge')

    cv2.createTrackbar('threshold1','edge',2000,5000,lambda x :None)
    cv2.createTrackbar('threshold2','edge',2000,5000,lambda x :None)

    start_time = time.time()
    p_t = start_time
    fps_per_frames = deque(maxlen=30)

    while True:
        ret,frame = cap.read()
        c_t = time.time()
        if not ret :
            print("프레임이 존재 하지 않습니다.")
            break
        
        th1 = cv2.getTrackbarPos('threshold1', 'edge')
        th2 = cv2.getTrackbarPos('threshold2', 'edge')

        ## set FPS
        f_t = c_t - p_t
        p_t = c_t
        fps = 1.0/f_t
        # print('fps: ' + str(fps))
        
        fps_per_frames.append(fps)
        avg_fps = np.mean(fps_per_frames)
        # print(len(fps_per_frames))
        # print('avg fps: ' + str(avg_fps))

        result = preprocess(frame,th1,th2)
        cv2.putText(result,f"FPS: {avg_fps:.1f}",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
        cv2.imshow('Result',result)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__" :
    cap = capture_cam()
    show_display(cap)