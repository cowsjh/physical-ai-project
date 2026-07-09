import cv2
import numpy as np
import matplotlib
import time
from collections import deque

class FPSCounter :
    def __init__(self,frame_len = 30) :
        self.prev_time = time.time()
        self.history = deque(maxlen = frame_len)

    def calc_fps(self):
        current_time=time.time()
        d = current_time - self.prev_time
        self.prev_time = current_time
        fps = 1.0/max(d,1e-6)
        self.history.append(fps)
        return np.mean(self.history)

def capture_cam () :
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    
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

def to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def apply_blur(frame,sig):
    return cv2.GaussianBlur(frame,(0,0),sig)

def detect_edges(frame,th1,th2):
    return cv2.Canny(frame,th1,th2)

def preprocess(frame,th1,th2,sig):

    gray = to_grayscale(frame.copy())
    blur = apply_blur(gray.copy(),sig)
    canny = detect_edges(blur,th1,th2)

    return (gray,blur,canny)

def draw_fps(frame, fps):
    return cv2.putText(frame,f"FPS: {fps:.1f}",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))

def combine_frames(frame,edges):
    return np.hstack((frame,cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)))

def main ():
    cap = capture_cam()
    if not cap :
        return None

    cv2.namedWindow('Parameters')
    cv2.createTrackbar('threshold1','Parameters',50,200,lambda x :None)
    cv2.createTrackbar('threshold2','Parameters',50,200,lambda x :None)
    cv2.createTrackbar('sigma','Parameters',1,10,lambda x :None)

    fps_counter = FPSCounter(frame_len = 30)
    while True:
        ret,frame = cap.read()
        
        if not ret :
            print("프레임이 존재 하지 않습니다.")
            break

        th1 = cv2.getTrackbarPos('threshold1','Parameters')
        th2 = cv2.getTrackbarPos('threshold2','Parameters')
        sig = cv2.getTrackbarPos('sigma','Parameters')
        
        gray, blur, canny = preprocess(frame,th1,th2,max(sig,1))
        combined_frame = combine_frames(frame,canny)

        fps = fps_counter.calc_fps()
        result = draw_fps(combined_frame,fps)

        cv2.imshow('Result',result)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__" :
    main()