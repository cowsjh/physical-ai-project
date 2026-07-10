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

#===============================          week2         ================================

RED = ([0,120,70],[10,255,255]), ([170,120,70], [180,255,255])
BLUE = ([100,150,50],[130,255,255])
GREEN = ([40,70,70],[80,255,255])
YELLOW = ([20,100,100],[40,255,255])
TARGET_COLOR = RED

def create_mask(frame):
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    mask = None
    for lower, upper in TARGET_COLOR :
        lower = np.array(lower,dtype=np.uint8)
        upper = np.array(upper,dtype=np.uint8)
        
        sample = cv2.inRange(hsv, lower,upper)
        if mask is None :
            mask = sample
        else :
            mask = cv2.bitwise_or(mask,sample)
    
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

MIN_AREA = 500

def find_objects(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    objects = []
    x, y, w, h = (0,0,0,0)
    area = 0
    for c in contours :
        area = cv2.contourArea(c)
        if area < MIN_AREA :
            continue
        x, y, w, h = cv2.boundingRect(c)    
        cx = x + w//2
        cy = y + h//2

        objects.append((cx, cy, area, (x,y,w,h)))

    return objects

def draw_detection(frame, objects):
    color = (50,50,255)
    for cx, cy, area, rec in objects :
        pt1 = (rec[0], rec[1])
        pt2 = (rec[0] + rec[2], rec[1] + rec[3])

        cv2.rectangle( frame, pt1, pt2, color )
        cv2.circle( frame, (cx,cy), 2, color )
        cv2.putText( frame, f"{cx}, {cy}",(rec[0],rec[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color )

    return frame

def detect(frame):
    mask = create_mask( frame )
    objects = find_objects( mask ) 
    draw_detection( frame, objects )

    return frame, mask, objects


def main ():
    cap = capture_cam()
    if not cap :
        return None

    fps_counter = FPSCounter(frame_len = 30)
    while True:
        ret,frame = cap.read()
        
        if not ret :
            print("프레임이 존재 하지 않습니다.")
            break

        fps = fps_counter.calc_fps()
        draw_fps(frame,fps)

        frame, mask, objects = detect(frame)
        objects_info = []
        objects_count = len(objects)
        for cx, cy, area, rec in objects :
            info = f"중심: ({cx}, {cy}) | 면적: {area}px²"
            objects_info.append(info)
        
        if objects_info != [] :
            info_str = ", ".join(str(item) for item in objects_info)
            print(f"{objects_count}개 탐지 -", info_str)

        cv2.imshow('Result',combine_frames(frame, mask))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__" :
    main()