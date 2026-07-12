import cv2
import numpy as np 
import matplotlib.pyplot as plt
import time
from collections import deque
import transform

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
    if contours != [] :
            
        objects = None
        max_area = 0
        
        for c in contours :
            area = cv2.contourArea(c)
            max_area = max(area,max_area)

        for c in contours :
            area = cv2.contourArea(c)
            if area == max_area :
                x, y, w, h = cv2.boundingRect(c)    
                cx = x + w//2
                cy = y + h//2

                objects = (cx, cy, area, (x,y,w,h))

        return objects
    else :
        return None


def draw_detection(frame, objects):
    if objects != None :
        color = (50,50,255)
        cx, cy, area, rec = objects 
        pt1 = (rec[0], rec[1])
        pt2 = (rec[0] + rec[2], rec[1] + rec[3])
        
        cv2.rectangle( frame, pt1, pt2, color )
        cv2.circle( frame, (cx,cy), 2, color )

        point = (cx, cy)
        res = (640,480)
        world_x, world_y = pixel_to_world(point, res)
        text = f"pixel: {cx}, {cy}\nworld: {world_x:.1f}, {world_y:.1f}"
        
        cv2.putText( frame, text,(rec[0]+rec[2],rec[1]), cv2.FONT_HERSHEY_SIMPLEX, .5, color )

        
    return frame

def detect(frame):
    mask = create_mask( frame )
    objects = find_objects( mask ) 
    draw_detection( frame, objects )

    return frame, mask, objects

#===============================          week3         ================================

SCALE = 1.0
ANGLE = 0.0

def pixel_to_world(point, resolution):
    w,h = resolution
    tx = -w*0.5
    ty = -h*0.5
    world_pos = transform.translate_2d(point,tx,ty)
    world_pos = transform.scale_matrix_2d(SCALE,SCALE) @ world_pos
    world_pos = transform.rotation_matrix_2d(ANGLE) @ world_pos
    return world_pos


def main ():

    global ANGLE, SCALE
    


    cap = capture_cam()
    if not cap :
        return None

    fps_counter = FPSCounter(frame_len = 30)
    
    #plt 시각화
    plt.ion()
    fig, ax = plt.subplots()
    plt.show()
    pixel_x_his = deque(maxlen=100)
    pixel_y_his = deque(maxlen=100)
    world_x_his = deque(maxlen=100)
    world_y_his = deque(maxlen=100)


    frame_count = 0
    while True:
        ret,frame = cap.read()
        
        if not ret :
            print("프레임이 존재 하지 않습니다.")
            break

        fps = fps_counter.calc_fps()
        draw_fps(frame,fps)

        frame, mask, objects = detect(frame)


        if objects != None :
            cx, cy, area, rec = objects
            point = (cx, cy)
            res = (640,480)
            world_x, world_y = pixel_to_world(point,res)
            info = f"pixel: ({cx}, {cy}) | world: ({world_x:.1f}, {world_y:.1f})"

            print(info)

    # plot 값 append
            pixel_x_his.append(cx)
            pixel_y_his.append(cy)
            world_x_his.append(world_x)
            world_y_his.append(world_y)

        angle = ((ANGLE + 180) % 360) - 180
        
        cv2.putText( frame, f"Angle: {angle}",(50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
        cv2.imshow('Result',combine_frames(frame, mask))

    # plot 시각화
        
        if frame_count % 3 == 0:
            ax.clear()
            ax.scatter(pixel_x_his, pixel_y_his, c='blue', label='pixel')
            ax.scatter(world_x_his, world_y_his, c='red', label='world')
            ax.legend()
            plt.pause(0.01)

        key = cv2.waitKey(1) & 0xFF
        if  key == ord('q'):
            break
        elif key == ord('a'):
            ANGLE += 5.0
        elif key == ord('d'):
            ANGLE -= 5.0
        elif key == ord('+'):
            SCALE += 0.1
        elif key == ord('-'):
            SCALE -= 0.1
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__" :
    main()

    