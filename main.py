import cv2
import numpy as np
import matplotlib


def capCam () :
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

def nothing(x):
    pass

def grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def g_blur(frame,sigma):
    return cv2.GaussianBlur(frame,(0,0),sigma)

def preprocess(frame, th1, th2, sigma=1):
        gray = grayscale(frame.copy())
        blur = g_blur(gray.copy(),sigma)
        canny = cv2.Canny(blur.copy(), th1, th2, apertureSize = 5)

        cv2.imshow('Grayscale',gray)
        cv2.imshow('Gaussian',blur)
        cv2.imshow('Gaussian',canny)

def showDis (cap):

    if not cap :
        print(f"showDis : Camera가 연결 되지 않았습니다.")
        cap.release()
        return None

    cv2.namedWindow('edge')

    cv2.createTrackbar('threshold1','edge',2000,5000,nothing)
    cv2.createTrackbar('threshold2','edge',2000,5000,nothing)


    while True:
        ret,frame = cap.read()

        if not ret :
            print("프레임이 존재 하지 않습니다.")
            break
        
        
        th1 = cv2.getTrackbarPos('threshold1', 'edge')
        th2 = cv2.getTrackbarPos('threshold2', 'edge')

        # cv2.imshow('Camera',frame)
        preprocess(frame,th1,th2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__" :
    cap = capCam()
    showDis(cap)