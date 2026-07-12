import numpy as np

def rotation_matrix_2d(theta) :
    rad = np.deg2rad(theta)
    return np.array([
        [np.cos(rad), -np.sin(rad)],
        [np.sin(rad), np.cos(rad)]
    ])

def scale_matrix_2d(sx, sy):
    return np.array([
        [sx, 0],
        [0, sy]
    ])

def translate_2d(point,tx,ty):
    ident = np.eye(2)
    cx, cy = point
    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ]) @ np.array([cx,cy,1])
    ident = T[:2]
    return ident

