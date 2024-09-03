import numpy as np

def rotation_matrix(rvector: list):
    #return Rx(rvector[0]) @ Ry(rvector[1]) @ Rz(rvector[2])
    return Rz(rvector[2]) @ (Ry(rvector[1]) @ Rx(rvector[0]))

def Ry(angle: float):
    return np.array([
        [np.cos(angle),     0,  -np.sin(angle)],
        [0,                 1,              0],
        [-np.sin(angle),    0,   np.cos(angle)]
    ])

def Rx(angle: float):
    return np.array([
        [1,     0,              0],
        [0,     np.cos(angle),  -np.sin(angle)],
        [0,     np.sin(angle),  np.cos(angle)]
    ])

def Rz(angle: float):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle),  0],
        [0,             0,              1]
    ])

