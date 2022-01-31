"""
Modified from the second answer of the topic:
https://stackoverflow.com/questions/20120384/iterative-closest-point-icp-implementation-on-python
"""

import cv2
import numpy as np
import copy
import pylab
import time
import sys
import random
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize

def res(p, src, dst):
    # Creating 3 x 3 rigid transformation matrix from the pose
    # vector 
    T = np.matrix([[np.cos(p[2]), -np.sin(p[2]), p[0]],
                    [np.sin(p[2]), np.cos(p[2]), p[1]],
                    [0, 0, 1]])
    
    # xt is src expressed in the homogeneous coordinates 
    n = np.size(src, 0)
    xt = np.ones([n, 3])
    xt[:, :-1] = src
    
    # Applying the transformation
    xt = (xt*T.T).A

    # Calculating the square of the distances between the
    # the transformed points and dst
    d = np.zeros(np.shape(src))
    d[:, 0] = xt[:, 0] - dst[:, 0]
    d[:, 1] = xt[:, 1] - dst[:, 1]
    r = np.sum(np.square(d[:, 0]) + np.square(d[:, 1]))
    return r


def jac(p, src, dst):
    """Function constructing the Jacobian matrix (i.e. the first 
    derivatives).

    Args:
        p (1 x 3 numpy array): pose vector
        src (n x 2 numpy array): source xy points
        dst (n x 2 numpy array): destination xy points

    Returns:
        1 x 3 numpy array: Jacobian matrix
    """

    # The first part is identical to the function res
    T = np.matrix([[np.cos(p[2]), -np.sin(p[2]), p[0]],
                    [np.sin(p[2]), np.cos(p[2]), p[1]],
                    [0, 0, 1]])
    n = np.size(src, 0)
    xt = np.ones([n, 3])
    xt[:, :-1] = src
    xt = (xt*T.T).A
    d = np.zeros(np.shape(src))
    d[:, 0] = xt[:, 0] - dst[:, 0]
    d[:, 1] = xt[:, 1] - dst[:, 1]
    
    # The derivative of the rotation matrix with respect to
    # theta (in here: p[2])
    dUdth_R = np.matrix([[-np.sin(p[2]), -np.cos(p[2])],
                        [ np.cos(p[2]), -np.sin(p[2])]])
    
    dUdth = (src*dUdth_R.T).A
    g = np.array([np.sum(2*d[:, 0]),
                    np.sum(2*d[:, 1]),
                    np.sum(2*(d[:, 0]*dUdth[:, 0] + d[:, 1]*dUdth[:, 1]))])
    return g


def hess(p, src, dst):
    """The function constructing the Hessian matrix (i.e. the 
    second derivatives).

    Args:
        p (1 x 3 numpy array): Pose vector
        src (n x 2 numpy array): Source xy points
        dst (n x 2 numpy array): Destination xy points

    Returns:
        3 x 3 numpy array: Hessian matrix
    """
    n = np.size(src, 0)
    T = np.matrix([[np.cos(p[2]), -np.sin(p[2]), p[0]],
    [np.sin(p[2]), np.cos(p[2]), p[1]],
    [0, 0, 1]])
    n = np.size(src, 0)
    xt = np.ones([n, 3])
    xt[:, :-1] = src
    xt = (xt*T.T).A
    d = np.zeros(np.shape(src))
    d[:, 0] = xt[:, 0]-dst[:, 0]
    d[:, 1] = xt[:, 1]-dst[:, 1]
    dUdth_R = np.matrix([[-np.sin(p[2]), -np.cos(p[2])], 
                        [np.cos(p[2]), -np.sin(p[2])]])
    dUdth = (src*dUdth_R.T).A
    H = np.zeros([3, 3])
    H[0, 0] = n*2
    H[0, 2] = np.sum(2*dUdth[:, 0])
    H[1, 1] = n*2
    H[1, 2] = np.sum(2*dUdth[:, 1])
    H[2, 0] = H[0, 2]
    H[2, 1] = H[1, 2]
    d2Ud2th_R = np.matrix([[-np.cos(p[2]), np.sin(p[2])], 
                            [-np.sin(p[2]), -np.cos(p[2])]])
    d2Ud2th = (src*d2Ud2th_R.T).A
    H[2, 2] = np.sum(2*(np.square(dUdth[:, 0]) + np.square(dUdth[:, 1]) + d[:, 0]*d2Ud2th[:, 0] + d[:, 0]*d2Ud2th[:, 0]))
    return H


def icp(a, b, max_time=1):
    t0 = time.time()
    init_pose = (0, 0, 0)
    src = np.array([a.T], copy=True).astype(np.float32)
    dst = np.array([b.T], copy=True).astype(np.float32)
    Tr = np.array([[np.cos(init_pose[2]), -np.sin(init_pose[2]), init_pose[0]],
                   [np.sin(init_pose[2]), np.cos(init_pose[2]), init_pose[1]],
                   [0,                    0,                   1          ]])
    src = cv2.transform(src, Tr[0:2])
    p_opt = np.array(init_pose)
    T_opt = np.array([])
    error_max = sys.maxsize
    first = False
    while not(first and time.time() - t0 > max_time):
        _, indices = NearestNeighbors(n_neighbors=1).fit(dst[0]).kneighbors(src[0])
        p = minimize(res, [0, 0, 0], args=(src[0], dst[0, indices.T][0]),
                    method='Newton-CG', jac=jac, hess=hess).x
        T = np.array([[np.cos(p[2]), -np.sin(p[2]), p[0]],
                        [np.sin(p[2]), np.cos(p[2]),p[1]]])
        p_opt[:2] = (p_opt[:2]*np.matrix(T[:2, :2]).T).A
        p_opt[0] += p[0]
        p_opt[1] += p[1]
        p_opt[2] += p[2]
        src = cv2.transform(src, T)
        Tr = (np.matrix(np.vstack((T, [0,0,1])))*np.matrix(Tr)).A
        error = res([0,0,0], src[0], dst[0, indices.T][0])

        if error < error_max:
            error_max = error
            first = True
            T_opt = Tr

    p_opt[2] = p_opt[2] % (2*np.pi)
    return T_opt, error_max


def find_rigid_transform(src, dst):
    p = minimize(res, [0, 0, 0], args=(src, dst),
                                method='Newton-CG', jac=jac, hess=hess).x
    T = np.array([[np.cos(p[2]), -np.sin(p[2]), p[0]],
                    [np.sin(p[2]), np.cos(p[2]), p[1]]])
    
    return T
    # src = cv2.transform(src, T)
    # print(src)
    # print(dst)

    # distances, _ = NearestNeighbors(n_neighbors=1).fit(dst[0]).kneighbors(src[0])
    # print(distances)


def test_main():
    n1 = 100
    n2 = 75
    bruit = 1/10
    center = [random.random()*(2-1)*3,random.random()*(2-1)*3]
    radius = random.random()
    deformation = 2

    # template = np.array([
    #     [np.cos(i*2*np.pi/n1)*radius*deformation for i in range(n1)], 
    #     [np.sin(i*2*np.pi/n1)*radius for i in range(n1)]
    # ])

    # data = np.array([
    #     [np.cos(i*2*np.pi/n2)*radius*(1+random.random()*bruit)+center[0] for i in range(n2)], 
    #     [np.sin(i*2*np.pi/n2)*radius*deformation*(1+random.random()*bruit)+center[1] for i in range(n2)]
    # ])

    saved = np.load("02p.npz")
    template = saved["p1"].T #* 7.13333333333333333333333333333/5
    data = saved["p2"].T

    T, error = icp(data, template)
    dx = T[0, 2]
    dy = T[1, 2]
    rotation = np.arcsin(T[0,1]) * 360 / 2 / np.pi

    print("T", T)
    print("error", error)
    print("rotation°", rotation)
    print("dx", dx)
    print("dy", dy)

    result = cv2.transform(np.array([data.T], copy=True).astype(np.float32), T).T
    plt.plot(template[0], template[1], label="template")
    plt.plot(data[0], data[1], label="data")
    plt.plot(result[0], result[1], label=f"result: {rotation:.2f} ° - [{dx:.2f}, {dy:.2f}]")
    plt.legend(loc="upper left")
    plt.axis('square')
    plt.show()

def test_main2():
    src = np.array([[0, 0], [0, 5], [5, 5], [5, 0]], np.float32)
    a = np.pi / 9.0
    M = np.array([[np.cos(a), -np.sin(a), 3],
                    [np.sin(a), np.cos(a), 1]], np.float32)
    dst = cv2.transform(src.reshape(1, -1, 2), M)

    find_rigid_transform(src.reshape(1, -1, 2), dst)

def test_main3():

    # 1. Ensin karkeasti bboxeilla rigidin kautta muunnokset (4)
    # 2. Kappaleen äärireunoilla (reiät mukaan luettuna) lähimmillä naapureilla etäisyyksien summa
    # 3. Se muunnos, millä kakkoskohdassa pienin summa, alkuarvausmuunnokseksi
    # 4. icp

    saved = np.load("02.npz")
    src = saved["p1"]
    dst = saved["p2"]
    distances, _ = NearestNeighbors(n_neighbors=1).fit(dst).kneighbors(src)
    sum_dist = np.sum(distances)
    print(sum_dist)

    plt.plot(src[:, 0], src[:, 1], label="template")
    plt.plot(dst[:, 0], dst[:, 1], label="data")
    plt.axis('square')
    plt.show()


if __name__ == "__main__":
    test_main()