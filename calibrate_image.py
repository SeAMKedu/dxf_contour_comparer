import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def calibrate_with_single_image(img, grid_shape, square_width):
    """Calibrates a camera with a folder of images taken of a chessboard.

    Args:
        img (numpy array): The calibration image
        grid_shape (tuple): Number of inner corners in the chessboard (n_x, n_y)
        square_with (float): Square width in real-world units (i.e. in mm)

    Returns:
        float, string, 3x3 array, list, array, array : return value, message,
                                                        calibration matrix,
                                                        distortion coefficients,
                                                        new camera matrix,
                                                        object points
    """
    # If image is RGB, converting to gray scale
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objpoints = np.zeros((grid_shape[0]*grid_shape[1], 3), np.float32)
    objpoints[:, :2] = np.mgrid[0:grid_shape[0], 0:grid_shape[1]].T.reshape(-1, 2)
    objpoints *= square_width

    imgpoints = None # 2d points in image plane.

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(img, grid_shape, None)

    # If found, add object points, image points (after refining them)
    if ret:
        corners2 = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
        imgpoints = corners2

        # Calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objpoints], [imgpoints], img.shape[::-1], None, None)

        if ret:
            h, w = img.shape[:2]
            newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

            # Calculating calibration error
            imgpoints2, _ = cv2.projectPoints(objpoints, rvecs[0], tvecs[0], mtx, dist)
            error = cv2.norm(imgpoints, imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            msg = error
        else:
            msg = "Error occurred in calibration!"
    else:
        msg = "Could not find the chessboard corners!"
        mtx = dist = newcameramtx = objpoints = None

    return ret, msg, mtx, dist, newcameramtx, objpoints


def calibrate_main(chess_img, grid_shape, square_width_in_mm):
    """Main function for calibration

    Args:
        chess_img (numpy array): Image taken from a chessboard, the calibration image
        grid_shape (tuple): Number of rows and columns in the grid (cols, rows)
        square_width_in_mm (float): Chessboard square width in millimeters

    Returns:
        json-formed dict, string: Calibration data, error message (if 
                                    calibration was successful: reprojection error)
    """

    data = None

    # Convert to grayscale if needed
    if len(chess_img.shape) > 2:
        chess_img = cv2.cvtColor(chess_img, cv2.COLOR_BGR2GRAY)

    # Get the calibration matrix and other parameters
    ret, msg, mtx, dist, newcameramtx, opoints = \
        calibrate_with_single_image(chess_img, grid_shape, square_width_in_mm)

    if ret:
        # Undistort the original image
        cal_img_ud = cv2.undistort(chess_img, mtx, dist, None, newcameramtx)

        # Find the chess board corners from the undistorted image
        ret, corners = cv2.findChessboardCorners(cal_img_ud, grid_shape)

        # If found, add object points, image points (after refining them)
        if ret:
            result_img = cv2.cvtColor(cal_img_ud, cv2.COLOR_GRAY2BGR)
            result_img = cv2.drawChessboardCorners(result_img, grid_shape, corners, ret)

            # termination criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(cal_img_ud, corners, (11, 11), (-1, -1), criteria)

            corners = corners.reshape(-1, 2)
            nbrs = NearestNeighbors(n_neighbors=2).fit(corners)
            distances, _ = nbrs.kneighbors(corners)

            # Calculating ppmm
            mean_squre_width = np.mean(distances[:, 1])
            ppmm = mean_squre_width / square_width_in_mm

            # Perspective transformation (this could be done also applying
            # rvecs and tvecs, which could be smarter)
            dst_corners = opoints[:, :-1] * ppmm + corners[0]
            trans_matrix = cv2.findHomography(corners, dst_corners)[0]

            # To json type data
            data = json.dumps({"camera_matrix": mtx.tolist(),
                "distortion_coeffs": dist.tolist(),
                "new_camera_matrix": newcameramtx.tolist(),
                "perspective_transformation": trans_matrix.tolist(),
                "ppmm": ppmm}, indent=4)
            
            # For pyplot
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            plt.imshow(result_img)
            plt.title(f"""Found corners shown on the corrected image
                        reprojection error: {msg:.3f} pix""")
            plt.show(block=False)

        else:
            msg = "Could not find the chessboard corners from the undistorted image!"

    return data, msg