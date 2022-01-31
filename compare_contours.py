import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.neighbors import NearestNeighbors
from icp import icp


class ContourComparer():
    """Class for comparing hierarchical OpenCV contours. The matchShapes
    method of OpenCV compares only the external contours but not the holes.
    This implementation is designed for machine part and thus it has only
    two levels of hierarchy: the external contours and  the hole contours.

    Example usage:

        comparer = ContourComparer()
        comparer.set_model_contours(model_contours)

        ret = comparer.match_contour_to_model(contours_of_interest, 10, img)

    The arguments 'model_contours' and 'contours_of_interest' are lists of contours
    as returned by the findContours function of OpenCV. The numeral 10 means the 
    maximum allowed point-to-point distance (in pixels) between the model and the 
    interest contour, 'img' is the image the contours_of_interest has been extracted 
    from as a numpy array. If the image is given, the contours are visualized on
    the image (default: None) 
    """
    def __init__(self):
        self.__model_contours = None

    def set_model_contours(self, cnts):
        self.__model_contours = cnts


    @staticmethod
    def get_orientation_pca(pts):
        """Function calculates the orientation of the object based on the 
        angle of its first principal axes.

        Args:
            pts (n x 2 numpy array): xy points

        Returns:
            float, tuple: angle in radians, center point (x, y)
        """

        # Perform PCA analysis and calculate the center
        mean = np.empty((0))
        mean, eigenvectors, _ = cv2.PCACompute2(pts.reshape(-1, 2).astype(np.float64), mean)
        cntr = np.int0(mean[0])

        # Calculate the angle of the object
        angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) # orientation in radians
        
        return angle, cntr

    @staticmethod
    def get_trans_matrix_and_transform(theta, center_point, center_shift, contour):
        """Generates transformation matrix for rotating contour points around
        a certain center point and shifting the center (translating the points)

        Args:
            theta (float): Angle of rotation in radians
            center_point (1x2 numpy array): Centerpoint of rotation: x, y
            center_shift (1x2 numpy array): Centerpoint shift: delta_x, delta_y
            contour (numpy array): Contour points as returned by OpenCV

        Returns:
            tuple: transformed contour, transformation matrix
        """

        trans_matrix = np.array([[np.cos(theta), -np.sin(theta), center_shift[0]],
                [np.sin(theta), np.cos(theta), center_shift[1]]])
        cnt_trans = cv2.transform(contour - center_point, trans_matrix, -1) + center_point

        return cnt_trans, trans_matrix

    @staticmethod
    def flip_contours(contours):
        """Flips a list of contours around the x-axis of the
        outermost contour

        Args:
            contours (list): List of contours as returned by findContours
                            with the method RETR_TREE 

        Returns:
            list: List of flipped contours in same format
        """

        # Finding the centroid of the outermost contour
        M = cv2.moments(contours[0])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        # Flipping the contours one by one
        contours_flipped = []
        for cnt in contours:
            cnt_norm = cnt - [cx, cy]
            cnt_flipped = cnt_norm.copy()
            cnt_flipped[:, :, 1] *= -1 
            cnt_flipped += [cx, cy]
            contours_flipped.append(cnt_flipped)
        
        return contours_flipped

    def find_initial_guess_transform(self, model_contours, cnts):
        
        theta_model, center_model = self.get_orientation_pca(model_contours[0])
        theta_part, center_part = self.get_orientation_pca(cnts[0])
        theta = theta_part - theta_model
        center_shift = [center_part[0] - center_model[0], center_part[1] - center_model[1]]

        cnt_trans, trans_matrix = self.get_trans_matrix_and_transform(theta,
            center_model, center_shift, model_contours[0])
        cnt_trans_180, trans_matrix_180 = self.get_trans_matrix_and_transform(theta + np.pi,
            center_model, center_shift, model_contours[0])

        # Checking, which fits better
        distances, _ = NearestNeighbors(n_neighbors=1).fit(
            cnt_trans.reshape(-1, 2)).kneighbors(cnts[0].reshape(-1, 2))
        distances_180, _ = NearestNeighbors(n_neighbors=1).fit(
            cnt_trans_180.reshape(-1, 2)).kneighbors(cnts[0].reshape(-1, 2))
        if np.sum(distances_180) < np.sum(distances):
            cnt_trans = cnt_trans_180
            trans_matrix = trans_matrix_180
            min_sum = np.sum(distances_180)
        else:
            min_sum = np.sum(distances)

        return cnt_trans, trans_matrix, center_model, min_sum


    def match_contour_to_model(self, cnts, max_dev, img_ppmm, mdl_ppmm, img, figure_mode="show",
                                savename=None):
        """Checking if the contours got from the image match with the
        defined model contours.

        Args:
            cnts (list of numpy arrays): Contours list as returned by OpenCV
            max_dev (float): Maximum deviation from the model (in pixels)
            img_ppmm (float): Pixel per millimeter ratio for the image.
            mdl_ppmm (float): Pixel per millimeter ratio for the model image.
            img (numpy array): Image to overlay the results on.
            figure_mode (string, optional): Either "show" (shows the result) or 
                                            "return" (returns the result as an numpy array
                                            image). Defaults to "plot".
            savename (string, optional): If given, the name the result image will be saved

        Raises:
            ValueError: If the parameter figure_mode is not "show" or "return"

        Returns:
            bool, numpy array: Did the part pass, the result image (None if figure_mode=="show")
        """

        final_result = True
        fail_reason = ""
        
        figure_mode_types = ["show", "return"]
        if figure_mode not in figure_mode_types:
            raise ValueError(f"Invalid value figure_mode! Expected one of: {figure_mode_types}.")

        # First rule: the number of holes have to match
        if len(cnts) != len(self.__model_contours):
            final_result = False
            fail_reason = "The number of holes do not match"

        scale = img_ppmm / mdl_ppmm
        model_contours = self.__model_contours.copy()
        model_contours[0] = model_contours[0] * scale

        flipped_model_contours = self.flip_contours(self.__model_contours.copy())
        flipped_model_contours[0] = flipped_model_contours[0] * scale

        # Finding the initial guess for the ICP from PCA for the original contours
        # and flipped contours.
        cnt_trans, trans_matrix, center_model, sum_distances = self.find_initial_guess_transform(model_contours, cnts)
        cnt_flipped_trans, trans_matrix_flipped, center_flipped_model, sum_distances_flipped = self.find_initial_guess_transform(flipped_model_contours, cnts)
        if sum_distances_flipped < sum_distances:
            cnt_trans = cnt_flipped_trans
            model_contours = flipped_model_contours
            trans_matrix = trans_matrix_flipped
            center_model = center_flipped_model
        
        # Now we have the initial estimate for the transform and the ICP
        # can be applied
        second_trans_matrix, _ = icp(cnt_trans.reshape(-1, 2).T, cnts[0].reshape(-1, 2).T)
        to_object = cv2.transform(cnt_trans, second_trans_matrix[:2])
        to_object = to_object.reshape(-1, 2)

        # Finding the centers of each inner image contour to pair them up with
        # the inner model contours
        cnt_centers = np.zeros((len(cnts) - 1, 2))
        for i, cnt in enumerate(cnts[1:]):
            rect = cv2.minAreaRect(cnt)
            cnt_centers[i][0] = rect[0][0]
            cnt_centers[i][1] = rect[0][1]

        inds_and_dists = {}
        cnt_inds_and_mdl_inds = {0: 0}
        trans_model_contour_points = [to_object]
        
        for i, cnt in enumerate(model_contours[1:]):
            cnt = cnt * scale
            cnt_trans = cv2.transform(cnt - center_model, trans_matrix, -1) + center_model
            cnt_trans = cv2.transform(cnt_trans, second_trans_matrix[:2])
            trans_points = cnt_trans.reshape(-1, 2)
            center_p = np.mean(trans_points, axis=0)
            trans_model_contour_points.append(trans_points)
            
            # If there are inner contours (holes) in the imaged part
            if cnt_centers.size > 0:
                # Finding the correct inner object contour to compare the
                # current inner model contour with
                distances = np.sqrt(np.sum((cnt_centers - center_p)**2, axis=1))
                center_distance = np.min(distances)
                cnt_ind = np.argmin(distances) + 1 # Because the external contour is excluded

                # This check is done to pair the contours correctly in the situation where
                # there are more holes in the object than in the model.
                if cnt_ind not in inds_and_dists or center_distance < inds_and_dists[cnt_ind]:
                    # nearest_cnt = cnts[cnt_ind]
                    cnt_inds_and_mdl_inds[cnt_ind] = i + 1
                    inds_and_dists[cnt_ind] = center_distance

        # Now all the object contours and model contours should have been paired.
        # Going their through one by one and checking the distances.
        fail_point_inds_per_cnt = [[]] * len(cnts)
        for cnt_ind in cnt_inds_and_mdl_inds:
            
            mdl_ind = cnt_inds_and_mdl_inds[cnt_ind]

            distances, _ = NearestNeighbors(n_neighbors=1).fit(
                trans_model_contour_points[mdl_ind]).kneighbors(cnts[cnt_ind].reshape(-1, 2))
            is_over = distances.reshape(1, -1)[0] > max_dev
            fail_point_inds_per_cnt[cnt_ind], = np.where(is_over)
        
        # Checking, if there are extra holes in the part and adding them
        # to failing points
        all_cnt_indexes = list(range(len(cnts)))
        extra_cnt_indexes = list(set(all_cnt_indexes) -
                                set(cnt_inds_and_mdl_inds.keys()))
        for cnt_ind in extra_cnt_indexes:
            fail_point_inds_per_cnt[cnt_ind] = list(range(len(cnts[cnt_ind])))

        # Drawing the result
        model_color = "C1"
        object_color = "C0"
        fail_color = "r."
        fail_line = None
        fig = plt.figure(linewidth=10)
        if figure_mode == "return":
            canvas = FigureCanvasAgg(fig)
        plt.imshow(img, cmap="gray")
        for i, cnt in enumerate(cnts):
            img_line, = plt.plot(cnt.reshape(-1, 2)[:, 0], cnt.reshape(-1, 2)[:, 1], color=object_color)
            if fail_point_inds_per_cnt[i] != []:
                fail_points = cnt.reshape(-1, 2)[fail_point_inds_per_cnt[i], :]
                fail_line, = plt.plot(fail_points[:, 0], fail_points[:, 1], fail_color, markersize=5)
                if final_result:
                    final_result = False
                    fail_reason = "Too big a deviation"
        for to_object in trans_model_contour_points:
            mdl_line, = plt.plot(to_object[:, 0], to_object[:, 1], color=model_color)

        img_line.set_label("image")
        mdl_line.set_label("model")
        if fail_line:
            fail_line.set_label("failing point")
        plt.legend()
        plt.axis("equal")
        if final_result:
            title_obj = plt.title("PASS")
            plt.setp(title_obj, color="g")
            fig.set_edgecolor("g")
        else:
            title_obj = plt.title("FAIL")
            plt.setp(title_obj, color="r")
            fig.set_edgecolor("r")

        # Saving the figure, if desired
        if savename is not None:
            plt.savefig(savename)

        if figure_mode == "show":
            plt.show()
            result_image = None
        elif figure_mode == "return":
            # Retrieving a view on the renderer buffer
            canvas.draw()
            buf = canvas.buffer_rgba()

            # Converting to a NumPy array
            result_image = np.asarray(buf)

        return final_result, result_image