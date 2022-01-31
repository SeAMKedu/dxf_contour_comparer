import ezdxf as ez
import numpy as np
import cv2

class Dxf2ContourReader():
    """A class to read DXF files (made with Siemens CAD tools) and transform
    them to OpenCV contour representation. The module ezdxf is utilized
    in reading the DXF files. Basic information about the DXF files in general
    can be found at https://documentation.help/AutoCAD-DXF/
    and information about the entity representation of ezdxf can be found at
    https://ezdxf.mozman.at/docs/tutorials/getting_data.html#entity-queries
    
    Example usage:
    
        reader = Dxf2ContourReader()
        reader.read_file(filename, spat_reso)
        cnt, hier = reader.get_contours()
        reader.draw_contours()

    The argument 'spat_reso' gives the spatial resolution. When reading the
    DXF file, the measures are in metric system, in the contours generated, 
    they are in pixels. So dimension_in_pix = spat_reso * dimension
    The method 'get_contours' returns the contours
    The method 'draw_contours' draws the contours

    
    Raises:
        ValueError: if the filename given to the read_file method raises
                    an exception (file does not exist or is not a valid dxf)
    """

    def __init__(self):
        self.contours = None
        self.hierarchy = None
        self.img_size = None

    @staticmethod
    def _find_ul_and_lr(points):
        """Finding the upper left (ul) and lower right (lr) point of the
        bounding box for a list of 2D points.

        The DXF file should have parameters $EXTMIN and $EXTMAX that define the
        bounding box of the drawing. This is needed when defining the size of
        the drawing for the contours. Unfortunately, the parameters are usually missing.
        Thus, we are finding the bounding box manually with this function.,
        
        Arguments:
            points {list of 2-length lists or nx2 numpy array} -- the 2D points [[x1, y1,], [x2, y2], ..., [xn, yn]]
        
        Returns:
            list -- the ul and lr points [[ulx, uly], [lrx, lry]]
        """

        # TUTKI MAHDOLLISUUTTA KÄYTTÄÄ CONTOURIEN BOUNDIX BOX -FUNKKARIA TÄHÄN. 
        # TÄLLÖIN LUULTAVASTI KANNATTAISI SYÖTTÄÄ VAIN YKSI RIPSU 2D-PISTEITÄ
        # TÄNNE EIKÄ SELVITTÄÄ UL- JA LR-PISTEITÄ JOKA MUODOLLE ERIKSEEN ENSIN.

        # Initializing the extreme points
        ul = [1e4, 1e4]  
        lr = [-1e4, -1e4]

        # Checking all xy points one by one
        for p in points:
            if p[0] < ul[0]:
                ul[0] = p[0]
            if p[0] > lr[0]:
                lr[0] = p[0]
            if p[1] < ul[1]:
                ul[1] = p[1]
            if p[1] > lr[1]:
                lr[1] = p[1]
        
        return ul, lr

    def _find_arc_bbox(self, start_angle, end_angle, arc_center, arc_r):
        """Find the xy coordinates of the upper left and the lower 
        right corner of the bounding box for an arc
        
        Arguments:
            start_angle {float} -- start angle of the arc in degrees
            end_angle {float} -- end angle of the arc in degrees
            arc_center {2-length list of floats} -- the center coordinates [x, y]
            arc_r {float} -- arc radius
        
        Returns:
            list -- the ul and lr points [[ulx, uly], [lrx, lry]]
        """
        # First, adding the start and end point of the arc to the list of 
        # the arc extreme points (the leftmost, the rightmost, the top and 
        # the bottom)
        start_angle_in_rad = start_angle / 180.0 * np.pi
        end_angle_in_rad = end_angle / 180.0 * np.pi

        p1 = [arc_center[0] + arc_r * np.cos(start_angle_in_rad), 
            arc_center[1] + arc_r * np.sin(start_angle_in_rad)]
        p2 = [arc_center[0] + arc_r * np.cos(end_angle_in_rad), 
            arc_center[1] + arc_r * np.sin(end_angle_in_rad)]
        arc_extremes = [p1, p2]

        # Then, checking, which extreme points of the (potential) circle the
        # arc is passing by and adding them to the list
        #  3 o'clock:  0 degrees
        # 12 o'clock: 270 degrees
        #  9 o'clock: 180 degrees
        #  6 o'clock: 90 degrees
        # y axis points down (image coordinates)
        if start_angle < end_angle:
            if start_angle <= 90 and end_angle > 90:
                arc_extremes.append([arc_center[0], arc_center[1] + arc_r])
            if start_angle <= 180 and end_angle > 180:
                arc_extremes.append([arc_center[0] - arc_r, arc_center[1]])
            if start_angle <= 270 and end_angle > 270:
                arc_extremes.append([arc_center[0], arc_center[1] - arc_r])
        else:
            # Because the arc is always dwawn via the 0 degree angle
            arc_extremes.append([arc_center[0] + arc_r, arc_center[1]])
            if start_angle >= 270 and end_angle > 180 and start_angle - end_angle < 180:
                arc_extremes.append([arc_center[0], arc_center[1] + arc_r])
            if start_angle >= 180 and end_angle > 90 and start_angle - end_angle < 180:
                arc_extremes.append([arc_center[0] - arc_r, arc_center[1]])
            if start_angle >= 90 and end_angle < 90 and start_angle - end_angle < 180:
                arc_extremes.append([arc_center[0], arc_center[1] - arc_r])


        # Finally, finding the bounding box corners (upper left 
        # and lower right) for the list of the extreme points and
        # returning them
        return self._find_ul_and_lr(arc_extremes)

    def _get_ul_and_dr(self, ents):
        """Finds upper left and lower right xy points from the
        list of DXF entities given by the library ezdxf.
        
        Arguments:
            ents {list} -- the list of entities from ezdxf
        
        Returns:
            list -- the ul and dr points [[ulx, uly], [drx, dry]]
        """

        # Initialization (all the entities have their 
        # own ul and br points)
        points = 2 * len(ents) * [[]] 
        
        # Collecting the extreme points of the entities to one list
        for i, e in enumerate(ents):
            entity_type = e.dxftype()
            if entity_type == "LINE":
                points[2*i] = [e.dxf.start[0], e.dxf.start[1]] 
                points[2*i + 1] = [e.dxf.end[0], e.dxf.end[1]]
            elif entity_type == "CIRCLE":
                points[2*i] = [e.dxf.center[0] - e.dxf.radius, 
                            e.dxf.center[1] - e.dxf.radius]
                points[2*i + 1] = [e.dxf.center[0] + e.dxf.radius, 
                                e.dxf.center[1] + e.dxf.radius]
            elif entity_type == "ARC":
                arc_ul, arc_lr = self._find_arc_bbox(e.dxf.start_angle, 
                                                    e.dxf.end_angle, 
                                                    [e.dxf.center[0], 
                                                    e.dxf.center[1]], 
                                                    e.dxf.radius)
                points[2*i] = arc_ul
                points[2*i + 1] = arc_lr

            elif entity_type == "SPLINE":
                # First, finding the extreme points of the spline
                pts = self._find_ul_and_lr(np.array(e.control_points))

                # Then, adding these points to the list
                points[2*i] = pts[0]
                points[2*i + 1] = pts[1]
        
        return self._find_ul_and_lr(points)

    @staticmethod
    def find_endpoints(bw):
        """Finding end points from a binary skeleton.

        Args:
            bw (numpy array): Binary skeletonized image

        Returns:
            n x 2 numpy array: The end point coordinates [[x1, y1], [x2, y2], ...]
        """
        # Defining structuring elements for finding end points
        # (ref. Gonzales, Woods: Digital Image Processing,
        # 3rd edition, Pearson Prentice Hall, 2008, p. 655)
        res_bw = np.zeros_like(bw)
        strels = [np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, -1]]), 
                    np.array([[-1, -1, 1], [-1, 1, -1], [-1, -1, -1]]),
                    np.array([[-1, -1, -1], [-1, 1, -1], [-1, -1, 1]]),
                    np.array([[-1, -1, -1], [-1, 1, -1], [1, -1, -1]]),
                    np.array([[0, -1, -1], [1, 1, -1], [0, -1, -1]]),
                    np.array([[0, 1, 0], [-1, 1, -1], [-1, -1, -1]]),
                    np.array([[-1, -1, 0], [-1, 1, 1], [-1, -1, 0]]),
                    np.array([[-1, -1, -1], [-1, 1, -1], [0, 1, 0]])]
        for strel in strels:
            tmp = cv2.morphologyEx(bw, cv2.MORPH_HITMISS, strel)
            res_bw += tmp
        
        end_ys, end_xs = np.where(res_bw == 255)
        end_points = np.vstack((end_xs, end_ys)).T

        return end_points

    def check_gaps(self, bw, spat_reso):
        """Finding the gaps in the drawing.

        Args:
            bw (Numpy array): Binary image
            spat_reso (int): Spatial resolution

        Returns:
            list of 2-length-lists of 2-length-lists: The x, y point pairs defining the gaps
        """
        end_ps = self.find_endpoints(bw)
        pairs = []
        if end_ps.shape[0] > 1:
            # Contininuing as long as there are non-paired points
            # (if for some reason there are erroneous spurs far away
            # from each other, they will not be paired but ditched)
            while end_ps.shape[0] > 1:
                prev_len = end_ps.shape[0]
                dists = end_ps[1:] - end_ps[0]
                dists = np.sqrt(dists[:, 0]**2 + dists[:, 1]**2)
                if min(dists) < 2 * spat_reso: # Minimum distance -> the pair
                    i = np.argmin(dists) + 1
                    pairs.append(([end_ps[0], end_ps[i]]))
                    end_ps = np.delete(end_ps, [0, i], axis=0)
                else:
                    end_ps = end_ps[1:]
        return pairs


    def read_file(self, filename, spat_reso):
        """Reading the DXF file and constructing an OpenCV 
        image from it.

        Args:
            filename (string): Path of the DXF file
            spat_reso (int): Spatial resolution for constructing the image

        Raises:
            ValueError: If the path given for the DXF file is erroneous
        """
        
        # Reading the DXF file and extracting the model space.
        try:
             doc = ez.readfile(filename)
        except:
            raise ValueError("Error! " + filename + " not valid path for a DXF file!" )
        msp = doc.modelspace()

        # The layer OUTER_LOOP includes all the outer edges, the layer 
        # INTERIOR_LOOPS holes. This is the standard of SolidEdge 
        # (and maybe MCD?)
        layers = ["OUTER_LOOP", "INTERIOR_LOOPS"]

        # Going through the layers and finding the entities on each of them.
        for layer in layers:

            # Entity types checked: lines, circles, arcs and splines
            entities = msp.query('LINE CIRCLE ARC SPLINE[layer=="%s"]' % layer)
            
            # Finding the bounding box of the object based on the outer
            # edges. (This information should be stored to the DXF but it 
            # seems it is not.) The bounding box information is needed for
            # 1) Translating all the points to the positive xy plane (image
            #    coordinates cannot be negative)
            # 2) Initializing the image the object in drawn to find the 
            #    contours 
            if layer == "OUTER_LOOP":
                # +2: small margin
                ul, lr = self._get_ul_and_dr(entities)
                img_size = (int(np.ceil((lr[1] - ul[1]) * spat_reso)) + 2, 
                            int(np.ceil((lr[0] - ul[0]) * spat_reso)) + 2)
                img = np.zeros(img_size, np.uint8)
                self.img_size = img_size

            # Drawing the entities. Always translating the points first to the
            # positive xy plane Then, scaling according to the defined spatial
            #  resolution.
            for e in entities:
                dxf_type = e.dxftype()
                if dxf_type == "LINE":
                    start_p = ( int(round((e.dxf.start[0] - ul[0]) * spat_reso)), 
                                int(round((e.dxf.start[1] - ul[1]) * spat_reso)) ) 
                    end_p = ( int(round((e.dxf.end[0] - ul[0]) * spat_reso)), 
                            int(round((e.dxf.end[1] - ul[1]) * spat_reso)) ) 

                    cv2.line(img, start_p, end_p, 255)

                elif dxf_type == "CIRCLE" or dxf_type == "ARC":
                    center_p = ( int(round((e.dxf.center[0] - ul[0]) * spat_reso)),
                                int(round((e.dxf.center[1] - ul[1]) * spat_reso)) )
                    r = int(round(e.dxf.radius * spat_reso))

                    if dxf_type == "CIRCLE":
                        cv2.circle(img, center_p, r, 255)
                    else:
                        # OpenCV shifts the start and end point always so that 
                        # start_p < end_p. Therefore, checking that in the beginning to draw the
                        # correct "piece" of the circle.                        
                        if e.dxf.start_angle > e.dxf.end_angle:
                            cv2.ellipse(img, center_p, (r, r), 0, e.dxf.start_angle, 360, 255)
                            cv2.ellipse(img, center_p, (r, r), 0, 0, e.dxf.end_angle, 255)
                        else:
                            cv2.ellipse(img, center_p, (r, r), 0, e.dxf.start_angle, e.dxf.end_angle, 255)
                
                elif dxf_type == "SPLINE":
                    pts = np.array(e.control_points)
                    pts = np.int32(np.round((pts[:, :2] - ul) * spat_reso))
                    cv2.polylines(img, [pts], e.closed, 255)

        # Sometimes some gaps remain in the contour. Maybe this is because
        # of rounding errors? Or some difference between the OpenCV drawing 
        # commands and CAD program drawing commands? Patching the gaps if
        # they exist.
        pairs = self.check_gaps(img, spat_reso)
        if pairs != []:
            for pair in pairs:
                cv2.line(img, tuple(pair[0]), tuple(pair[1]), 255)

        # Flood filling.
        # Finding first the seedpoint based on the upper leftmost point of the form,
        # then creating the mask and flood filling.
        img_coords = np.where(img == 255)
        min_distance_ind = np.argmin(np.sqrt(img_coords[0]**2 + img_coords[1]**2))
        seed_point = ( int(img_coords[1][min_distance_ind] + 1), 
                    int(img_coords[0][min_distance_ind]) + 1)
        mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)
        _, img, _, _ = cv2.floodFill(img, mask, seed_point, 255)
        
        # Sometimes there might be a one-pixel-sized erroneous contour
        # because of some problem with the DXF files (exact reason not clear
        # yet (2.9.2020), hopefully will be understood in the future)
        # This fixes the problem by clearing smaller objects.
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img)
        if n_labels > 2:  # Background and one object
            max_label = np.argmax(stats[:, 4])
            labels[labels != max_label] = 0
            labels[labels == max_label] = 255
            img = np.uint8(labels)

        #img = cv2.flip(img, 0) # Flipping around the y axis (image coordinate origin is in the tl corner)

        # Finding the contours
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        self.contours = contours
        self.hierarchy = hierarchy
    
    def draw_contours(self, thickness=2):
        """Drawing the contours on black background.

        Args:
            thickness (int, optional): Line thickness. Defaults to 2.
        """
        if self.contours is None:
            print("No image to show. Read the DXF file first.")
        else:
            # Initializing the black background
            img = np.zeros((self.img_size[0], self.img_size[1]), np.uint8) * 255
            cv2.drawContours(img, self.contours, -1, 255, thickness)
            # for i, cnt in enumerate(self.contours):
            #     # If the contour has no parents, it is the outer edge
            #     if self.hierarchy[0][i][3] == -1:
            #         cv2.drawContours(img, [cnt], -1, 255, -1)
            #     else:
            #         cv2.drawContours(img, [cnt], -1, 0, -1)
            win_name = "Drawing"
            cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(win_name, img)
            cv2.waitKey(0)
    
    def get_contour_image(self, thickness=2):
        """Drawing the contours on black background
        and returning the image. 
                
        Args:
            thickness (int, optional): Line thickness. Defaults to 2.
        """
        if self.contours is None:
            print("No image to draw. Read the DXF file first.")
        else:
            # Initializing the black background
            img = np.zeros((self.img_size[0], self.img_size[1]), np.uint8) * 255
            cv2.drawContours(img, self.contours, -1, 255, thickness)
            return img

    def get_contours(self):
        """Returns the read contours and their hierarchy.

        Returns:
            tuple -- (contours as a list of numpy arrays, hierarchy as a list)
        """
        if self.contours is None:
            print("No contours to return. Read the DXF file first.")
        return (self.contours, self.hierarchy)



if __name__ == "__main__":
    #filename = r"C:\Users\k5000582\Documents\Hankkeet\EDIT\DXF-contour\150x100xD30.dxf"
    #filename = r"C:\Users\k5000582\Documents\Hankkeet\EDIT\DXF-contour\SupportPart_v1.dxf"
    # filename = r"C:\Users\k5000582\Documents\Hankkeet\EDIT\DXF-contour\SupportPart_v2.dxf"
    filename = r"C:\Users\k5000582\Documents\Hankkeet\EDIT\DXF-contour\Oikeat_kuvat\004\004.dxf"
    spat_reso = 5.0

    reader = Dxf2ContourReader()
    reader.read_file(filename, spat_reso)
    reader.draw_contours()