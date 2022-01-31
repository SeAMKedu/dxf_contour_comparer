import json
import tkinter.filedialog as fd
import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk
from pypylon import pylon
from dxf_to_contour import Dxf2ContourReader
from compare_contours import ContourComparer

class CheckGui():
    def __init__(self):
        # Defining the GUI
        self.root = tk.Tk()
        self.img_preview_shape = None # Will be read from the parameter file
        self.window_shape = None

        # The components for the analysis
        self.reader = Dxf2ContourReader()
        self.comparer = ContourComparer()
        self.img = None
        self.model_loaded = False
        self.calib_data = None
        self.adjust_params_window = None

        # Initializing the parameters to None (they will be loaded from
        # a separate file next)
        self.default_exp_time = None  # Default exposure time for the camera
        self.default_bright_thresh = None  # Default values for the thresholds
        self.default_dark_thresh = None
        self.default_mdl_ppmm = None    # Default value for model ppmm. This is 
                                        # used when the contour is drawn as a 
                                        # picture from the DXF file: the bigger 
                                        # the number the more pixels per curve
        self.exp_time = None
        self.bright_thresh = None
        self.dark_thresh = None
        self.mdl_ppmm = None
        self.img_ppmm = None  # To be loaded from the calibration file
        self.params_file = "parameters.json" 

        # Loading parameters and default parameters from the defined json file.
        # Loading also the GUI dimensions.
        self.load_parameters()
        
        # Creating the camera. If a Basler camera is found, it is used. Otherwise,
        # the webcam is used.
        try:
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.grabResult = None
            self.basler_camera_exists = True
        except:
            self.basler_camera_exists = False
            self.cap = None # Initalization for the possible webcam

        # Defining the GUI
        # The overall area
        self.left_frame = tk.Frame(self.root, width=self.window_shape[0]/2)
        self.right_frame = tk.Frame(self.root)

        # Text label for the loaded model
        self.model_text = tk.StringVar()
        self.model_text.set("No model loaded")
        self.model_label = tk.Label(self.left_frame, textvariable=self.model_text)

        # Text label for the loaded calibration
        self.calib_text = tk.StringVar()
        self.calib_text.set("No calibration loaded")
        self.calib_label = tk.Label(self.left_frame, textvariable=self.calib_text)

        # Button for loading the DXF file
        self.load_mdl_button = tk.Button(self.left_frame,
                                    text="Load Model",
                                    command=self.load_model)
        
        # Button for loading the calibration file
        self.load_cal_button = tk.Button(self.left_frame,
                                    text ="Load Calibration",
                                    command=self.load_calibration)
    
        # Button for running the check
        self.check_button = tk.Button(self.left_frame, 
                                    text="Run Check",
                                    command=self.run_check)

        # Button for adjusting the parameters
        self.adjust_params_button = tk.Button(self.left_frame, 
                                    text="Adjust Parameters",
                                    command=self.adjust_params)


        # Button for loading the image
        self.load_img_button = tk.Button(self.right_frame,
                                    text="Load Image",
                                    command=self.load_image)
        
        self.max_distance_label = tk.Label(self.left_frame,
                                        text="Max model-image deviation (mm)")
        self.max_distance_entry = tk.Entry(self.left_frame)
        self.max_distance_entry.insert(0, 1)
        
        img_src_label = tk.Label(self.right_frame, text="Image Source:")
        self.radio_var = tk.IntVar()
        self.radio_var.set(1)  # initializing the choice
        self.offline_radiobutton = tk.Radiobutton(self.right_frame, text="Offline images", 
                                            variable=self.radio_var, value=1, command=self.stream_off)
        self.camera_radiobutton = tk.Radiobutton(self.right_frame, text="Camera stream", 
                                            variable=self.radio_var, value=2, command=self.stream_on)
        
        color_label = tk.Label(self.right_frame, text="Object Color:")
        self.color_var = tk.IntVar()
        self.color_var.set(1)  # initializing the choice
        self.dark_radiobutton = tk.Radiobutton(self.right_frame, text="Dark on bright", 
                                            variable=self.color_var, value=1)
        self.bright_radiobutton = tk.Radiobutton(self.right_frame, text="Bright on dark", 
                                            variable=self.color_var, value=2)

        self.img_label = tk.Label(self.root)
        self.mdl_preview_label = tk.Label(self.root)

        self.left_frame.grid(row=0, column=0, padx=20, pady=20)
        self.right_frame.grid(row=0, column=1)
        self.root.grid_columnconfigure(0, minsize=self.window_shape[0] / 2)
        self.root.grid_columnconfigure(1, minsize=self.window_shape[0] / 2)
        self.load_mdl_button.pack()
        self.model_label.pack()
        self.load_cal_button.pack()
        self.calib_label.pack()
        self.max_distance_label.pack()
        self.max_distance_entry.pack()
        self.check_button.pack()
        self.adjust_params_button.pack()
        img_src_label.pack()
        self.offline_radiobutton.pack()
        self.camera_radiobutton.pack()
        self.load_img_button.pack()
        color_label.pack()
        self.dark_radiobutton.pack()
        self.bright_radiobutton.pack()

        self.mdl_preview_label.grid(row=1, column=0, padx=20, pady=20)
        self.img_label.grid(row=1, column=1)
        self.root.mainloop()

    def on_closing(self):
        """Function that runs when the window for adjusting the parameters 
        is closed. Activates the buttons and fields of the main widget and 
        closes the window.
        """
        self.load_mdl_button["state"] = "normal"
        self.load_cal_button["state"] = "normal"
        self.max_distance_entry["state"] = "normal"
        self.check_button["state"] = "normal"
        self.adjust_params_button["state"] = "normal"
        self.offline_radiobutton["state"] = "normal"
        self.camera_radiobutton["state"] = "normal"
        self.load_img_button["state"] = "normal"
        self.dark_radiobutton["state"] = "normal"
        self.bright_radiobutton["state"] = "normal"
        
        self.adjust_params_window.destroy()
    
    @staticmethod
    def check_param(param, is_thresh=False):
        """Checks that a parameter is in a correct format

        Args:
            param (string): The parameter from the adjust_params window
            is_thresh (bool, optional): True if the parameter is a threshold parameter. 
                                        Defaults to False.

        Returns:
            bool, int: True if parameter was given correctly, parameter as 
                       int (or None if it was not given correctly)
        """

        param_ok = True
        try:
            param = int(param)
        except ValueError:
            param_ok = False
            param = None
        else:
            if param < 0:
                param_ok = False
            elif is_thresh and param > 255:
                param_ok = False

        return param_ok, param

    def update_params(self):
        """Checks the entries of adjust_params_window and updates the 
        parameters.
        """
        erroneous_params_exist = False
        err_msg = ""

        bright_thresh = self.bright_thresh_entry.get()
        ok, bright_thresh = self.check_param(bright_thresh, True)
        if not ok:
            erroneous_params_exist = True
            err_msg += "Bright threshold should be an integer 0...255\n"

        dark_thresh = self.dark_thresh_entry.get()
        ok, dark_thresh = self.check_param(dark_thresh, True)
        if not ok:
            erroneous_params_exist = True
            err_msg += "Dark threshold should be an integer 0...255\n"
        
        mdl_ppmm = self.mdl_ppmm_entry.get()
        ok, mdl_ppmm = self.check_param(mdl_ppmm)
        if not ok:
            erroneous_params_exist = True
            err_msg += "Model ppmm threshold should be an integer greater than zero\n"

        exp_time = self.exp_time_entry.get()
        ok, exp_time = self.check_param(exp_time)
        if not ok:
            erroneous_params_exist = True
            err_msg += "Exposure time should be an integer greater than zero."
        
        if erroneous_params_exist:
            tk.messagebox.showerror("Error", err_msg)
        else:
            # Update the parameters
            self.bright_thresh = bright_thresh
            self.dark_thresh = dark_thresh
            self.mdl_ppmm = mdl_ppmm
            self.exp_time = exp_time

            # If the camera is currently viewing image, let's update
            # its exposure time on the fly
            if self.basler_camera_exists and self.camera.IsGrabbing():
                self.camera.ExposureTimeAbs.SetValue(self.exp_time)

            # Write the new parameters to the parameter file
            with open(self.params_file) as json_file:
                param_data = json.load(json_file)
                json_file.close()

                param_data["bright_thresh"] = bright_thresh
                param_data["dark_thresh"] = dark_thresh
                param_data["mdl_ppmm"] = mdl_ppmm
                param_data["exp_time"] = exp_time

                json_data = json.dumps(param_data, indent=4)
            
            with open(self.params_file, "w") as outfile:
                outfile.write(json_data)
                outfile.close()

            # Closing the window after the parameter update    
            self.on_closing()

    def restore_default_params(self):
        """Setting the parameters back to the defaults
        """
        self.exp_time_entry.delete(0, "end")
        self.exp_time_entry.insert(0, self.default_exp_time)
        self.bright_thresh_entry.delete(0, "end")
        self.bright_thresh_entry.insert(0, self.default_bright_thresh)
        self.dark_thresh_entry.delete(0, "end")
        self.dark_thresh_entry.insert(0, self.default_dark_thresh)
        self.mdl_ppmm_entry.delete(0, "end")
        self.mdl_ppmm_entry.insert(0, self.default_mdl_ppmm)  

    def adjust_params(self):
        """Creating the window for adjusting parameters
        """
        # Disabling the buttons of the main widget
        self.load_mdl_button["state"] = "disabled"
        self.load_cal_button["state"] = "disabled"
        self.max_distance_entry["state"] = "disabled"
        self.check_button["state"] = "disabled"
        self.adjust_params_button["state"] = "disabled"
        self.offline_radiobutton["state"] = "disabled"
        self.camera_radiobutton["state"] = "disabled"
        self.load_img_button["state"] = "disabled"
        self.dark_radiobutton["state"] = "disabled"
        self.bright_radiobutton["state"] = "disabled"
        
        # Creating the window
        self.adjust_params_window = tk.Toplevel(self.root)
        self.adjust_params_window.title("Adjust the Parameters")
        self.adjust_params_window.geometry("450x220")
        self.adjust_params_window.attributes("-topmost", True)
    
        # Field for changing the bright threshold
        tk.Label(self.adjust_params_window,
            text ="Threshold for bright objects on dark background").pack()
        self.bright_thresh_entry = tk.Entry(self.adjust_params_window)
        self.bright_thresh_entry.insert(0, self.bright_thresh)
        self.bright_thresh_entry.pack()
        
        # Field for changing the dark threshold
        tk.Label(self.adjust_params_window,
            text ="Threshold for dark objects on bright background").pack()
        self.dark_thresh_entry = tk.Entry(self.adjust_params_window)
        self.dark_thresh_entry.insert(0, self.dark_thresh)
        self.dark_thresh_entry.pack()

        # Field for changing the model ppmm
        tk.Label(self.adjust_params_window,
            text ="PPMM used when generating the contour model from the DXF file").pack()
        self.mdl_ppmm_entry = tk.Entry(self.adjust_params_window)
        self.mdl_ppmm_entry.insert(0, self.mdl_ppmm)
        self.mdl_ppmm_entry.pack()

        # Field for changing the exposure time
        tk.Label(self.adjust_params_window,
            text ="Exposure time for the camera (us)").pack()
        self.exp_time_entry = tk.Entry(self.adjust_params_window)
        self.exp_time_entry.insert(0, self.exp_time)
        self.exp_time_entry.pack()

        tk.Button(self.adjust_params_window, text="Update Parameters",
                        command=self.update_params).pack()
        
        tk.Button(self.adjust_params_window, text="Restore Defaults",
                        command=self.restore_default_params).pack()

        self.adjust_params_window.protocol("WM_DELETE_WINDOW", self.on_closing)

        if not self.basler_camera_exists:
            self.exp_time_entry["state"] = "disabled"


    def stream_on(self):
        """Opening a camera device when radio button is clicked
        """
        self.load_img_button["state"] = "disabled"

        # Using a Basler camera
        if self.basler_camera_exists:
            # Opening the camera object and setting the exposure time
            self.camera.Open()
            self.camera.ExposureTimeAbs.SetValue(self.exp_time)  # given in us 

            # Checking image size
            new_width = self.camera.Width.GetValue() - self.camera.Width.GetInc()
            if new_width >= self.camera.Width.GetMin():
                self.camera.Width.SetValue(new_width)

            # Starting grabbing images
            self.camera.StartGrabbing()
            self.show_frame()

        # Using a webcam
        else:
            # Using the webcam (mostly for testing purposes)
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if cap is None or not cap.isOpened():
                self.cap = None

                tk.messagebox.showerror("Error", "No Basler camera or webcam found!")
                self.load_img_button["state"] = "normal"
            else:
                self.cap = cap
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_preview_shape[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_preview_shape[1])
                self.show_frame()
        

    def stream_off(self):
        """Closing the camera device when the radio button is clicked.
        """
        if self.basler_camera_exists:
            self.grabResult.Release()
            self.camera.Close()
        else:
            self.cap.release()
        self.load_img_button["state"] = "normal"
    
    @staticmethod
    def find_and_filter_contours(bimg):
        """Checks how many blobs there are in the image. If there are
        multiple blobs, only the biggest is left. This is to remove
        possible trash and the frames that are left after undistorting and 
        warping the image from the image. If there are bigger blobs
        (20 % of the area of the biggest), an exception is raised.

        Args:
            bimg (numpy array): Binary image

        Raises:
            RuntimeError: If there are none or multiple objects in the image

        Returns:
            list, list: Contours of the biggest blob, hierarchy
        """

        n_blobs, labels, stats, _ = cv2.connectedComponentsWithStats(bimg)
        if n_blobs > 2: # Because background is counted as one blob
            max_area = np.max(stats[1:, -1])
            if np.sum(stats[1:, -1] > 0.2 * max_area) > 2:
                raise RuntimeError("There seems to be multiple objects in the image. " +
                                    "This program is for checking a single part only.")

            new_bimg = np.zeros_like(bimg)
            max_idx = np.where(stats[1:, -1] == max_area)[0][0] + 1
            new_bimg[labels == max_idx] = 255
            bimg = new_bimg

        cnts, hier = cv2.findContours(bimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return cnts, hier

    def run_check(self):
        """Callback for the button Check. Compares the loaded image to
        the loaded model.
        """
        max_dist_in_mm = self.max_distance_entry.get()
        max_dist_erroneous = False
        try:
            max_dist_in_mm = float(max_dist_in_mm)
        except ValueError:
            max_dist_erroneous = True
        if self.img is None:
            tk.messagebox.showerror("Error", "Load the image first!")
        elif not self.model_loaded:
            tk.messagebox.showerror("Error", "Load the model first!")
        elif self.calib_data is None:
            tk.messagebox.showerror("Error", "Load the calibration data first!")
        elif max_dist_erroneous:
            tk.messagebox.showerror("Error", "Max distance should be numeric!")
        else:
            if self.color_var.get() == 1:
                thresh = self.dark_thresh
                thresh_mode = cv2.THRESH_BINARY_INV
            else:
                thresh = self.bright_thresh
                thresh_mode = cv2.THRESH_BINARY
                
            _, test_bw = cv2.threshold(self.img, thresh, 255, thresh_mode)

            # Checking if the image is empty (has to be done before undistorting,
            # which adds some black borders) 
            contours, _ = cv2.findContours(test_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if (len(contours) == 0 or 
                cv2.contourArea(max(contours, key=cv2.contourArea)) < 400):
                tk.messagebox.showerror("Error", "No objects detected in the image.")
            else:
                # Undistorting and correcting the perspective
                img_ud = cv2.undistort(self.img, self.calib_data["camera_matrix"], 
                                        self.calib_data["distortion_coeffs"], None, 
                                        self.calib_data["new_camera_matrix"])
                img_ptrans = cv2.warpPerspective(img_ud, 
                                                self.calib_data["perspective_transformation"],
                                                (self.img.shape[1], self.img.shape[0]))
                _, bw = cv2.threshold(img_ptrans, thresh, 255, thresh_mode)

                # Finding contours from the scaled and rotated binary image
                try:
                    contours, _ = self.find_and_filter_contours(bw)
                except RuntimeError as re:
                    tk.messagebox.showerror("Error", str(re))
                else:
                    # Doing the comparison
                    self.comparer.match_contour_to_model(contours, max_dist_in_mm * self.img_ppmm, 
                                                        self.img_ppmm, self.mdl_ppmm, 
                                                        img_ptrans)


    def load_model(self):
        """Callback for the Load model button. Loading DXF file to the model
        """
        filename = fd.askopenfilename(filetypes=[("DXF files", "*.dxf")])

        if filename != "":
            # Update the model name in the GUI
            self.model_text.set(f"Model: {filename}")
            self.model_loaded = True

            self.reader.read_file(filename, self.mdl_ppmm)
            model_contours, _ = self.reader.get_contours()
            self.comparer.set_model_contours(model_contours)

            # Getting the preview image
            model_image = self.reader.get_contour_image()
            render = self.render_image(model_image, self.img_preview_shape[0]/2)
            self.mdl_preview_label.configure(image=render)
            self.mdl_preview_label.image = render
    
    def load_calibration(self):
        """Call back for Load calibration file button. Loading a json file for calibration
        """
        filename = fd.askopenfilename(filetypes=[("json files", "*.json")])
        file_ok = True

        if filename != "":
            with open(filename) as json_file:
                json_data = json.load(json_file)
                json_file.close()

            cal_data = {}
            try:
                cal_data["camera_matrix"] = np.asarray(json_data["camera_matrix"])
                cal_data["distortion_coeffs"] = np.asarray(json_data["distortion_coeffs"])
                cal_data["new_camera_matrix"] = np.asarray(json_data["new_camera_matrix"])
                cal_data["perspective_transformation"] = np.asarray(json_data["perspective_transformation"])
                ppmm = json_data["ppmm"]
            except:
                file_ok = False
            else:
                if (cal_data["camera_matrix"].shape != (3, 3) or
                    cal_data["distortion_coeffs"].shape != (1, 5) or
                    cal_data["new_camera_matrix"].shape != (3, 3) or
                    cal_data["perspective_transformation"].shape != (3, 3) or
                    not(isinstance(ppmm, (int, float)))):
                    file_ok = False
            
            if file_ok:
                # Update the model name in the GUI
                self.calib_text.set(f"Calibration: {filename}")
                self.calib_data = cal_data
                self.img_ppmm = ppmm
            else:
                tk.messagebox.showerror("Error", f"Calibration file {filename} is in incorrect format!")
    
    def load_parameters(self):
        """Reads the parameters from the parameter json file and checks that
        the file is in the expected format and the parameter in the expected 
        ranges.
        """
        file_ok = True
        err_msg = None

        with open(self.params_file) as json_file:
            json_data = json.load(json_file)
            json_file.close()
        
        try:
            default_bright_thresh = json_data["default_bright_thresh"]
            default_dark_thresh = json_data["default_dark_thresh"]
            default_mdl_ppmm = json_data["default_mdl_ppmm"]
            default_exp_time = json_data["default_exp_time"]
            bright_thresh = json_data["bright_thresh"]
            dark_thresh = json_data["dark_thresh"]
            mdl_ppmm = json_data["mdl_ppmm"]
            exp_time = json_data["exp_time"]
            gui_height = json_data["gui_height"]
            gui_width = json_data["gui_width"]
        except:
            file_ok = False
            err_msg = f"The parameter file {self.params_file} is in wrong format!"
        else:
            params_as_list = [default_bright_thresh, default_dark_thresh,
                                default_exp_time, bright_thresh, dark_thresh, 
                                default_mdl_ppmm, mdl_ppmm, exp_time, gui_height,
                                gui_width]
            if not all(isinstance(i, int) for i in params_as_list):
                file_ok = False
                err_msg = f"All parameters in {self.params_file} should be integers"
            elif not all(i > 0 for i in params_as_list):
                file_ok = False
                f"All parameters in {self.params_file} should be greater than zero"
            elif not all(i <= 255 for i in params_as_list[:5]):
                f"All threshold values in {self.params_file} should be less or equal to 255"
                file_ok = False
        
        if file_ok:
            # Setting the GUI window size
            prev_width = int(gui_width / 2) - 160  # Image preview width. Bit less than half of the window. 
            self.window_shape = (gui_width, gui_height)
            self.img_preview_shape = (prev_width, int(round(3/4*prev_width)))
            self.root.geometry(f"{gui_width}x{gui_height}")

            # Setting the analysis parameters
            self.default_bright_thresh = default_bright_thresh
            self.default_dark_thresh = default_dark_thresh
            self.default_exp_time = default_exp_time
            self.default_mdl_ppmm = default_mdl_ppmm
            self.bright_thresh = bright_thresh
            self.dark_thresh = dark_thresh
            self.mdl_ppmm = mdl_ppmm
            self.exp_time = exp_time
        else:
            tk.messagebox.showerror("Error", err_msg)
            self.root.destroy()

    
    def render_image(self, image=None, max_dim=None):
        """Rendering the image so that tkinter can show it.

        Args:
            image (numpy array, optional): Image in OpenCV type. If not given,
                                            self.img will be used. Defaults to None.
            max_dim (int, optional): Maximum dimension of the preview image. If not given, 
                                            self.img_previw_width will be used. Defaults to None.

        Returns:
            tk image: Image in tkinter compatible format
        """
        if image is None:
            image = self.img
        if max_dim is None:
            max_dim = self.img_preview_shape[0]
        h, w = image.shape
        if h > max_dim or w > max_dim:
            fx = max_dim / w
            fy = max_dim / h
            scale = min(fx, fy)
            preview_img = cv2.resize(image, None, fx=scale, fy=scale)
        else:
            preview_img = image
        pil_img = Image.fromarray(preview_img)
        render = ImageTk.PhotoImage(pil_img)
        return render

    def load_image(self):
        """Callback for the Load image button.
        """
        filename = fd.askopenfilename(filetypes=[("Png files", "*.png"),
                                                ("Jpg files", "*.jpg"), 
                                                ("Tiff files", "*.tiff"), 
                                                ("Tif files", "*.tif")])
        if filename != "":
            img = cv2.imread(filename, 0)
            self.img = img
            render = self.render_image()
            self.img_label.configure(image=render)
            self.img_label.image = render

    def show_frame(self):
        """Showing the frame grabbed from the camera device in the GUI.
        """
        # Initializing
        frame = np.zeros(self.img_preview_shape, np.uint8)

        # If the radio button is selected, continuing capturing frames
        if self.radio_var.get() == 2:
            
            # Using the Basler camera
            if self.basler_camera_exists:

                # Reading images one by one as long as the camera is grabbing them
                if self.camera.IsGrabbing():
                    grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                    self.grabResult = grabResult

                    if grabResult.GrabSucceeded():
                        # Reading the image data, scaling it to fit the window in 
                        # the GUI and showing it.
                        img = grabResult.Array
                        frame = cv2.resize(img, self.img_preview_shape)
                        self.img = img
                    
                    else:
                        if self.img is not None:
                            frame = cv2.resize(self.img, self.img_preview_shape) # Using the previous image

            else:
                # Using the webcam
                _, frame = self.cap.read()
                frame = cv2.flip(frame, 1)
                self.img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Modifying the image to be tkinter compatible.
            pil_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            pil_img = Image.fromarray(pil_img)
            render = ImageTk.PhotoImage(image=pil_img)

            self.img_label.configure(image=render)
            self.img_label.image = render
            self.img_label.after(10, self.show_frame)
        
if __name__ == "__main__":
    app = CheckGui()
