import tkinter as tk
import tkinter.filedialog as fd
import tkinter.messagebox as mb
import cv2
import numpy as np
from PIL import Image, ImageTk
from pypylon import pylon
from calibrate_image import calibrate_main

class CalGui():
    """The GUI for the camera calibration. Uses the calibration tools of OpenCV.
    A single image of a chessboard can be used or such can be grabbed from the camera
    connected to the computer. Currently, Basler cameras and integrated webcams (LOL)
    are supported. This is not for calibrations for 3D reconstruction but only for planar
    calibration, where the mission is to get rid of lens distortion, fix the perspective
    (remove the small tilt) and get the spatial resolution (pixels per millimeters).
    The calibration results can be saved in JSON. The DXF based part check tool uses
    calibration data given in this particular format.
    """
    def __init__(self):
        """Constructing the GUI
        """
        self.root = tk.Tk()
        gui_size = (1200, 800)
        self.root.geometry(f"{gui_size[0]}x{gui_size[1]}")
        self.img = None
        prev_width = int(gui_size[0] * 0.5)
        self.img_preview_shape = (prev_width, int(round(3/4*prev_width)))
        self.exp_time = 1000 # in us

        # Basler camera
        try:
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.grabResult = None
            self.basler_camera_exists = True
        except:
            self.basler_camera_exists = False
            self.cap = None # Initalization for the possible webcam
        
        # The overall area
        self.left_frame = tk.Frame(self.root, width=int(gui_size[0]/4))
        self.right_frame = tk.Frame(self.root)
    
        # Button for running the check
        self.calib_button = tk.Button(self.left_frame, 
                                    text="Calibrate",
                                    command=self.calibrate)

        # Button for loading the image
        self.load_img_button = tk.Button(self.left_frame,
                                    text="Load Image",
                                    command=self.load_image)

        # Exposure time
        self.exp_time_label = tk.Label(self.left_frame,
                                    text="Exposure time (im um): ")
        self.exp_time_entry = tk.Entry(self.left_frame)
        self.exp_time_entry.insert(0, self.exp_time)

        # Button for loading the image
        self.update_exp_time_button = tk.Button(self.left_frame,
                                        text="Update Exposure Time",
                                        command=self.update_exp_time)

        # Disable changing exposure time until stream is on
        self.exp_time_entry["state"] = "disabled"
        self.update_exp_time_button["state"] = "disabled"

        # Choosing if the image is read from a file or from a camera
        img_src_label = tk.Label(self.left_frame, text="Image Source:")
        self.radio_var = tk.IntVar()
        self.radio_var.set(1)  # initializing the choice
        self.offline_radiobutton = tk.Radiobutton(self.left_frame, text="Offline images", 
                                            variable=self.radio_var, value=1, command=self.stream_off)
        self.camera_radiobutton = tk.Radiobutton(self.left_frame, text="Camera stream", 
                                            variable=self.radio_var, value=2, command=self.stream_on)
        self.img_label = tk.Label(self.root)

        # Chessboard dimensions
        chess_dim_title = tk.Label(self.right_frame,
                                    text="Chessboard dimensions")
        self.n_rows_label = tk.Label(self.right_frame,
                                    text="Number or rows: ")
        self.n_rows_entry = tk.Entry(self.right_frame)
        self.n_rows_entry.insert(0, 11)

        self.n_cols_label = tk.Label(self.right_frame,
                                    text="Number or columns: ")
        self.n_cols_entry = tk.Entry(self.right_frame)
        self.n_cols_entry.insert(0, 15)

        self.sq_w_label = tk.Label(self.right_frame,
                                    text="Square width (im mm): ")
        self.sq_w_entry = tk.Entry(self.right_frame)
        self.sq_w_entry.insert(0, 15)

        # Frames
        self.left_frame.grid(row=0, column=0, padx=20, pady=20)
        self.right_frame.grid(row=0, column=1)
        self.root.grid_columnconfigure(1, minsize=self.img_preview_shape[0])

        # Making a grid for chessboard parameters
        chess_dim_title.grid(row=1, column=2)
        self.n_rows_label.grid(row=2, column=1)
        self.n_rows_entry.grid(row=2, column=2)
        self.n_cols_label.grid(row=3, column=1)
        self.n_cols_entry.grid(row=3, column=2)
        self.sq_w_label.grid(row=4, column=1)
        self.sq_w_entry.grid(row=4, column=2)

        # Packing the rest of the buttons and fields
        self.calib_button.pack()
        img_src_label.pack()
        self.offline_radiobutton.pack()
        self.camera_radiobutton.pack()
        self.load_img_button.pack()
        self.exp_time_label.pack()
        self.exp_time_entry.pack()
        self.update_exp_time_button.pack()

        self.img_label.grid(row=1, column=1)
        self.root.mainloop()

    def stream_on(self):
        """Opening a camera device when radio button is clicked
        """
        self.load_img_button["state"] = "disabled"

        if self.basler_camera_exists:
            self.exp_time_entry["state"] = "normal"
            self.update_exp_time_button["state"] = "normal"

        # Using a Basler camera
        if self.basler_camera_exists:
            # Opening the camera object and setting the exposure time
            self.camera.Open()
            self.camera.ExposureTimeAbs.SetValue(self.exp_time)  # in us 

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
        self.exp_time_entry["state"] = "disabled"
        self.update_exp_time_button["state"] = "disabled"
    

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
                            frame = cv2.resize(self.img, self.img_preview_shape)

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
    

    def render_image(self, image=None, max_dim=None):
        """Rendering the image so that tkinter can show it.

        Args:
            image (numpy array, optional): Image in OpenCV type. If not given,
                                            self.img will be used. Defaults to None.
            max_dim (int, optional): Maximum width of the preview image. If not given, 
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
        pil_img = Image.fromarray(preview_img)
        render = ImageTk.PhotoImage(pil_img)
        return render

    def load_image(self):
        """Callback for pressing the Load Image button.
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
    
    def update_exp_time(self):
        """Callback for pressing the Update Exposure Time button
        """
        exp_time = self.exp_time_entry.get()
        exp_time = self.read_parameter("exposure time", exp_time)
        if exp_time is not None:
            self.exp_time = exp_time
            if self.camera.IsGrabbing():
                self.camera.ExposureTimeAbs.SetValue(exp_time)
    
    @staticmethod
    def read_parameter(param_name, param, param_type="int"):
        """ Reading a chessboard parameter from the GUI and checking
        that it is in right format (integer or float)

        Args:
            param_name (string): The name of the parameter in the GUI
            param (string): The parameter read from the GUI as text
            param_type (string): Expected parameter type (Defaults to "int")

        Returns:
            int/None: The parameter as integer or None if the format is not correct
        """
        try:
            if param_type == "int":
                param = int(param)
            elif param_type == "float":
                param = float(param)
            else:
                raise ValueError(f"Incorrect type {param_type}!")
        except ValueError:
            mb.showerror("Error", f"Parameter '{param_name}' should be {param_type}!")
            return None
        else:
            if param < 0:
                mb.showerror("Error", f"Parameter '{param_name}' should be greater than zero!")
                return None
        
        return param
        

    def calibrate(self):
        """Callback for pressing the Calibrate button.
        """
        if self.img is None:
            mb.showerror("Error", "Load the image first!")
        else:
            # Reading the parameters from GUI and checking them
            n_rows = self.n_rows_entry.get()
            n_cols = self.n_cols_entry.get()
            square_width = self.sq_w_entry.get()
            n_rows = self.read_parameter("number of rows", n_rows)
            n_cols = self.read_parameter("number of columns", n_cols)
            square_width = self.read_parameter("square width", square_width, "float")

            # If all parameters are correctly given, let's calibrate!
            if (n_rows is not None and n_cols is not None and
                square_width is not None):

                cal_data, info = calibrate_main(self.img, (n_cols - 1, n_rows - 1), square_width)

                # Checking the result
                if cal_data is None:
                    mb.showerror("Error", info)
                else:
                    if info < 0.5:
                        message_start = "Calibration succeeded!"
                    else:
                        message_start = """The reprojection error was more than 0.5 pixels. The 
                        calibration may be unstrustworthy."""

                    message = message_start + " " + "Click Ok to save the calibration data."
                    answer = mb.askokcancel("Save Results?", message)

                    # If the user wants to save the data, let's proceed
                    if answer:
                        savename = fd.asksaveasfilename(filetypes=[("Json files", "*.json")])

                        if savename != "":
                            if savename[-5:] != ".json":
                                savename += ".json"
                            with open(savename, 'w') as outfile:
                                outfile.write(cal_data)
                                outfile.close()

        
if __name__ == "__main__":
    app = CalGui()