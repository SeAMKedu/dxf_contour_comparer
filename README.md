# EDIT Contour-to-DXF comparer

This tool compares the image of a 2D part to the DXF file describing its geometry. The tool first binarizes the image and then extracts its contours to compare their similarity with the geometry in the DXF file. Therefore, good illumination is crucial. The tool compares the 2D geometry so it is usable with sheet metal parts. This was originally done in the EDIT project for the machine vision inspection point for the FMS cell of the machine laboratory of SeAMK.

[This video](https://www.youtube.com/watch?v=Q_JCKgVhxt8) shows the functioning of the tool. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

The must thing to have is 
- [Python 3.X](https://www.python.org/downloads/). The version used in the development was 3.7.0.

The tool can analyze offline images and images captured from the camera. Currently, only Basler cameras and integrated webcams are supported. 
If you are willing to use the Basler camera with the software, you also need 
- [Pylon](https://www.baslerweb.com/en/sales-support/downloads/software-downloads/#type=pylonsoftware;language=all;version=all)


### Installing

It is recommed to use virtual environments with the Python projects. You may only use Python in PATH. Then, you do not need the virtual environment. A virtual environment is made like this with a Windows computer:

```
<path_to_python>\python.exe -m venv <name_of_the_environment>
```

The folder with the name <name_of_the_environment> appears. First, clone this repository. Then, make the virtual environment inside the project folder.

If you are using Visual Studio Code, it should detect the virtual environment when opening it inside the project folder. Either by choosing the folder from File -> Open Folder or from command prompt with a command

```
code .
```
if you are in the project folder. The virtual environment name should be seen in the lower left corner after the interpreter name, i.e. Python 3.7.0 64-bit ('virtual_env'). If it does not, it can be selected manually by cliking the interpreter name -> Enter interpreter path... and browsing to the _<name_of_the_environment>\Scripts\python.exe_ When the virtual environment is activated, its name is in bracket in the terminal before the path. Like this:

```
(virtal_env) C:\Users\<my_id>\<my_fancy_folder>\>
```

The virtual enviroment can be activated manually in Visual Studio Code by opening a new terminal window from the + symbol. Command prompt is recommended. PowerShell may have issues with rights.

Next, install the requirements

```
pip install -r requirements.txt
```

Now, everything should be installed (I hope).

## Running the tests

### DXF-contour-checks

The folder _test_files_ has everything needed for testing. The main function is **gui.py**. Launch it. It will look like this:

![GUI image](/documentation_images/gui_info.png)

After selecting the calibration file, the dxf file and the image (also possible to grab images from the camera after changing the image source), the results are displayed in another window when clicking "Check". Like this:

![Successful result image](/documentation_images/result1.png)

If there are some contours that are not matching, they are drawn in red. Like in here:

![Failed result image](/documentation_images/result2.png)

There are four (modified) image that fail with the default 1 mm maximum deviation:
- 002/Image__fail.png (a hole in wrong location, the one shown in the second image)
- 002/Image__fail2.png (the largest curve in the outermost contour bit different, actually the fail is seen in the hole since the outer contours are always matched as well as possible, see [Operation logic](#operation-logic) for explanation)
- 002/Image__fail3.png (extra holes)
- 002/Image__fail4.png (cracks)

Of course, fails can be simulated by using the wrong dxf file for the image

![Failed result with wrong dxf](/documentation_images/result3.png)

or playing with the webcam.

![Failed result with webcam](/documentation_images/result4.png)

### Calibration

A file like the used _calibration_file.json_ can be constructed with **calib_gui.py**. When it is run, the GUI opens:

![Calibration GUi image](/documentation_images/calib_gui_info.png?)

There is one test file for calibration, _calib_image.png_. The file _calibration_file.json_ has been done with that image. The camera has to be recalibrated if
- the focus changes
- the zoom changes
- its angle or the target plane's angle changes
- the working distance changes

In a nutshell: if someone touches the camera, it has to be recalibrated. The camera is calibrated by using a print of a chessboard. It can be acquired from [here](https://calib.io/pages/camera-calibration-pattern-generator).

## Operation logic

### Reading the dxf file

The DXF file by applying the library [**ezdxf**](https://ezdxf.mozman.at/). The library supports rendering an image out of the dxf file but the pixel per millimeter resolution is not defined. That is why the class _Dxf2ContourReader_ in **dxf_to_contour.py** is used to translate the information in the dxf file to the OpenCV contour format.

### Finding the contours from the image

This is done just by thresholding the image with the OpenCV function _threshold_ and then finding the contours by _findContours_. The radio button in the GUI for selecting the object color/background color swaps between the flags THRESHOLD_BINARY and THRESHOLD_BINARY_INV.

### Aligning the contours

Alignment is done by the class _ContourComparer_ in the file **compare_contours.py**. The outermost contour is used in alignment. The fine alignment is done with iterative closest points (ICP) algorithm. The implementation used is in the file **icp.py** and its is from the second answer (by Vincent Bénet) in [here](https://stackoverflow.com/questions/20120384/iterative-closest-point-icp-implementation-on-python). As ICP is an iterative algorithm, it needs a good guess for the initial pose. Otherwise, it may end in a local minimum and thus give bad results. The initial guess is done with primary component analysis (PCA). The function _get_orientation_pca_ in the file **compare_contours.py** does this and it is done based on the [tutorial in OpenCV pages](https://docs.opencv.org/3.4/d1/dee/tutorial_introduction_to_pca.html).

PCA finds the orientation of a group of xy points by finding the direction the data varies most. So it kind of finds the coordinate axes of the data. However, the direction of the axis ("the direction of +") is arbitrary. That is why the angle of the axis is calculated and then that angle and angle + 180 are both tested. Below an initial situation for two contours is shown:

![Initial situation](/documentation_images/contours_initial.png)

Then, PCA is applied and the center and orientation are calculated. Below are the contours aligned with that information. Orientation and orientation + 180 degrees both shown:

![After PCA](/documentation_images/contours_pca.png)

Nearest neighbors algorithm of the library **sklearn** is used to calculate, which transformation fits better. That is chosen for the initial guess for the ICP. ICP still refines the result as can be seen below:

![After ICP](/documentation_images/contours_icp.png)

After the best transformation is detected with ICP, also the inner contours are transformed with that. Then, the closest inner contours in the model and the image are matched and. Nearest neighbor algorithm is used again to find the distances of the contours from each other.

### Known issues

- As PCA is based on finding the orientation of images, it will fail with symmetrical forms such as squares and circles. These special cases need some other strategy. Maybe the symmetry should be first tested and then just orientations +90 degrees and +270 should tested as well. Well, for the circles that is not enough... Also, doing the PCA for all blob points, not only the outer contour points, would utilize also the information of the inner shapes in finding the orientation.
- As the outermost contours are used in aligning the shapes as discussed in the section of test file Image__fail2 in [DXF-contour-checks](#dxf-contour-checks), slight deviations in them are not detected since the ICP algorithm finds the best possible transformation between the shapes and thus minimizes the local errors. If there are inner contours, this causes deviation to them that could be detected. This could be tackled by calculating parameters as the area or the perimeter from the outer contours of the model and the image and comparing them to each other. Maybe the OpenCV function matchShapes could help as well but it is hard to find the right threshold with that (is it good when its below 0.01 or 0.001?!)
- The code of the function _match_contour_to_model_ in the file **compare_contours.py** is full of calls to function .reshape(). This looks pretty messy but is kind of compulsory since OpenCV and NumPy functions handle coordinate arrays differently: for OpenCV, they are usually in form [[[x1, y1]], [[x2, y2]], ...], for NumPy, they are usually in form [[x1, y1], [x2, y2]] and also most of the functions in these modules expect the coordinates to be in those formats. So if one needs functions in both modules one after another, one has to reshape the arrays all the time.
- Currently, any numerical value for failures is not calculated. The average, maximum and minimum point-to-point distances could be returned or logged or something. That is an easy task and can be done from the _distances_ array returned by _NearestNeigbors_

## Authors

**Juha Hirvonen**, the initial work

