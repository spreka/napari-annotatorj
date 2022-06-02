# napari-annotatorj

[![License](https://img.shields.io/pypi/l/napari-annotatorj.svg?color=green)](https://github.com/spreka/napari-annotatorj/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-annotatorj.svg?color=green)](https://pypi.org/project/napari-annotatorj)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-annotatorj.svg?color=green)](https://python.org)
[![tests](https://github.com/spreka/napari-annotatorj/workflows/tests/badge.svg)](https://github.com/spreka/napari-annotatorj/actions)
[![codecov](https://codecov.io/gh/spreka/napari-annotatorj/branch/main/graph/badge.svg)](https://codecov.io/gh/spreka/napari-annotatorj)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-annotatorj)](https://napari-hub.org/plugins/napari-annotatorj)

The napari adaptation of the ImageJ/Fiji plugin [AnnotatorJ](https://github.com/spreka/annotatorj) for easy image annotation.

![image](https://drive.google.com/uc?export=view&id=1fVfvanffTdrXvLE0m1Yo6FV5TAjh6sb2)

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using with [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/docs/plugins/index.html
-->

## Installation

Installation is possible with [pip](#pip), [napari](#bundled-napari-app) or [scripts](#script).
### Pip
You can install `napari-annotatorj` via [pip]:

    pip install napari[all]
	pip install napari-annotatorj



To install latest development version :

    pip install git+https://github.com/spreka/napari-annotatorj.git


On Linux distributions, the following error may arise upon napari startup after the installation of the plugin: `Could not load the Qt platform plugin “xcb” in “” even though it was found`. In this case, the manual install of `libxcb-xinerama0` for Qt is required:

	sudo apt install libxcb-xinerama0

### Bundled napari app
The bundled application version of [napari](https://github.com/napari/napari/releases) allows the pip install of plugins in the .zip distribution. After installation of this release, napari-annotatorj can be installed from the `Plugins --> Install/Uninstall plugins...` menu by searching for its name and clicking on the `Install` button next to it.

### Script
Single-file install is supported on [**Windows**](#windows) and [Linux](#linux) (currently). It will create a virtual environment named `napariAnnotatorjEnv` in the parent folder of the cloned repository, install the package via pip and start napari. It requires a valid Python install.

#### Windows
To start it, run in the Command prompt

	git clone https://github.com/spreka/napari-annotatorj.git
	cd napari-annotatorj
	install.bat

Or download [install.bat](https://github.com/spreka/napari-annotatorj/blob/main/install.bat) and run it from the Command prompt.

After install, you can use [startup_napari.bat](https://github.com/spreka/napari-annotatorj/blob/main/startup_napari.bat) to activate your installed virtual environment and run napari. Run it from the Command prompt with:

	startup_napari.bat


#### Linux
To start it, run in the Terminal

	git clone https://github.com/spreka/napari-annotatorj.git
	cd napari-annotatorj
	install.sh

Or download [install.sh](https://github.com/spreka/napari-annotatorj/blob/main/install.sh) and run it from the Terminal.

After install, you can use [startup_napari.sh](https://github.com/spreka/napari-annotatorj/blob/main/startup_napari.sh) to activate your installed virtual environment and run napari. Run it from the Terminal with:

	startup_napari.sh

***
## Intro

napari-annotatorj has several convenient functions to speed up the annotation process, make it easier and more fun. These *modes* can be activated by their corresponding checkboxes on the left side of the main AnnotatorJ widget.

- [Contour assist mode](#contour-assist-mode)
- [Edit mode](#edit-mode)
- [Class mode](#class-mode)
- [Overlay](#overlay)

Freehand drawing is enabled in the plugin. The "Add polygon" tool is selected by default upon startup. To draw a freehand object (shape) simply hold the mouse and drag it around the object. The contour is visualized when the mouse button is released.

See the [guide](#how-to-annotate) below for a quick start or a [demo](#demo).

***
## How to annotate

1. Open --> opens an image
2. (Optionally) 
	- ... --> Select annotation type --> Ok --> a default tool is selected from the toolbar that fits the selected annotation type
	- The default annotation type is instance
	- Selected annotation type is saved to a config file
3. Start annotating objects
	- [instance](#instance-annotation): draw contours around objects
	- [semantic](#semantic-annotation): paint the objects' area
	- [bounding box](#bounding-box-annotation): draw rectangles around the objects
4. Save --> Select class --> saves the annotation to a file in a sub-folder of the original image folder with the name of the selected class

5. (Optionally)
	- Load --> continue a previous annotation
	- Overlay --> display a different annotation as overlay (semi-transparent) on the currently opened image
	- Colours --> select annotation and overlay colours
	- ... (coming soon) --> set options for semantic segmentation and *Contour assist* mode
	- checkboxes --> Various options
		- (default) Add automatically --> adds the most recent annotation to the ROI list automatically when releasing the left mouse button
		- Smooth (coming soon) --> smooths the contour (in instance annotation type only)
		- Show contours --> displays all the contours in the ROI list
		- Contours assist --> suggests a contour in the region of an initial, lazily drawn contour using the deep learning method U-Net
		- Show overlay --> displays the overlayed annotation if loaded with the Overlay button
		- Edit mode --> edits a selected, already saved contour in the ROI list by clicking on it on the image
		- Class mode --> assigns the selected class to the selected contour in the ROI list by clicking on it on the image and displays its contour in the class's colour (can be set in the Class window); clicking on the object a second time unclassifies it
	- [^] --> quick export in 16-bit multi-labelled .tiff format; if classified, also exports by classes

***
## Instance annotation
Allows freehand drawing of object contours (shapes) with the mouse as in ImageJ.

Shape contour points are tracked automatically when the left mouse button is held and dragged to draw a shape. The shape is closed when the mouse button is released, automatically, and added to the default shapes layer (named "ROI"). In direct selection mode (from the layer controls panel), you can see the saved contour points. The slower you drag the mouse, the more contour points saved, i.e. the more refined your contour will be.

Click to watch demo video below.

[![instance-annot-demo](https://drive.google.com/uc?export=view&id=1Qd0LirjJX1Gvy_NWJ2eBV74vQjDQt5Gn)](https://drive.google.com/uc?export=view&id=18bIaqNboGMAwEN_bBnPnyXEbkAgNBmQ0)

***
## Semantic annotation
Allows painting with the brush tool (labels).

Useful for semantic (e.g. scene) annotation. Currently saves all labels to binary mask only (foreground-background).

***
## Bounding box annotation
Allows drawing bounding boxes (shapes, rectangles) around objects with the mouse.

Useful for object detection annotation.

***
## Contour assist mode
Assisted annotation via a pre-trained deep learning model's suggested contour.

1. initialize a contour with mouse drag around an object
2. the suggested contour is displayed automatically
3. modify the contour:
    - edit with mouse drag or 
    - erase holding "Alt"
4. finalize it
    - accept with pressing "q" or
    - reject with pressing "Ctrl" + "Del"

- if the suggested contour is a merge of multiple objects, you can erase the dividing line around the object you wish to keep, and keep erasing (or splitting with the eraser) until the object you wish to keep is the largest, then press "q" to accept it
- this mode requires a Keras model to be present in the [model folder](#configure-model-folder)

Click to watch demo video below

[![contour-assist-demo](https://drive.google.com/uc?export=view&id=1xAGJu1SeM3mxMgxTQ-uIBECDEFNZL-8L)](https://drive.google.com/uc?export=view&id=1VTd6RScjNfAwi3vMk-bU87U4ucPmOO_M "Click to watch contour assist demo")

***
## Edit mode
Allows to modify created objects with a brush tool.

1. select an object (shape) to modify by clicking on it
2. an editing layer (labels layer) is created for painting automatically
3. modify the contour:
    - edit with mouse drag or 
    - erase holding "Alt"
4. finalize it
    - accept with pressing "q" or
    - delete with pressing "Ctrl" + "Del" or
    - revert changes with pressing "Esc" (to the state before editing)

- if the edited contour is a merge of multiple objects, you can erase the dividing line around the object you wish to keep, and keep erasing (or splitting with the eraser) until the object you wish to keep is the largest, then press "q" to accept it

Click to watch demo video below

[![edit-mode-demo](https://drive.google.com/uc?export=view&id=1Mqjd6hdKyE24xXEOlyLO1yai3hnGSEyR)](https://drive.google.com/uc?export=view&id=10MQm53hblLKQlfBNrfUsi1vxvIdTbzCZ "Click to watch edit mode demo")

***
## Class mode
Allows to assign class labels to objects by clicking on shapes.

1. select a class from the class list to assign
2. click on an object (shape) to assign the selected class label to it
3. the contour colour of the clicked object will be updated to the selected class colour, plus the class label is updated in the text properties of object (turn on "display text" on the layer control panel to see the text properties as `objectID:(classLabel)` e.g. 1:(0) for the first object)

- optionally, you can set a default class for all currently unlabelled objects on the ROI (shapes) layer by selecting a class from the drop-down menu on the right to the text label "Default class"
- class colours can be changed with the drop-down menu right to the class list; upon selection, all objects whose class label is the currently selected class will have their contour colour updated to the selected colour
- clicking on an object that has already been assigned a class label will unclassify it: assign the label *0* to it

Click to watch demo video below

[![class-mode-demo](https://drive.google.com/uc?export=view&id=1sAuTTjPaFs_qlbIj3NQlht-UjI2WKsHr)](https://drive.google.com/uc?export=view&id=1uOmznUvfHEFvviWTtOnUHty8rkKyWR7Q "Click to watch class mode demo")

***
## Export
See also: [Quick export](#quick-export)

The exporter plugin AnnotatorJExport can be invoked from the Plugins menu under the plugin name `napari-annotatorj`. It is used for batch export of annotations to various formats directly suitable to train different types of deep learning models. See a [demonstrative figure](https://raw.githubusercontent.com/spreka/annotatorj/master/demos/annotation_and_export_types.png) in the [AnnotatorJ repository](https://github.com/spreka/annotatorj) and further description in its [README](https://github.com/spreka/annotatorj#export) or [documentation](https://github.com/spreka/annotatorj/blob/master/AnnotatorJ_documentation.pdf).

1. browse original image folder with either the
    - "Browse original..." button or
    - text input field next to it
2. browse annotation folder with either the
    - "Browse annot..." button or
    - text input field next to it
3. select the export options you wish to export the annotations to (see tooltips on hover for help)
    - at least one export option must be selected to start export
    - (optional) right click on the checkbox "Coordinates" to switch between the default COCO format and YOLO format; see [explanation](#coordinate-formats)
4. click on "Export masks" to start the export
    - this will open a progress bar in the napari window and close it upon finish

The folder structure required by the exporter is as follows:

```
image_folder
	|--- image1.png
	|--- another_image.png
	|--- something.png
	|--- ...

annotation_folder
	|--- image1_ROIs.zip
	|--- another_image_ROIs.zip
	|--- something_ROIs.zip
	|--- ...
```

Multiple export options can be selected at once, any selected will create a subfolder in the folder where the annotations are saved.

***
## Quick export
Click on the "[^]" button to quickly save annotations and export to mask image. It saves the current annotations (shapes) to an ImageJ-compatible roi.zip file and a generated a 16-bit multi-labelled mask image to the subfolder "masks" under the current original image's folder.


***
## Coordinate formats
In the AnnotatorJExport plugin 2 coordinates formats can be selecting by right clicking on the Coordinates checkbox: COCO or YOLO. The default is COCO.

*COCO format*:
- `[x, y, width, height]` based on the top-left corner of the bounding box around the object
- coordinates are not normalized
- annotations are saved with header to 
    - .csv file
    - tab delimeted

*YOLO format*:
- `[class, x, y, width, height]` based on the center point of the bounding box around the object
- coordinates are normalized to the image size as floating point values between 0 and 1
- annotations are saved with header to
    - .txt file
    - whitespace delimeted
    - class is saved as the 1st column

***
## Overlay
A separate annotation file can be loaded as overlay for convenience, e.g. to compare annotations.

1. load another annotation file with the "Overlay" button

- (optional) switch its visibility with the "Show overlay" checkbox
- (optional) change the contour colour of the overlay shapes with the ["Colours" button](#change-colours)

***
## Change colours
Clicking on the "Colours" button opens the Colours widget where you can set the annotation and overlay colours.

1. select a colour from the drop-down list either next to the text label "overlay" or "annotation"
2. click the "Ok" button to apply changes

- contour colour of shapes on the annotation shapes layer (named "ROI") that already have a class label assigned to them will **not** be updated to the new annotation colour, only those not having a class label (the class label can be displayed with the "display text" checkbox on the layer controls panel as `objectID:(classLabel)` e.g. 1:(0) for the first object)
- contour colour of shapes on the overlay shapes layer (named "overlay") will all have the overlay colour set, regardless of any existing class information saved to the annotation file loaded as overlay

***
## Configure model folder
The Contour assist mode imports a pre-trained Keras model from a folder named *models* under exactly the path *napari_annotatorj*. This is automatically created on the first startup in your user folder:
- `C:\Users\Username\.napari_annotatorj` on Windows
- `\home\username\.napari_annotatorj` on Linux

A pre-trained model for nucleus segmentation is automatically downloaded from the GitHub repository of the [ImageJ version of AnnotatorJ](https://github.com/spreka/annotatorj/releases/tag/v0.0.2-model). The model will be saved to `[your user folder]\.napari_annotatorj\models\model_real.h5`. This location is printed to the console (command prompt or powershell on Windows, terminal on Linux).

(deprecated) When bulding from source the model folder is located at *path\to\napari-annotatorj\src\napari_annotatorj\models* whereas installing from pypi it is located at *path\to\virtualenv\Lib\site-packages\napari_annotatorj\models*.

The model must be in either of these file formats:
- config .json file + weights file: *model_real.json* and *model_real_weights.h5*
- combined weights file: *model_real.hdf5*

You can also train a new model on your own data in e.g. Python and save it with this code block:

```python
	# save model as json
	model_json=model.to_json()
	with open(‘model_real.json’, ‘w’) as f:
		f.write(model_json)
	
	# save weights too
	model.save_weights(‘model_real_weights.h5’)

```
This configuration will change in the next release to allow model browse and custom model name in an options widget.

***
## Demo
Run a demo of napari-annotatorj with sample data: a small 3-channel RGB image as original image and an ImageJ roi.zip file as annotations loaded.

```shell
    # from the napari-annotatorj folder
	python src/napari_annotatorj/load_imagej_roi.py
```
Alternatively, you can startup the napari-annotatorj plugin by running

```shell
    # from the napari-annotatorj folder
	python src/napari_annotatorj/startup_annotatorj.py
```

***
## Setting device for deep learning model prediction
The [Contour assist](#contour-assist-mode) mode uses a pre-trained U-Net model for suggesting contours based on a lazily initialized contour drawn by the user. The default configuration loads and runs the model on the CPU so all users can run it. It is possible to switch to GPU if you have:
- a CUDA-capable GPU in your computer
- nVidia's CUDA toolkit + cuDNN installed

See installation guide on [nVidia's website](https://developer.nvidia.com/cuda-downloads) according to your system.

To switch to GPU utilization, edit [_dock_widget.py](https://github.com/spreka/napari-annotatorj/blob/main/src/napari_annotatorj/_dock_widget.py#L112) and set to the device you would like to use. Valid values are `'cpu','0','1','2',...`. The default value is `cpu`. The default GPU device is `0` if your system has any CUDA-capable GPU. If the device you set cannot be found or utilized by the code, it will fall back to `cpu`. An informative message is printed to the console upon plugin startup.

***
## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-annotatorj" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/spreka/napari-annotatorj/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
