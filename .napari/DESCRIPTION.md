

<!-- This file is designed to provide you with a starting template for documenting
the functionality of your plugin. Its content will be rendered on your plugin's
napari hub page.

The sections below are given as a guide for the flow of information only, and
are in no way prescriptive. You should feel free to merge, remove, add and 
rename sections at will to make this document work best for your plugin. 

# Description

This should be a detailed description of the context of your plugin and its 
intended purpose.

If you have videos or screenshots of your plugin in action, you should include them
here as well, to make them front and center for new users. 

You should use absolute links to these assets, so that we can easily display them 
on the hub. The easiest way to include a video is to use a GIF, for example hosted
on imgur. You can then reference this GIF as an image.

![Example GIF hosted on Imgur](https://i.imgur.com/A5phCX4.gif)

Note that GIFs larger than 5MB won't be rendered by GitHub - we will however,
render them on the napari hub.

The other alternative, if you prefer to keep a video, is to use GitHub's video
embedding feature.

1. Push your `DESCRIPTION.md` to GitHub on your repository (this can also be done
as part of a Pull Request)
2. Edit `.napari/DESCRIPTION.md` **on GitHub**.
3. Drag and drop your video into its desired location. It will be uploaded and
hosted on GitHub for you, but will not be placed in your repository.
4. We will take the resolved link to the video and render it on the hub.

Here is an example of an mp4 video embedded this way.

https://user-images.githubusercontent.com/17995243/120088305-6c093380-c132-11eb-822d-620e81eb5f0e.mp4

# Intended Audience & Supported Data

This section should describe the target audience for this plugin (any knowledge,
skills and experience required), as well as a description of the types of data
supported by this plugin.

Try to make the data description as explicit as possible, so that users know the
format your plugin expects. This applies both to reader plugins reading file formats
and to function/dock widget plugins accepting layers and/or layer data.
For example, if you know your plugin only works with 3D integer data in "tyx" order,
make sure to mention this.

If you know of researchers, groups or labs using your plugin, or if it has been cited
anywhere, feel free to also include this information here.

# Quickstart

This section should go through step-by-step examples of how your plugin should be used.
Where your plugin provides multiple dock widgets or functions, you should split these
out into separate subsections for easy browsing. Include screenshots and videos
wherever possible to elucidate your descriptions. 

Ideally, this section should start with minimal examples for those who just want a
quick overview of the plugin's functionality, but you should definitely link out to
more complex and in-depth tutorials highlighting any intricacies of your plugin, and
more detailed documentation if you have it.

# Additional Install Steps (uncommon)
We will be providing installation instructions on the hub, which will be sufficient
for the majority of plugins. They will include instructions to pip install, and
to install via napari itself.

Most plugins can be installed out-of-the-box by just specifying the package requirements
over in `setup.cfg`. However, if your plugin has any more complex dependencies, or 
requires any additional preparation before (or after) installation, you should add 
this information here.

# Getting Help

This section should point users to your preferred support tools, whether this be raising
an issue on GitHub, asking a question on image.sc, or using some other method of contact.
If you distinguish between usage support and bug/feature support, you should state that
here.

# How to Cite

Many plugins may be used in the course of published (or publishable) research, as well as
during conference talks and other public facing events. If you'd like to be cited in
a particular format, or have a DOI you'd like used, you should provide that information here. -->

# Description
This plugin allows easy object annotation on 2D images. Annotation is made quick, easy and fun, just start drawing! See a [quick start](#quick-start) guide below.

![image](https://drive.google.com/uc?export=view&id=1fVfvanffTdrXvLE0m1Yo6FV5TAjh6sb2)

It is the napari version of the ImageJ plugin [AnnotatorJ](https://github.com/spreka/annotatorj).

## What kind of data it works on
**2D images**. That's the only requirement. Whether you have microscopy images of cells or tissue, natural photos of cats and dogs, vehichle dash-cam footage, industrial pipeline monitoring etc., just open the image and you can start annotating.

Annotations are save to ImageJ-compatible roi.zip files. [Export](#export) is possible to several file formats depending on the intended application.

## Intended users
**Anyone**. No experience in computer science or underlying technology is needed; if you know how to use MS Paint, you are ready to start annotating. Biologists, programmers, even children can use it. See [quick start](#quick-start) guide or [demos](#demo). If you experience any issues or have questions, feel free to open an [issue](https://github.com/spreka/napari-annotatorj/issues) on GitHub.

## Main features

Why choose napari-annotatorj?
- Assisted annotation is possible with automatic deep learning-based [contour suggestion](#contour-assist-mode),
- freehand contour drawing in [instance annotation](#instance-annotation),
- shape [editing](#edit-mode) via painting labels,
- [class annotation](#class-mode),
- [export](#export) to formats directly suitable for deep CNN training
- import of previous annotations as [overlay](#overlay); e.g. when comparing annotations or curating
- and more.

See [demos](#demo).

## Quick start
Demo data is available in the GitHub repository's [demo](https://github.com/spreka/napari-annotatorj/tree/main/demo) folder.

napari-annotatorj has several convenient functions to speed up the annotation process, make it easier and more fun. These *modes* can be activated by their corresponding checkboxes on the left side of the main AnnotatorJ widget.

- [Contour assist mode](#contour-assist-mode)
- [Edit mode](#edit-mode)
- [Class mode](#class-mode)
- [Overlay](#overlay)

Freehand drawing is enabled in the plugin. The "Add polygon" tool is selected by default upon startup. To draw a freehand object (shape) simply hold the mouse and drag it around the object. The contour is visualized when the mouse button is released.

See the [guide](#how-to-annotate) below or a [demo](#demo-scripts) script.

## Instance annotation
Allows freehand drawing of object contours (shapes) with the mouse as in ImageJ.

Shape contour points are tracked automatically when the left mouse button is held and dragged to draw a shape. The shape is closed when the mouse button is released, automatically, and added to the default shapes layer (named "ROI"). In direct selection mode (from the layer controls panel), you can see the saved contour points. The slower you drag the mouse, the more contour points saved, i.e. the more refined your contour will be.

Click to watch demo video below.

[![instance-annot-demo](https://drive.google.com/uc?export=view&id=1Qd0LirjJX1Gvy_NWJ2eBV74vQjDQt5Gn)](https://drive.google.com/uc?export=view&id=18bIaqNboGMAwEN_bBnPnyXEbkAgNBmQ0)

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


## Semantic annotation
Allows painting with the brush tool (labels).

Useful for semantic (e.g. scene) annotation. Currently saves all labels to binary mask only (foreground-background).

## Bounding box annotation
Allows drawing bounding boxes (shapes, rectangles) around objects with the mouse.

Useful for object detection annotation.


## Demo
## Instance annotation mode
See [above](#instance-annotation).

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


## Quick export
Click on the "[^]" button to quickly save annotations and export to mask image. It saves the current annotations (shapes) to an ImageJ-compatible roi.zip file and a generated a 16-bit multi-labelled mask image to the subfolder "masks" under the current original image's folder.


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


## Overlay
A separate annotation file can be loaded as overlay for convenience, e.g. to compare annotations.

1. load another annotation file with the "Overlay" button

- (optional) switch its visibility with the "Show overlay" checkbox
- (optional) change the contour colour of the overlay shapes with the ["Colours" button](#change-colours)


## Change colours
Clicking on the "Colours" button opens the Colours widget where you can set the annotation and overlay colours.

1. select a colour from the drop-down list either next to the text label "overlay" or "annotation"
2. click the "Ok" button to apply changes

- contour colour of shapes on the annotation shapes layer (named "ROI") that already have a class label assigned to them will **not** be updated to the new annotation colour, only those not having a class label (the class label can be displayed with the "display text" checkbox on the layer controls panel as `objectID:(classLabel)` e.g. 1:(0) for the first object)
- contour colour of shapes on the overlay shapes layer (named "overlay") will all have the overlay colour set, regardless of any existing class information saved to the annotation file loaded as overlay


# For coding users
## Demo scripts
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

## Setting device for deep learning model prediction
The [Contour assist](#contour-assist-mode) mode uses a pre-trained U-Net model for suggesting contours based on a lazily initialized contour drawn by the user. The default configuration loads and runs the model on the CPU so all users can run it. It is possible to switch to GPU if you have:
- a CUDA-capable GPU in your computer
- nVidia's CUDA toolkit + cuDNN installed

See installation guide on [nVidia's website](https://developer.nvidia.com/cuda-downloads) according to your system.

To switch to GPU utilization, edit [_dock_widget.py](https://github.com/spreka/napari-annotatorj/blob/main/src/napari_annotatorj/_dock_widget.py#L112) and set to the device you would like to use. Valid values are `'cpu','0','1','2',...`. The default value is `cpu`. The default GPU device is `0` if your system has any CUDA-capable GPU. If the device you set cannot be found or utilized by the code, it will fall back to `cpu`. An informative message is printed to the console upon plugin startup.

## License
Distributed under the terms of the [BSD-3](https://opensource.org/licenses/BSD-3-Clause) license,
"napari-annotatorj" is free and open source software.

## Getting help
If you experience any issues or have questions, feel free to open an [issue](https://github.com/spreka/napari-annotatorj/issues) on GitHub.

## How to cite
Réka Hollandi, Ákos Diósdi, Gábor Hollandi, Nikita Moshkov, Péter Horváth (2020): “AnnotatorJ: an ImageJ plugin to ease hand-annotation of cellular compartments”, Molecular Biology of the Cell, Vol. 31, No. 20, 2179-2186, https://doi.org/10.1091/mbc.E20-02-0156