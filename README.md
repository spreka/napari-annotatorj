# napari-annotatorj

[![License](https://img.shields.io/pypi/l/napari-annotatorj.svg?color=green)](https://github.com/spreka/napari-annotatorj/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-annotatorj.svg?color=green)](https://pypi.org/project/napari-annotatorj)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-annotatorj.svg?color=green)](https://python.org)
[![tests](https://github.com/spreka/napari-annotatorj/workflows/tests/badge.svg)](https://github.com/spreka/napari-annotatorj/actions)
[![codecov](https://codecov.io/gh/spreka/napari-annotatorj/branch/main/graph/badge.svg)](https://codecov.io/gh/spreka/napari-annotatorj)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-annotatorj)](https://napari-hub.org/plugins/napari-annotatorj)

The napari adaptation of the ImageJ/Fiji plugin AnnotatorJ for easy image annotation.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using with [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/docs/plugins/index.html
-->

## Installation

You can install `napari-annotatorj` via [pip]:

    pip install napari-annotatorj



To install latest development version :

    pip install git+https://github.com/spreka/napari-annotatorj.git


## How to use

napari-annotatorj has several convenient functions to speed up the annotation process, make it easier and more fun. These *modes* can be activated by their corresponding checkboxes on the left side of the main AnnotatorJ widget.

Freehand drawing is enabled in the plugin. The "Add polygon" tool is selected by default upon startup. To draw a freehand object (shape) simply hold the mouse and drag it around the object. The contour is visualized when the mouse button is released.

![image](https://drive.google.com/uc?export=view&id=1fVfvanffTdrXvLE0m1Yo6FV5TAjh6sb2)

### Contour assist mode
- assisted annotation via a pre-trained deep learning model's suggested contour
    1. initialize a contour with mouse drag around an object
    2. the suggested contour is displayed automatically
    3. modify the contour:
        - edit with mouse drag or 
        - erase holding "Alt"
    4. finalize it
        - accept with pressing "q" or
        - reject with pressing "Ctrl" + "Del"
- click to watch demo below

[![contour-assist-demo](https://drive.google.com/uc?export=view&id=1xAGJu1SeM3mxMgxTQ-uIBECDEFNZL-8L)](https://drive.google.com/uc?export=view&id=1VTd6RScjNfAwi3vMk-bU87U4ucPmOO_M "Click to watch contour assist demo")

### Edit mode
- allows to modify created objects with a brush tool
    1. select an object (shape) to modify by clicking on it
    2. an editing layer (labels layer) is created for painting automatically
    3. modify the contour:
        - edit with mouse drag or 
        - erase holding "Alt"
    4. finalize it
        - accept with pressing "q" or
        - delete with pressing "Ctrl" + "Del"
        - revert changes with pressing "Esc" (to the state before editing)
- click to watch demo below

[![edit-mode-demo](https://drive.google.com/uc?export=view&id=1Mqjd6hdKyE24xXEOlyLO1yai3hnGSEyR)](https://drive.google.com/uc?export=view&id=10MQm53hblLKQlfBNrfUsi1vxvIdTbzCZ "Click to watch edit mode demo")

### Class mode
- allows to assign class labels to objects
    1. select a class from the class list to assign
    2. click on an object (shape) to assign the selected class label to it
    3. the contour colour of the clicked object will be updated to selected the class colour, plus the class label is updated in the text properties of object (turn on "display text" on the layer control panel to see the text properties as `objectID:(classLabel)` e.g. 1:(0) for the first object)
- optionally, you can set a default class for all currently unlabelled objects on the ROI (shapes) layer by selecting a class from the drop-down menu on the right to the text label "Default class"
- class colours can be changed with the drop-down menu right to the class list; upon selection, all objects whose class label is the currently selected class will have their contour colour updated to the selected colour
- clicking on an object that has already been assigned a class label will unclassify it: assign the label *0* to it
- click to watch demo below (coming soon, screenshot only for now)

![image](https://drive.google.com/uc?export=view&id=1sAuTTjPaFs_qlbIj3NQlht-UjI2WKsHr)


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
