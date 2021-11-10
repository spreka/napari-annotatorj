"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton
from magicgui import magic_factory

import os
import skimage.io
from roifile import ImagejRoi,ROI_TYPE
from napari.layers import Shapes
import numpy
from qtpy.QtWidgets import QFileDialog


class AnnotatorJ(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        print('AnnotatorJ plugin is starting up...')

        # ------------------
        # add some demo data
        # ------------------
        self.test_image=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),'demo/img.png')
        print('Found demo image file: {}'.format(self.test_image))
        self.test_rois=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),'demo/img_ROIs.zip')
        print('Found demo roi file: {}'.format(self.test_rois))

        # ------------
        # add defaults
        # ------------
        self.imp=None # the original image
        #self._manager=self.initRoiManager # the set of ROIs as in ImageJ
        #self._manager=None
        self.initRoiManager() # the set of ROIs as in ImageJ
        self.classColourLUT=None
        self.testMode=False # for initial testing
        self.defDir=''
        self.defFile=''

        # ---------------------------
        # add buttons and ui elements
        # ---------------------------
        btnOpen = QPushButton('Open')
        btnOpen.clicked.connect(self.openNew)

        btnLoad = QPushButton('Load')
        btnLoad.clicked.connect(self.loadROIs)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btnOpen)
        self.layout().addWidget(btnLoad)

        # greeting
        print('AnnotatorJ plugin is started | Happy annotations!')

    def openNew(self):
        # temporarily open a test image
        # later this will start a browser dialog to select the input image file
        if self.testMode==True:
            if os.path.exists(self.test_image):
                img=skimage.io.imread(self.test_image)
                print('Test image read successfully')
            else:
                print('Test image could not be found')
        else:
            # browse an original image
            # TODO
            destNameRaw,_=QFileDialog.getOpenFileName(
                self,"Select an image",
                str(os.path.join(self.defDir,self.defFile)),"Images (*.png *.bmp *.jpg *.jpeg *.tif *.tiff *.gif)")
            print(destNameRaw)
            if os.path.exists(destNameRaw):
                self.defDir=os.path.dirname(destNameRaw)
                self.defFile=os.path.basename(destNameRaw)
                img=skimage.io.imread(self.test_image)
                print('Opened file: {}'.format(destNameRaw))
            else:
                print('Could not open file: {}'.format(destNameRaw))
                return

        imageLayer = self.viewer.add_image(img,name='Image')



    def loadROIs(self):
        # temporarily load a test ImageJ ROI.zip file with contours created in ImageJ and saved with AnnotatorJ
        # later this will start a browser dialog to select the annotation file
        if self.testMode==True:
            if os.path.exists(self.test_rois):
                rois=ImagejRoi.fromfile(self.test_rois)
                print('Test roi file read successfully')
            else:
                print('Test roi file could not be found')
        else:
            # browse an ImageJ ROI zip file
            # TODO
            roiFileName,_=QFileDialog.getOpenFileName(
                self,"Select an annotation (ROI) .zip file",
                str(os.path.join(self.defDir,self.defFile)),"Archives (*.zip)")
            print(roiFileName)
            if os.path.exists(roiFileName):
                loadedROIfolder=os.path.dirname(roiFileName)
                loadedROIname=os.path.basename(roiFileName)
                rois=ImagejRoi.fromfile(roiFileName)
                print('Opened ROI: {}'.format(roiFileName))
            else:
                print('Failed to open ROI .zip file: {}'.format(roiFileName))
                return

        self.add2RoiManager(rois)
        shapesLayer=self.extractROIdata(rois)
        self.viewer.add_layer(shapesLayer)
        print('Loaded {} ROIs successfully'.format(len(rois)))

    def initRoiManager(self):
        # the rois will be stored in this object as in ImageJ's RoiManager
        self._manager=None
        # TODO

    def add2RoiManager(self,rois):
        # store all ROIs in a set as in ImageJ
        if self._manager is None:
            self._manager=rois
        elif isinstance(rois,list):
            # list of rois, add them to the current rois
            if isinstance(self._manager,ImagejRoi):
                # 1 roi object so far
                self._manager=[self._manager]+rois
            else:
                print(self._manager.dtype)
                self._manager=self._manager+rois
        elif isinstance(rois,ImagejRoi):
            # 1 roi object to add
            if isinstance(self._manager,ImagejRoi):
                # 1 roi object so far
                self._manager=[self._manager]+[rois]
            else:
                self._manager=self._manager+[rois]


    def extractROIdata(self,rois):
        # fetch the coordinates and other data from the ImageJ ROI.zip file already imported with the roifile package
        # Inputs:
        #   rois: list of ImageJ ROI objects
        # Outputs:
        #   shapesLayer: shapes layer created from the list of coordinates fetched from the input rois

        roiList=[]
        roiType='polygon' # default to this
        defColour='white'

        hasColour=False
        roiColours=[]
        roiProps={'name':[],'class':[],'nameInt':[]}
        roiTextProps={
            'text': '{nameInt}: ({class})',
            'anchor': 'center',
            'size': 10,
            'color': 'black',
            'visible':False
        }

        # loop through the rois
        for curROI in rois:
            xy=curROI.coordinates() # a list of (x,y) coordinates in the wrong order
            yx=numpy.array([[y,x] for x,y in xy]) # swapping to (y,x) coordinates
            roiList.append(yx)

            # check roi type
            if curROI.roitype==ROI_TYPE.FREEHAND:
                # freehand roi drawn in instance annotation mode
                roiType='polygon'
            elif curROI.roitype==ROI_TYPE.RECT:
                # rectangle drawn in bounding box annotation mode
                roiType='rectangle'
            else:
                # leave at the default
                roiType='polygon'

            # check if it has group attribute used as class in AnnotatorJ
            curClass=curROI.group
            if curClass>0:
                hasColour=True
                # get class colour lut
                curColour=None
                if self.classColourLUT is None:
                    self.initClassColourLUT(rois)
                
                curColour=self.classColourLUT[curClass]
                roiColours.append(curColour)
                roiProps['class'].append(curClass)
            else:
                roiColours.append(defColour)
                roiProps['class'].append(0)

            # store the roi's name
            roiProps['name'].append(curROI.name)
            roiProps['nameInt'].append(int(curROI.name))

            # TODO: fetch more data from the rois

        # fill (face) colour of rois is transparent by default, only the contours are visible
        # edge_width=0.5 actually sets it to 1
        shapesLayer = Shapes(data=roiList,shape_type=roiType,name='ROI',edge_width=0.5,edge_color=roiColours,face_color=[0,0,0,0],properties=roiProps,text=roiTextProps)

        return shapesLayer

    def initClassColourLUT(self,rois):
        # setup a colour lut
        # loop through all ROIs and assign colours by classes
        classes=[]
        for roi in rois:
            classes+=roi.group
        # find the unique class indexes
        classIdxs=numpy.unique(classes)
        # get a list of the 9 basic colours also present in AnnotatorJ's class mode
        colours=['red','green','blue','cyan','magenta','yellow','orange','white','black']
        # TODO: add much more colours!

        self.classColourLUT={}
        for x in classIdxs:
            self.classColourLUT.update({x:colours[x]})

    def setTestMode(self,mode=False):
        self.testMode=mode

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return AnnotatorJ
