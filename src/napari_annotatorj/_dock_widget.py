"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
from qtpy.QtWidgets import QWidget, QHBoxLayout,QVBoxLayout, QPushButton, QCheckBox,QLabel,QMessageBox,QFileDialog,QDialog,QComboBox,QListWidget,QAbstractItemView,QLineEdit,QMenu,QRadioButton
from magicgui import magic_factory

import os
import skimage.io
from roifile import ImagejRoi,ROI_TYPE,roiwrite
from napari.layers import Shapes, Image, Labels
import numpy
from qtpy.QtCore import Qt,QSize,QRect
from qtpy.QtGui import QPixmap,QCursor
#from napari.layers.Shapes import mode
from napari.layers.shapes import _shapes_key_bindings as key_bindings
from napari.layers.shapes import _shapes_mouse_bindings as mouse_bindings
from napari.layers.labels import _labels_mouse_bindings as labels_mouse_bindings
import warnings
from cv2 import cv2
from copy import deepcopy
#from napari.qt import create_worker #thread_worker
from napari.qt.threading import thread_worker #create_worker

# suppress numpy's FutureWarning: numpy\core\numeric.py:2449: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison!
warnings.filterwarnings('ignore', category=FutureWarning)


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
        #self.manager=self.initRoiManager # the set of ROIs as in ImageJ
        #self.manager=None
        self.initRoiManager() # the set of ROIs as in ImageJ
        self.roiCount=0
        self.roiLayer=None # remember the current ROI shapes layer
        self.classColourLUT=None
        self.testMode=False # for initial testing
        self.defDir=''
        self.defFile=''
        self.editMode=False
        self.startedEditing=False
        self.editROIidx=-1
        self.origEditedROI=None
        self.brushSize=5
        self.imgSize=None

        # logo files: annotatorj_logo_dark, annotatorj_logo_light, annotatorj_logo_red
        self.logoFile='annotatorj_logo_dark'

        self.closeingOnPurpuse=False
        self.started=False
        self.curFileList=None
        self.curFileIdx=-1
        self.stepping=False
        self.enableMaskLoad=False
        self.autoMaskLoad=False
        self.maskFolderInited=False

        self.inAssisting=False
        self.addAuto=False
        self.contAssist=False
        self.classMode=False

        self.imageFromArgs=False
        self.finishedSaving=False
        self.startedClassifying=False
        self.loadedROI=False

        self.trainedUNetModel=None
        self.modelJsonFile='model_real'
        self.modelWeightsFile='model_real_weights.h5'
        self.modelFullFile='model_real.hdf5'
        self.modelFolder=os.path.join(os.path.dirname(__file__),'models')
        self.selectedCorrMethod=0 # U-Net
        self.invertedROI=None
        self.curPredictionImage=None
        self.curPredictionImageName=None
        self.curOrigImage=None
        # '0' is the default GPU if any, otherwise fall back to cpu
        # valid values are: ['cpu','0','1','2',...]
        self.gpuSetting='cpu'

        # default options for contour assist and semantic annotation
        # can be overwritten in config file
        # threshold of intensity difference for contour assisting region growing
        self.intensityThreshVal=0.1 #0.1
        self.intensityThreshValR=0.2
        self.intensityThreshValG=0.4
        self.intensityThreshValB=0.2
        # threshold of distance in pixels from the existing contour in assisting region growing
        self.distanceThreshVal=17
        # brush sizes
        self.correctionBrushSize=10
        self.semanticBrushSize=50

        self.defColour='white'

        self.classesFrame=None
        self.listModelClasses=None
        self.selectedClassColourIdx=None
        self.classFrameNames=[]
        self.selectedClassNameNumber=-2
        self.usedClassNameNumbers=[]
        self.defaultClassNumber=-1
        self.classNumberCounter=0
        self.classNameLUT={} # dict: string:int
        self.classFrameColours=[]

        self.ExportFrame=None

        # get a list of the 9 basic colours also present in AnnotatorJ's class mode
        self.colours=['red','green','blue','cyan','magenta','yellow','orange','white','black']

        # supported image formats
        self.imageExsts=['.png','.bmp','.jpg','.jpeg','.tif','.tiff']

        # ---------------------------
        # add buttons and ui elements
        # ---------------------------
        self.btnOpen = QPushButton('Open')
        self.btnOpen.clicked.connect(self.openNew)

        self.btnLoad = QPushButton('Load')
        self.btnLoad.clicked.connect(self.loadROIs)

        self.btnSave = QPushButton('Save')
        self.btnSave.clicked.connect(self.saveData)

        self.btnOverlay = QPushButton('Overlay')
        #self.btnOverlay.clicked.connect(self.setOverlay)

        # quick export
        self.btnExport = QPushButton('[^]')
        self.btnExport.clicked.connect(self.quickExport)

        # steppers
        self.buttonPrev = QPushButton('<')
        self.buttonPrev.clicked.connect(self.prevImage)
        #self.buttonPrev.setEnabled(False)

        self.buttonNext = QPushButton('>')
        self.buttonNext.clicked.connect(self.nextImage)
        #self.buttonNext.setEnabled(False)

        # options
        self.buttonOptions=QPushButton('...')
        self.buttonOptions.setToolTip('Show options')
        #self.buttonOptions.clicked.connect(self.)

        self.btnColours=QPushButton('Colours')
        self.btnColours.setToolTip('Set colour for annotations or overlay')
        #self.btnColours.clicked.connect(self.)

        # checkboxes
        # edit mode
        self.chkEdit = QCheckBox('Edit mode')
        self.chkEdit.setChecked(False)
        self.chkEdit.setToolTip('Allows switching to contour edit mode. Select with mouse click, accept with "q".')
        self.chkEdit.stateChanged.connect(self.setEditMode)

        # add auto mode
        self.chckbxAddAutomatically = QCheckBox('Add automatically')
        self.chckbxAddAutomatically.setChecked(True)
        self.chckbxAddAutomatically.setEnabled(False)
        self.chckbxAddAutomatically.setToolTip('Adds contours to annotations, always active (used in the ImageJ version)')
        self.chckbxAddAutomatically.setStyleSheet("color: gray")
        # smooth mode
        self.chkSmooth = QCheckBox('Smooth')
        self.chkSmooth.setToolTip('Applies smoothing to contour')
        self.chkSmooth.setChecked(False)
        #self.chkSmooth.stateChanged.connect(self.setSmooth)
        # show contours
        self.chkShowContours = QCheckBox('Show contours')
        self.chkShowContours.setChecked(True)
        self.chkShowContours.stateChanged.connect(self.showCnt)
        # assist mode
        self.chckbxContourAssist = QCheckBox('Contour assist')
        self.chckbxContourAssist.setChecked(False)
        self.chckbxContourAssist.setToolTip('Helps fit contour to object boundaries. Press \"q\" to add contour after correction. Press Ctrl+\"delete\" to delete suggested contour. (You must press either before you could continue!)')
        self.chckbxContourAssist.stateChanged.connect(self.setContourAssist)
        # show overlay
        self.chkShowOverlay = QCheckBox('Show overlay')
        self.chkShowOverlay.setChecked(False)
        #self.chkShowOverlay.stateChanged.connect(self.showOverlay)
        # class mode
        self.chckbxClass = QCheckBox('Class mode')
        self.chckbxClass.setChecked(False)
        self.chckbxClass.stateChanged.connect(self.setClassMode)
        self.chckbxClass.setToolTip('Allows switching to contour classification mode. Select with mouse click.')


        # add labels
        self.roiLabel=QLabel('ROIs')
        self.lblCurrentFile=QLabel('(1/1) [image name]')

        self.logo=QLabel()
        max_size=QSize(250,250)
        pixmap=QPixmap(os.path.join(os.path.dirname(__file__),'icon',self.logoFile+'.svg'))
        scaled=pixmap.scaled(max_size,Qt.KeepAspectRatio,Qt.SmoothTransformation)
        self.logo.setPixmap(scaled)

        bsize=int(self.btnOpen.size().width())
        self.bsize2=70
        labelsize=120

        # set button sizes
        self.btnOpen.setStyleSheet(f"max-width: {self.bsize2}px")
        self.btnLoad.setStyleSheet(f"max-width: {self.bsize2}px")
        self.btnSave.setStyleSheet(f"max-width: {self.bsize2}px")
        self.btnExport.setStyleSheet(f"max-width: {self.bsize2}px")
        self.btnOverlay.setStyleSheet(f"max-width: {self.bsize2}px")
        self.buttonPrev.setStyleSheet(f"min-width: {int(self.bsize2/2)}px;")
        self.buttonNext.setStyleSheet(f"min-width: {int(self.bsize2/2)}px;")
        self.buttonOptions.setStyleSheet(f"max-width: {self.bsize2}px")
        self.btnColours.setStyleSheet(f"max-width: {self.bsize2}px")
        self.lblCurrentFile.setStyleSheet(f"width: {labelsize}px")

        # set layouts
        self.mainVbox=QVBoxLayout()
        self.hBoxLogo=QHBoxLayout()
        self.hBoxTitle=QHBoxLayout()
        self.hBoxUp=QHBoxLayout()
        self.hBoxDown=QHBoxLayout()
        self.hBoxDownInnerLeft=QHBoxLayout()
        self.hBoxDownInnerRight=QHBoxLayout()
        self.hBoxDownCont=QHBoxLayout()
        self.vBoxDownCont=QVBoxLayout()
        self.vBoxDown=QVBoxLayout()
        self.vBoxLeft=QVBoxLayout()
        self.hBoxRight=QHBoxLayout()
        self.vBoxRightReal=QVBoxLayout()
        self.vBoxRightDummy=QVBoxLayout()

        self.hBoxLogo.addWidget(self.logo)
        self.hBoxTitle.addWidget(self.roiLabel)

        self.hBoxLogo.setAlignment(Qt.AlignCenter)

        self.vBoxLeft.setAlignment(Qt.AlignTop)
        self.vBoxLeft.addWidget(self.chckbxAddAutomatically)
        self.vBoxLeft.addWidget(self.chkSmooth)
        self.vBoxLeft.addWidget(self.chkShowContours)
        self.vBoxLeft.addWidget(self.chckbxContourAssist)
        self.vBoxLeft.addWidget(self.chkShowOverlay)
        self.vBoxLeft.addWidget(self.chkEdit)
        self.vBoxLeft.addWidget(self.chckbxClass)

        # add dummy buttons as spacers
        self.vBoxRightDummy.setAlignment(Qt.AlignTop)
        self.vBoxRightDummy.addSpacing(54)
        self.vBoxRightDummy.addWidget(self.btnExport)
        
        self.vBoxRightReal.setAlignment(Qt.AlignTop)
        self.vBoxRightReal.addWidget(self.btnOpen)
        self.vBoxRightReal.addWidget(self.btnLoad)
        self.vBoxRightReal.addWidget(self.btnSave)
        self.vBoxRightReal.addWidget(self.btnOverlay)

        self.hBoxDownInnerLeft.addWidget(self.lblCurrentFile)
        self.hBoxDownInnerRight.addWidget(self.buttonPrev)
        self.hBoxDownInnerRight.addWidget(self.buttonNext)
        self.hBoxDownInnerLeft.setAlignment(Qt.AlignLeft)
        self.hBoxDownInnerRight.setAlignment(Qt.AlignRight)
        self.hBoxDown.addLayout(self.hBoxDownInnerLeft)
        self.hBoxDown.addSpacing(3)
        self.hBoxDown.addLayout(self.hBoxDownInnerRight)

        self.vBoxDown.addWidget(self.buttonOptions)
        self.vBoxDown.addWidget(self.btnColours)
        self.vBoxDownCont.setAlignment(Qt.AlignBottom)
        self.vBoxDownCont.addLayout(self.hBoxDown)

        self.hBoxDownCont.setAlignment(Qt.AlignBottom)
        self.hBoxDownCont.addLayout(self.vBoxDownCont)
        self.hBoxDownCont.addLayout(self.vBoxDown)

        self.hBoxRight.addLayout(self.vBoxRightDummy)
        self.hBoxRight.addLayout(self.vBoxRightReal)

        self.hBoxUp.addLayout(self.vBoxLeft)
        self.hBoxUp.addLayout(self.hBoxRight)
        self.mainVbox.addLayout(self.hBoxLogo)
        self.mainVbox.addLayout(self.hBoxTitle)
        self.mainVbox.addLayout(self.hBoxUp)
        #self.mainVbox.addLayout(self.hBoxDown)
        self.mainVbox.addLayout(self.hBoxDownCont)

        self.setLayout(self.mainVbox)

        '''
        self.setLayout(QHBoxLayout())
        self.layout().addWidget(self.btnOpen)
        self.layout().addWidget(self.btnLoad)
        self.layout().addWidget(self.btnSave)
        self.layout().addWidget(self.btnExport)
        self.layout().addWidget(self.chkEdit)
        '''

        # greeting
        #print('AnnotatorJ plugin is started | Happy annotations!')
        print('----------------------------\nAnnotatorJ plugin is started\nHappy annotations!\n----------------------------')

        self.startUnet()

    def openNew(self):
        # temporarily open a test image
        # later this will start a browser dialog to select the input image file
        self.editMode=False

        if self.testMode==True:
            if os.path.exists(self.test_image):
                img=skimage.io.imread(self.test_image)
                print('Test image read successfully')
            else:
                print('Test image could not be found')
        else:
            # browse an original image

            # closing & saving previous annotations moved to separate fcn
            self.closeWindowsAndSave()

            # TODO: add checkbox checks

            # check contour assist setting
            if self.contAssist:
                self.addAuto=False
                self.chckbxAddAutomatically.setEnabled(False)
                print('< contour assist mode is active')
                self.editMode=False

                self.classMode=False
                self.chckbxClass.setEnabled(False)
                self.chckbxClass.setStyleSheet("color: gray")

            # check edit mode setting
            if self.editMode:
                self.addAuto=False
                self.chckbxAddAutomatically.setEnabled(False)
                print('< edit mode is active')
                self.contAssist=False
                self.chckbxContourAssist.setEnabled(False)
                self.chckbxContourAssist.setStyleSheet("color: gray")

                self.classMode=False
                self.chckbxClass.setEnabled(False)
                self.chckbxClass.setStyleSheet("color: gray")


            if not self.imageFromArgs:
                if self.stepping:
                    # concatenate file path with set new prev/next image name and open it without showing the dialog
                    destNameRaw=os.path.join(self.defDir,self.defFile)
                    self.stepping=False
                else:
                    destNameRaw,_=QFileDialog.getOpenFileName(
                        self,"Select an image",
                        str(os.path.join(self.defDir,self.defFile)),"Images (*.png *.bmp *.jpg *.jpeg *.tif *.tiff *.gif)")

                print(destNameRaw)
                if os.path.exists(destNameRaw):
                    self.defDir=os.path.dirname(destNameRaw)
                    self.defFile=os.path.basename(destNameRaw)
                    img=skimage.io.imread(destNameRaw)
                    print('Opened file: {}'.format(destNameRaw))
                else:
                    print('Could not open file: {}'.format(destNameRaw))
                    return

                self.curPredictionImageName=self.defFile
                self.curPredictionImage=None
                self.curOrigImage=None

        imageLayer = self.viewer.add_image(img,name='Image')

        # check if a shapes layer already exists for the rois
        # if so, bring it forward
        roiLayer=self.findROIlayer(True)
        if roiLayer is None:
            # create new ROI layer if none present
            self.initRoiManager()
        self.viewer.reset_view()

        self.curFileList=[f for f in os.listdir(self.defDir) if os.path.isfile(os.path.join(self.defDir,f)) and os.path.splitext(f)[1] in self.imageExsts]
        fileListCount=len(self.curFileList)

        # find current file in the list
        try:
            self.curFileIdx=self.curFileList.index(self.defFile)
        except ValueError:
            print('Could not find the currently selected file name in the list of image files')


        # update file name tag on main window to check which image we are annotating
        displayedName=self.defFile
        maxLength=13 #13
        # check how long the file name is (if it can be displayed)
        nameLength=len(self.defFile)
        if nameLength>maxLength:
            nameSplit=os.path.splitext(self.defFile)
            displayedName=self.defFile[:min(maxLength-3,len(nameSplit[0]))]+'..'+nameSplit[1]
        
        self.lblCurrentFile.setText(' ('+str(self.curFileIdx+1)+'/'+str(fileListCount)+'): '+displayedName)

        # MOVING FCN mods:
        #self.lblCurrentFile.addMouseListener

        # inactivate prev/next buttons if needed
        if self.curFileIdx==0:
            # first image in folder, inactivate prev:
            self.buttonPrev.setEnabled(False)
        else:
            self.buttonPrev.setEnabled(True)

        if self.curFileIdx==len(self.curFileList)-1:
            # last image, inactivate next:
            self.buttonNext.setEnabled(False)
        else:
            self.buttonNext.setEnabled(True)


        # TODO: add missing settings


        # reset contour assist layer
        if self.contAssist:
            self.setContourAssist(Qt.Checked)
        else:
            self.setContourAssist(False)

        # when open function finishes:
        self.started=True


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
            if not self.started or (self.findImageLayer() is None or self.findImageLayer().data is None):
                warnings.warn('Open an image and annotate it first')
                return

            # check if we have annotations in the list before loading anything to it
            roiLayer=self.findROIlayer()
            curROInum=len(roiLayer.data) if roiLayer is not None else 0
            print('Before loading we had '+str(curROInum)+' contours');
            prevROIcount=curROInum
            if self.loadedROI:
                # currently the loaded rois are appended to the current roi list
                # TODO: ask if those should be deleted first
                pass

            # check if masks can be loaded (false by default)
            if self.enableMaskLoad:
                # TODO
                pass

            else:
                # normal way, import ROI.zip file
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


                #self.add2RoiManager(rois)
                shapesLayer=self.extractROIdata(rois)
                self.viewer.add_layer(shapesLayer)
                self.findROIlayer(True)
                print('Loaded {} ROIs successfully'.format(len(rois)))

                # select the "select shape" mode from the controls by default
                #shapesLayer.mode = 'select'
                # select the "add polygon" mode from the controls by default to enable freehand ROI drawing
                shapesLayer.mode = 'add_polygon'

                self.addFreeROIdrawing(shapesLayer)
                self.addKeyBindings(shapesLayer)

                self.viewer.reset_view()

            self.loadedROI=True
            roiLayer=self.findROIlayer()
            curROInum=len(roiLayer.data)
            print('After loading we have '+str(curROInum)+' contours');

            # rename the loaded contours if there were previous contours added
            # TODO

            # check if the rois have class info saved
            isClassified=False
            for idx,r in enumerate(roiLayer.data):
                if roiLayer.properties['class'][idx]>0:
                    isClassified=True
                    break

            if isClassified:
                # set class vars accordingly
                # TODO: set the vars
                self.startedClassifying=True


    def loadROIs2(self):
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

        #self.add2RoiManager(rois)
        shapesLayer=self.findROIlayer(True)
        self.addROIdata(shapesLayer,rois)
        print('Loaded {} ROIs successfully'.format(len(rois)))

        # select the "select shape" mode from the controls by default
        #shapesLayer.mode = 'select'
        # select the "add polygon" mode from the controls by default to enable freehand ROI drawing
        shapesLayer.mode = 'add_polygon'

    def initRoiManager(self):
        # the rois will be stored in this object as in ImageJ's RoiManager
        self.manager=None
        roiProps={'name':['0001'],'class':[0],'nameInt':[1]}
        roiTextProps={
            'text': '{nameInt}: ({class})',
            'anchor': 'center',
            'size': 10,
            'color': 'black',
            'visible':False
        }
        # add an empty shapes layer
        shapesLayer=Shapes(data=numpy.array([[0,0],[1,1]]),shape_type='polygon',name='ROI',edge_width=0.5,edge_color='white',face_color=[0,0,0,0],properties=roiProps,text=roiTextProps)
        self.viewer.add_layer(shapesLayer)
        # remove dummy shape used to init text props and its visibility
        shapesLayer._data_view.remove(0)
        # select the "select shape" mode from the controls by default
        #shapesLayer.mode = 'select'
        # select the "add polygon" mode from the controls by default to enable freehand ROI drawing
        shapesLayer.mode = 'add_polygon'

        '''
        # mock a dummy shape adding for properties
        roiProps['class'].append(0)
        roiProps['name'].append(1)
        roiProps['nameInt'].append('0001')
        shapesLayer=Shapes(data=[numpy.array([[0,0], [1,1]])],shape_type='polygon',name='ROI',edge_width=0.5,edge_color='white',face_color=[0,0,0,0],properties=roiProps,text=roiTextProps)
        # select the "select shape" mode from the controls by default
        shapesLayer.mode = 'select'
        # select and remove this dummy shape
        shapesLayer.selected_data={0}
        shapesLayer.remove_selected()
        self.viewer.add_layer(shapesLayer)
        '''

        # 
        self.addFreeROIdrawing(shapesLayer)
        # TODO

    def add2RoiManager(self,rois):
        # store all ROIs in a set as in ImageJ
        if self.manager is None:
            self.manager=rois
            self.roiCount=len(rois)
        elif isinstance(rois,list):
            # list of rois, add them to the current rois
            if isinstance(self.manager,ImagejRoi):
                # 1 roi object so far
                self.manager=[self.manager]+rois
            else:
                print(self.manager)
                self.manager=self.manager+rois
            self.roiCount+=len(rois)
        elif isinstance(rois,ImagejRoi):
            # 1 roi object to add
            if isinstance(self.manager,ImagejRoi):
                # 1 roi object so far
                self.manager=[self.manager]+[rois]
            else:
                self.manager=self.manager+[rois]
            self.roiCount+=len(rois)

    def fetchShapes2ROIs(self):
        # store all ROIs on the ROI shape layer in a set as in ImageJ
        roiLayer=self.findROIlayer()
        # loop through all shapes
        n=len(roiLayer.data)
        if n==0:
            # no shapes yet
            return None

        rois=[]
        #for roi,roiType,name,classIdx in zip(roiLayer.data,roiLayer.shape_type,roiLayer.properties['name'],roiLayer.properties['class']):
        for i in range(n):
            # swap (x,y) coordinates back
            xy=roiLayer.data[i] # a list of (x,y) coordinates in the wrong order
            yx=numpy.array([[y,x] for x,y in xy]) # swapping to (y,x) coordinates
            newRoi=ImagejRoi.frompoints(yx)

            t=roiLayer.shape_type[i]
            if t=='polygon':
                newRoi.roitype=ROI_TYPE.FREEHAND
            elif t=='rectangle':
                newRoi.roitype=ROI_TYPE.RECT
            newRoi.name=roiLayer.properties['name'][i]
            newRoi.group=roiLayer.properties['class'][i]

            # find edge colour
            curColour=roiLayer.edge_color[i] # e.g. array([1., 0., 0., 1.]) is red
            newRoi.stroke_color=bytes.fromhex(self.rgb2Hex(curColour))

            # check version before saving; >=228 stores group attribute upon reading in ImageJ
            if newRoi.version<228:
                newRoi.version=228

            rois.append(newRoi)

        # return a list of ImagejRoi objects
        return rois


    def extractROIdata(self,rois):
        # fetch the coordinates and other data from the ImageJ ROI.zip file already imported with the roifile package
        # Inputs:
        #   rois: list of ImageJ ROI objects
        # Outputs:
        #   shapesLayer: shapes layer created from the list of coordinates fetched from the input rois

        roiList=[]
        roiType='polygon' # default to this
        #self.defColour='white'

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
            elif (curROI.roitype==ROI_TYPE.RECT and yx.shape[0]==4):
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
                roiColours.append(self.defColour)
                roiProps['class'].append(0)

            # store the roi's name
            roiProps['name'].append(curROI.name)
            roiProps['nameInt'].append(int(curROI.name))

            # TODO: fetch more data from the rois

        # rename any existing ROI layers so that this one is the new default
        self.renameROIlayers()

        # fill (face) colour of rois is transparent by default, only the contours are visible
        # edge_width=0.5 actually sets it to 1
        shapesLayer = Shapes(data=roiList,shape_type=roiType,name='ROI',edge_width=0.5,edge_color=roiColours,face_color=[0,0,0,0],properties=roiProps,text=roiTextProps)

        return shapesLayer

    def initClassColourLUT(self,rois):
        # setup a colour lut
        # loop through all ROIs and assign colours by classes
        classes=[]
        for roi in rois:
            classes.append(roi.group)
        # find the unique class indexes
        classIdxs=numpy.unique(classes)
        # get a list of the 9 basic colours also present in AnnotatorJ's class mode
        #colours=['red','green','blue','cyan','magenta','yellow','orange','white','black']
        # TODO: add much more colours!

        self.classColourLUT={}
        for x in classIdxs:
            self.classColourLUT.update({x:self.colours[x-1]}) # classes are only considered when class>0

    def rgb2Hex(self,rgb):
        # convert an array of rgb values to hex string representing colour
        if rgb.dtype=='float64':
            rgb=rgb*255
        # ignore alpha
        #rgb=rgb[:3]
        s=''
        if len(rgb)==4:
            # with alpha
            s=s+('%02x' % int(rgb[-1]))

        for idx,k in enumerate(rgb):
            if idx>2:
                break
            s=s+('%02x' % int(k))
        print(s)
        return s


    def findROIlayer(self,setLayer=False,layerName='ROI',quiet=False):
        for x in self.viewer.layers:
            if (x.__class__ is Shapes and x.name==layerName):
                if not quiet:
                    print('{} is the ROI shapes layer'.format(x.name))
                # bring it to the front
                n=len(self.viewer.layers)
                xLayerIdx=self.viewer.layers.index(x)
                if xLayerIdx!=n-1:
                    # need to move it
                    self.viewer.layers.selection.clear()
                    #self.viewer.layers.selection={x}
                    self.viewer.layers.move_selected(xLayerIdx, n-1) # n-1
                if setLayer:
                    self.roiLayer=x
                return x
            else:
                pass
        # log if the ROI layer could not be found
        print('Could not find the ROI layer')
        return None

    def renameROIlayers(self):
        for x in self.viewer.layers:
            if (x.__class__ is Shapes and x.name=='ROI'):
                print('{} was a ROI shapes layer'.format(x.name))
                # rename it
                newName='ROI_prev'
                k=0
                while newName in self.viewer.layers:
                    k+=1
                    newName=f'{newName}_{k}'
                x.name=newName
                return None
            else:
                pass


    def findImageLayer(self,echo=True):
        for x in reversed(self.viewer.layers):
            if (x.__class__ is Image):
                if echo:
                    print('{} is the uppermost image layer'.format(x.name))
                # return it
                return x
            else:
                pass
        # log if the Image layer could not be found
        print('Could not find the Image layer')


    def findImageLayerName(self,echo=True,layerName='Image'):
        for x in reversed(self.viewer.layers):
            if (x.__class__ is Image and x.name==layerName):
                if echo:
                    print('{} is the uppermost image layer'.format(x.name))
                # return it
                return x
            else:
                pass
        # log if the Image layer could not be found
        print('Could not find the Image layer')


    def findLabelsLayerName(self,echo=True,layerName='editing'):
        for x in reversed(self.viewer.layers):
            if (x.__class__ is Labels and x.name==layerName):
                if echo:
                    print('{} is the uppermost labels layer'.format(x.name))
                # return it
                return x
            else:
                pass
        # log if the Image layer could not be found
        print('Could not find the Labels layer')


    def addROIdata(self,layer,rois):
        roiType='polygon' # default to this
        #self.defColour='white'

        hasColour=False
        roiProps={'name':[],'class':[],'nameInt':[]}
        roiTextProps={
            'text': '{nameInt}: ({class})',
            'anchor': 'center',
            'size': 10,
            'color': 'black',
            'visible':False
        }
        # remember props from previous rois
        prevProps=layer.properties
        if prevProps=={}:
            # init
            prevProps={'name':[],'class':[],'nameInt':[]}

        # loop through the rois
        for curROI in rois:
            xy=curROI.coordinates() # a list of (x,y) coordinates in the wrong order
            yx=numpy.array([[y,x] for x,y in xy]) # swapping to (y,x) coordinates

            # check roi type
            if curROI.roitype==ROI_TYPE.FREEHAND:
                # freehand roi drawn in instance annotation mode
                roiType='polygon'
            elif (curROI.roitype==ROI_TYPE.RECT and yx.shape[0]==4):
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
                roiProps['class']=curClass
            else:
                curColour=self.defColour
                roiProps['class']=0

            # store the roi's name
            roiProps['name']=curROI.name
            roiProps['nameInt']=int(curROI.name)

            # fill (face) colour of rois is transparent by default, only the contours are visible
            # edge_width=0.5 actually sets it to 1
            layer.add(data=yx,shape_type=roiType,edge_width=0.5,edge_color=curColour,face_color=[0,0,0,0])#,properties=roiProps,text=roiTextProps)

            # TODO: fetch more data from the rois

            # add the new roi's props to the object:
            prevProps['class'].append(roiProps['class'])
            prevProps['name'].append(roiProps['name'])
            prevProps['nameInt'].append(roiProps['nameInt'])

        # set the props after adding the rois
        layer.properties=prevProps


        

    def setTestMode(self,mode=False):
        self.testMode=mode

    def saveData(self):
        # open a save dialog and save the rois to an imagej compatible roi.zip file
        self.finishedSaving=False
        if not self.started or (self.findImageLayer() is None or self.findImageLayer().data is None):
            warnings.warn('Open an image and annotate it first')
            if self.stepping:
                self.finishedSaving=True
            return

        print('saving...')
        if self.stepping:
            self.finishedSaving=False

        # TODO: rename rois

        if not self.imageFromArgs: # || origMaskFileNames==null || !roisFromArgs

            if self.startedClassifying:
                # see if any object was actually classified
                isClassified=False
                roiLayer=self.findROIlayer()
                for idx,r in enumerate(roiLayer.data):
                    if roiLayer.properties['class'][idx]>0:
                        isClassified=True
                        break

                if isClassified:
                    # no need for class selection dialog box
                    selectedClass='masks'

                else:

                    # ask class name in dialog box
                    # TODO
                    selectedClass='masks'
            else:
                # create new frame for optional extra element adding manually by the user (for new custom class):
                # TODO
                selectedClass='masks'
                pass

            
        else:
            # roi stack was imported, save to mask names
            # TODO: do this branch
            selectedClass='masks'
            pass


        # set output folder and create it
        destMaskFolder2=os.path.join(self.defDir,selectedClass)
        os.makedirs(destMaskFolder2,exist_ok=True)
        print('Created output folder: {}'.format(destMaskFolder2))

        # set output file name according to annotation type:
        # TODO: add the others
        # now we only have instance
        roiFileName=str(os.path.join(destMaskFolder2,'{}_ROIs.zip'.format(os.path.splitext(self.defFile)[0])))

        # check if annotation already exists for this image with this class
        if os.path.exists(roiFileName):
            # TODO: pop dialog to overwrite,rename,cancel
            newFileNum=0
            while os.path.exists(roiFileName):
                newFileNum+=1
                roiFileName=str(os.path.join(destMaskFolder2,'{}_ROIs_{}.zip'.format(os.path.splitext(self.defFile)[0],newFileNum)))

        print('Set output ROI.zip name: {}'.format(roiFileName))

        # TODO: prepare the roi files from the shapes layers
        # viewer.layers[1].__class__ should yield napari.layers.shapes.shapes.Shapes

        # save it

        # orig way with manager
        '''
        if self.manager is not None:
            print(self.manager)
            #outROI = self.manager.tobytes()
            #outROI.tofile(roiFileName)
            roiwrite(roiFileName,self.manager)
            print('Saved ROI: {}'.format(roiFileName))
        else:
            print('Failed to save ROI: {}'.format(roiFileName))
        '''

        # new way with roi list
        rois2save=self.fetchShapes2ROIs()
        if rois2save is None:
            # nothing to save
            print('Failed to save ROI: {}'.format(roiFileName))
        else:
            roiwrite(roiFileName,rois2save)
            print('Saved ROI: {}'.format(roiFileName))


        print('finished saving')
        self.finishedSaving=True



    # add mouse event handler for free roi drawing on the shapes layer
    def addFreeROIdrawing(self,shapesLayer=None):
        if shapesLayer is not None:
            shapesLayer.events.data.connect(self.updateNewROIprops,position='last')
            shapesLayer.mouse_drag_callbacks.append(self.freeHandROI)
            shapesLayer.mouse_drag_callbacks.append(self.editROI)
        else:
            return
        return

        '''
        for x in self.viewer.layers:
            if x.__class__ is not Shapes:
                pass
            else:
                print('{} is a shapes layer'.format(x.name))
                # add listeners to this layer

                # add a listener to mouse release events when a new shape was just added
                x.events.data.connect(self.updateNewROIprops,position='last')

                # add a listener to mouse drags in polygon adding mode to mimic freehand roi drawing
                @x.mouse_drag_callbacks.append
        '''


    def addFreeROIdrawingCA(self,shapesLayer=None):
        if shapesLayer is not None:
            #shapesLayer.events.data.connect(self.contAssistROI,position='last')
            shapesLayer.mouse_drag_callbacks.append(self.freeHandROI)
            shapesLayer.mouse_drag_callbacks.append(self.editROI)
        else:
            return
        return


    def freeHandROI(self,layer, event):
        #data_coordinates = layer.world_to_data(event.position)
        #cords = numpy.round(data_coordinates).astype(int)
        #print('Clicked at {} on layer {}'.format(cords,layer.name))
        yield
        #layer.data[0] is an array of points in the shape

        if layer.mode=='add_polygon':
            dragged=False
            freeCoords=[]
            #self.defColour='white'
            # on move
            while event.type == 'mouse_move':
                dragged = True
                coords = list(layer.world_to_data(event.position))
                freeCoords.append(coords)
                yield
            # on release
            if dragged:
                # drag ended
                # add closing position:
                coords = list(layer.world_to_data(event.position))
                freeCoords.append(coords)
                # mimic an 'esc' key press to quit the basic add_polygon method
                key_bindings.finish_drawing_shape(layer)
                #print(freeCoords)
                # add the coords as a new shape
                layer.add(data=freeCoords,shape_type='polygon',edge_width=0.5,edge_color=self.defColour,face_color=[0,0,0,0])
                if self.contAssist and not self.inAssisting:
                    self.contAssistROI()
            # else: do nothing
            #    print('clicked!')
        # else: do nothing
        #    print('---- not in adding mode ----')

    # this does not work yet:
    #@x.mouse_drag_callbacks.append
    def freeHandROIvis(layer, event):
        yield
        if layer.mode=='add_polygon':
            dragged=False
            #self.defColour='white'
            # on move
            while event.type == 'mouse_move':
                dragged = True
                coords = list(layer.world_to_data(event.position))
                # this is not working yet:
                mouse_bindings.vertex_insert(layer,event)
                yield
            # on release
            if dragged:
                # drag ended
                # mimic an 'esc' key press to quit the basic add_polygon method
                key_bindings.finish_drawing_shape(layer)
            # else: do nothing
            #    print('clicked!')
        # else: do nothing
        #    print('---- not in adding mode ----')


    # add a listener to clicks on shapes when in edit mode
    #@x.mouse_drag_callbacks.append
    def editROI(self,layer,event):
        # only do something when in edit mode
        if not self.editMode:
            return
        if layer.mode=='add_polygon' and not self.startedEditing:
            msg='Cannot start editing when {} is selected. Please select {}'.format(
                '\'Add polygons(P)\'','\'Select shapes(5)\'')
            warnings.warn(msg)
            return
        elif layer.mode!='select' and not self.startedEditing:
            msg='Cannot start editing. Please select {}'.format('\'Select shapes(5)\'')
            warnings.warn(msg)
            return
        elif self.startedEditing:
            msg='Already started editing a contour'
            warnings.warn(msg)
            return

        yield

        # start edit mode
        self.startedEditing=True

        pos=layer.world_to_data(event.position)

        # get the image size
        imageLayer=self.findImageLayer(echo=False)
        if imageLayer is None:
            print('No image layer found')
            self.startedEditing=False
            return
        s=imageLayer.data.shape
        self.imgSize=s
        
        if pos[0]<=0 or pos[1]<=0 or pos[0]>s[1] or pos[1]>s[0]:
            print('(Edit mode) not on the image')
            self.startedEditing=False
            self.origEditedROI=None
            self.editROIidx=-1
        else:
            print('(Edit mode) click on {}'.format(pos))

            roiLayer=self.findROIlayer()
            if roiLayer is None:
                return

            # check if the user clicked on a shape
            if len(roiLayer.selected_data)>0:
                # clicked on a shape
                # get the index of the shape and remove the selection in one go
                curIdx=roiLayer.selected_data.pop()
                #print('Selected {}. roi: {}'.format(curIdx,roiLayer.properties['name'][curIdx]))
                print('Selected \'{}\' ROI for editing'.format(roiLayer.properties['name'][curIdx]))

                curColour=roiLayer._data_view._edge_color[curIdx]

                # store the orig properties of this shape so after editing is finished it can be restored
                self.origEditedROI=BackupROI(roiLayer.data[curIdx],idx=curIdx,edgeColour=deepcopy(curColour),edgeWidth=deepcopy(roiLayer._data_view.edge_widths[curIdx]))
                self.editROIidx=curIdx

                # change the edge to show it is the selected one
                invColour=self.invertColour(curColour)
                roiLayer._data_view.update_edge_color(curIdx,invColour)
                roiLayer._data_view.update_edge_width(curIdx,2)
                roiLayer.refresh()
                #roiLayer.events.edge_color()
                #roiLayer.events.edge_width()
                
                # make a temp labels layer for editing the selected shape
                # copy the selected shape to a temp shapes layer then convert to a labels layer
                shapesLayer=Shapes(data=roiLayer.data[curIdx],shape_type='polygon',name='ROI tmp',edge_width=2,edge_color=invColour,face_color=[0,0,0,0])
                self.viewer.add_layer(shapesLayer)
                roiLayerCopy=self.findROIlayer(layerName='ROI tmp')
                labels = roiLayerCopy.to_labels([s[0], s[1]])
                # delete this temp shape layer
                self.viewer.layers.remove(shapesLayer)
                roiLayer.visible=False
                # convert to labels layer
                labelLayer = self.viewer.add_labels(labels, name='editing')

                roiLayer.visible=True


                # set the tool for an editing-capable one
                #roiLayer.mode='add_polygon';
                labelLayer.mode='paint'
                labelLayer.brush_size=self.brushSize
                labelLayer.opacity=0.5

                # add a modifier to the paint tool to erase when 'alt' is held
                labelLayer.mouse_drag_callbacks.insert(0,self.eraseBrush2)

                # bind the shortcut 'q' to acceptEdit function
                # 'ctrl+q' is by default bound to exit, so no ctrl here
                labelLayer.bind_key('q',func=self.acceptEdit)
                #labelLayer.bind_key('q',func=self.warnMissingCtrl)
                labelLayer.bind_key('Escape',func=self.rejectEdit)
                labelLayer.bind_key('Control-Delete',func=self.deleteEdit)

            else:
                print('Could not find the ROI associated with the selected point on the image.')
                self.startedEditing=False
                self.origEditedROI=None
                self.editROIidx=-1


    

    def invertColour(self,origColour):
        # invCol is like numpy.array([0.2,0.,0.,1.])
        invCol=numpy.array([1.,1.,1.,1.])-origColour
        # the last element is alpha, keep it on 100% so it is visible
        invCol[-1]=1.
        return invCol


    def addKeyBindings(self,layer):
        #viewer=self.viewer

        '''
        # "q" in editMode will accept the contour edit
        #@viewer.bind_key('q')
        @layer.bind_key('q')
        def bindQ(layer,event):
        #def bindQ(viewer,event):
            if self.startedEditing:
                self.acceptEdit(layer,event)

            def acceptEdit(viewer):
                pass
                # self._data_view.edit(index, vertices[:-1]) <-- new data
            yield
        '''


    def acceptEdit(self,labelLayer):
        if not self.startedEditing:
            print('Cannot accept edited contour when not \'startedEditing\'')
            return
        yield

        print('Q pressed - updating edited contour')
        # convert the edited shape on the temp labels layer to shape
        mask=labelLayer.data
        contour,hierarchy=cv2.findContours(mask.astype(numpy.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contour:
            # not empty
            if len(contour)>1:
                # fill holes to create 1 object
                if self.imgSize is not None:
                    from scipy.ndimage import binary_fill_holes
                    filled=numpy.zeros((self.imgSize[0],self.imgSize[1]),dtype=numpy.uint8)
                    binary_fill_holes(mask,output=filled)
                    contour,hierarchy=cv2.findContours(filled.astype(numpy.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    if not (contour and len(contour)==1):
                        msg='Cannot create roi from this label'
                        warnings.warn(msg)
                        return None
                else:
                    msg='Cannot create roi from this label'
                    warnings.warn(msg)
                    return None

            shape=numpy.array(numpy.fliplr(numpy.squeeze(contour)))
            roiLayer=self.findROIlayer()
            roiLayer._data_view.edit(self.editROIidx,shape)
            # reset colour and width
            roiLayer._data_view.update_edge_color(self.editROIidx,self.origEditedROI.edgeColour)
            roiLayer._data_view.update_edge_width(self.editROIidx,self.origEditedROI.edgeWidth)
            roiLayer.refresh()
            print('Saved edited ROI')

            # store updated brush size
            self.brushSize=labelLayer.brush_size

            # clear everything
            self.cleanUpAfterEdit(labelLayer,roiLayer)

            return shape
        else:
            # no contour found
            print('Could not find the contour after editing')
            return None


    def rejectEdit(self,labelLayer):
        if not self.startedEditing:
            print('Cannot reject edited contour when not \'startedEditing\'')
            return
        yield

        print('Esc pressed - restoring original contour')

        roiLayer=self.findROIlayer()
        shape=self.origEditedROI.shape
        if shape is None:
            print('Failed to find previous version of this ROI, cannot revert to it')
        else:
            roiLayer._data_view.edit(self.editROIidx,shape)
            # reset colour and width
            roiLayer._data_view.update_edge_color(self.editROIidx,self.origEditedROI.edgeColour)
            roiLayer._data_view.update_edge_width(self.editROIidx,self.origEditedROI.edgeWidth)
            roiLayer.refresh()
            print('Restored edited ROI to its original')

        # store updated brush size
        self.brushSize=labelLayer.brush_size

        # clear everything
        self.cleanUpAfterEdit(labelLayer,roiLayer)


    def deleteEdit(self,labelLayer):
        if not self.startedEditing:
            print('Cannot delete current contour when not \'startedEditing\'')
            return
        yield

        print('Ctrl+delete pressed - deleting current contour')

        roiLayer=self.findROIlayer()
        #roiLayer._data_view.remove(self.editROIidx)
        roiLayer.selected_data={self.editROIidx}
        roiLayer.remove_selected()
        roiLayer.refresh()
        print('Restored edited ROI to its original')

        # store updated brush size
        self.brushSize=labelLayer.brush_size

        # clear everything
        self.cleanUpAfterEdit(labelLayer,roiLayer)


    def cleanUpAfterEdit(self,labelLayer,roiLayer):
        self.startedEditing=False
        self.editROIidx=-1
        self.origEditedROI=None

        # delete this label layer
        self.viewer.layers.remove(labelLayer)

        # bring the ROI layer forward
        self.viewer.layers.selection.add(roiLayer)

        if roiLayer.selected_data:
            roiLayer.selected_data.pop()
        roiLayer.mode='select'


    def warnMissingCtrl(self,layer):
        if self.editMode and self.startedEditing:
            # ctrl+q already bound to exit
            msg='missing Ctrl --> cannot update current contour'
            warnings.warn(msg)


    def eraseBrush(self,layer,event):
        if 'Alt' in event.modifiers:
            # erase instead of painting
            layer.selected_label=layer._background_label
            yield
        else:
            # reset to painting
            layer.selected_label=layer._background_label+1
            yield

    def eraseBrush2(self,layer,event):
        if 'Alt' in event.modifiers:
            # erase instead of painting
            if layer.mode=='paint':
                layer.mode='erase'
            yield
        else:
            # reset to painting
            if layer.mode=='erase':
                layer.mode='paint'
            yield


    def rejectEdit2(self,labelLayer):
        if not self.startedEditing:
            print('Cannot reject edited contour when not \'startedEditing\'')
            return
        #no need to yield, not an event callback

        print('Esc pressed - restoring original contour')

        roiLayer=self.findROIlayer()
        shape=self.origEditedROI.shape
        if shape is None:
            print('Failed to find previous version of this ROI, cannot revert to it')
        else:
            roiLayer._data_view.edit(self.editROIidx,shape)
            # reset colour and width
            roiLayer._data_view.update_edge_color(self.editROIidx,self.origEditedROI.edgeColour)
            roiLayer._data_view.update_edge_width(self.editROIidx,self.origEditedROI.edgeWidth)
            roiLayer.refresh()
            print('Restored edited ROI to its original')

        # store updated brush size
        self.brushSize=labelLayer.brush_size

        # clear everything
        self.cleanUpAfterEdit(labelLayer,roiLayer)


    def acceptContAssist(self,labelLayer):
        if not self.inAssisting:
            print('Cannot accept suggested contour when not \'inAssisting\'')
            return
        yield

        print('Q pressed - adding suggested contour')
        # convert the edited shape on the temp labels layer to shape
        mask=labelLayer.data
        contour,hierarchy=cv2.findContours(mask.astype(numpy.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contour:
            # not empty
            if len(contour)>1:
                # fill holes to create 1 object
                if self.imgSize is not None:
                    from scipy.ndimage import binary_fill_holes
                    filled=numpy.zeros((self.imgSize[0],self.imgSize[1]),dtype=numpy.uint8)
                    binary_fill_holes(mask,output=filled)
                    contour,hierarchy=cv2.findContours(filled.astype(numpy.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    if not (contour and len(contour)==1):
                        msg='Cannot create roi from this label'
                        warnings.warn(msg)
                        return None
                else:
                    msg='Cannot create roi from this label'
                    warnings.warn(msg)
                    return None

            shape=numpy.array(numpy.fliplr(numpy.squeeze(contour)))
            roiLayer=self.findROIlayer()
            roiLayer.add_polygons(shape,edge_color=self.defColour,face_color=None,edge_width=0.5)
            roiLayer.refresh()
            print('Added ROI ('+str(len(roiLayer.data))+'.) - assist mode')

            # store updated brush size
            self.brushSize=labelLayer.brush_size

            # clear everything
            self.cleanUpAfterContAssist(labelLayer,roiLayer)

            return shape
        else:
            # no contour found
            print('Could not find the contour after editing suggested contour')
            return None


    def deleteContAssist(self,labelLayer):
        if not self.inAssisting:
            print('Cannot delete suggested contour when not \'inAssisting\'')
            return
        yield

        print('Ctrl+delete pressed - deleting suggested contour')

        roiLayer=self.findROIlayer()

        # store updated brush size
        self.brushSize=labelLayer.brush_size

        # clear everything
        self.cleanUpAfterContAssist(labelLayer,roiLayer)


    def cleanUpAfterContAssist(self,labelLayer,roiLayer):
        # reset vars
        self.inAssisting=False
        self.invertedROI=None
        self.ROIpositionX=0
        self.ROIpositionY=0
        self.acObjects=None
        self.startedEditing=False
        self.origEditedROI=None

        # delete this label layer
        self.viewer.layers.remove(labelLayer)

        if roiLayer.selected_data:
            roiLayer.selected_data.pop()
        roiLayer.mode='add_polygon'

        contAssistLayer=self.addContAssistLayer()
        self.addFreeROIdrawingCA(contAssistLayer)
        contAssistLayer.mode='add_polygon'

        # bring the ROI layer forward
        self.viewer.layers.selection.add(contAssistLayer)


    def contAssistROI(self):
        # only do something when in contour assist mode
        if not self.contAssist:
            print('Contour assist not selected, cannot proceed')
            return

        roiLayer=self.findROIlayer(layerName='contourAssist')
        if roiLayer is None:
            print('No ROI layer found for contour assist (contourAssist)')
            return

        if roiLayer.mode=='select' and not self.inAssisting:
            msg='Cannot start contour assist when {} is selected. Please select {}'.format(
                '\'Select shapes(5)\'','\'Add polygons(P)\'')
            warnings.warn(msg)
            return
        elif roiLayer.mode!='add_polygon' and not self.inAssisting:
            msg='Cannot start contour assist. Please select {}'.format('\'Add polygons(P)\'')
            warnings.warn(msg)
            return
        elif self.inAssisting:
            # do nothing on mouse release
            print('---- already inAssisting ----')
            return

        #yield

        print('Suggesting improved contour...')

        # get the image size
        imageLayer=self.findImageLayer(echo=False)
        if imageLayer is None:
            print('No image layer found')
            return
        elif imageLayer.data is None:
            print('No image opened')
            return
        s=imageLayer.data.shape
        self.imgSize=s
        

        # get current selection as init contour
        curROI=roiLayer.data[-1] if len(roiLayer.data)>0 else None
        if curROI is None:
            print('Empty ROI')
        else:
            print('curROI:')
            print(curROI)
        
            # can start suggestions

            # first start freehand selection tool for drawing --> done
                # on mouse release start contour correction -->

            # contour correction
            newROI=None
            
            # setting unet model paths
            modelJsonFile=os.path.join(self.modelFolder,self.modelJsonFile) #"model_real.json"
            modelWeightsFile=os.path.join(self.modelFolder,self.modelWeightsFile) #"model_real_weights.h5"
            modelFullFile=os.path.join(self.modelFolder,self.modelFullFile) #"model_real.hdf5"
            try:
                if self.selectedCorrMethod==0:
                    # unet correction
                    # debug:
                    #print('  >> unet correction')
                    jsonFileName=None
                    modelFileName=None
                    if os.path.isfile(modelWeightsFile):
                        jsonFileName=modelJsonFile
                        modelFileName=modelWeightsFile
                    else:
                        jsonFileName=None
                        if os.path.isfile(modelFullFile):
                            modelFileName=modelFullFile
                        else:
                            # model doesn't exist in the located folder
                            print('Cannot find model in Contour assist init')
                        
                    # prepare the ROI as a cv2 contour
                    labels=roiLayer.to_labels([s[0], s[1]]) # was labels
                    # convert to labels layer
                    #labelLayer = self.viewer.add_labels(labels, name='tmp')
                    #curROIdata=labelLayer.data
                    #self.viewer.layers.remove(labelLayer)
                    curROIdata,hierarchy=cv2.findContours(labels.astype(numpy.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    # delete temp init roi
                    roiLayer.data=[]
                    newROI=self.contourAssistUNet(imageLayer.data,curROI,curROIdata[0],self.intensityThreshVal,self.distanceThreshVal,jsonFileName,modelFileName)
                elif self.selectedCorrMethod==1:
                    # region growing
                    # debug:
                    #print('  >> classical correction')
                    newROI=self.contourAssist(imageLayer.data,curROI,self.intensityThreshVal,self.distanceThreshVal)
                
            except Exception as ex:
                print(ex)
                self.invertedROI=None
                raise(ex)
            
            if newROI is None:
                # failed, return
                print('Failed suggesting a better contour')
                self.invertedROI=None
            else:
                # display this contour
                roiLayer.add_polygons(newROI)

                # succeeded, nothing else to do
                print('Showing suggested contour')

                # user can check it visually -->
                        # set brush selection tool for contour modification -->

                labels = roiLayer.to_labels([s[0], s[1]])
                # delete this temp shape layer
                self.viewer.layers.remove(roiLayer)
                #roiLayer.visible=False
                # convert to labels layer
                labelLayer = self.viewer.add_labels(labels, name='editing')

                #roiLayer.visible=True


                # set the tool for an editing-capable one
                #roiLayer.mode='add_polygon';
                labelLayer.mode='paint'
                labelLayer.brush_size=self.brushSize
                labelLayer.opacity=0.5

                # TODO: add callbacks like in editmode
                # add a modifier to the paint tool to erase when 'alt' is held
                labelLayer.mouse_drag_callbacks.insert(0,self.eraseBrush2)

                # bind the shortcut 'q' to acceptEdit function
                # 'ctrl+q' is by default bound to exit, so no ctrl here
                labelLayer.bind_key('q',func=self.acceptContAssist)
                labelLayer.bind_key('Control-Delete',func=self.deleteContAssist)
                

                # detect pressing "q" when they add the new contour -->
                # TODO
                if not self.inAssisting:
                    self.inAssisting=True

                    # wait for keypress

                    # after key press:
                    # moved to key listener fcn

                else:
                    # do nothing
                    pass



    def updateNewROIprops(self,event):
        curROIlayerName='ROI' # if not self.contAssist else 'contourAssist'
        roiLayer=self.findROIlayer(layerName=curROIlayerName,quiet=True)

        # check the number of shapes on the layer
        n=len(roiLayer.data)

        if self.roiCount==n:
            # already done
            return
        #debug:
        print(f'---- {n} rois on layer ----')
        print(f'---- {self.roiCount} rois in manager ----')

        if n==1:
            # empty shapes layer, init the props
            roiLayer.properties={'name':['0001'],'class':[0],'nameInt':[1]}
        elif n==0:
            # this should never happen
            print('update ROI props function called on empty layer')
            pass
        #else:
        elif self.roiCount==n-1:
            # the latest roi is the new one, rename it
            lastNumber=roiLayer.properties['nameInt'][-2] # second last in the list
            roiLayer.properties['nameInt'][-1]=lastNumber+1
            roiLayer.properties['name'][-1]='{:04d}'.format(lastNumber+1)
            # default class is 0 (no class)
            roiLayer.properties['class'][-1]=0
        elif self.roiCount>n:
            self.roiCount=n-1
        else:
            print(f'unexpected values: {n} rois on layer, {self.roiCount} rois in managaer')

        # update text properties for display option
        roiLayer.text.refresh_text(roiLayer.properties)
        #self.roiCount+=1
        self.roiCount=len(roiLayer.data)
        print(f'roiCount: {self.roiCount}')
        self.roiLayer=roiLayer


    def initROItextProps(self):
        roiLayer=self.findROIlayer()
        initProps={'name': array(['0001'], dtype='<U4'),'class': array([0]),'nameInt': array([1])}
        roiLayer.text.add(initProps,1)


    def quickExport(self):
        # save the ImageJ ROI files as when pressing the "Save" button
        self.saveData()

        # construct mask file name
        # set output folder and create it
        selectedClass='masks'
        mainExportFolder='labelled_masks'
        exportFolder=os.path.join(self.defDir,selectedClass,mainExportFolder)
        os.makedirs(exportFolder,exist_ok=True)
        print('Created output folder: {}'.format(exportFolder))
        maskFileName=str(os.path.join(exportFolder,'{}.tiff'.format(os.path.splitext(self.defFile)[0])))

        # see if the roi layer is empty
        roiLayer=self.findROIlayer()
        roiCount=len(roiLayer.data)
        print('annotated objects: {}'.format(roiCount))
        if roiCount==0:
            # empty shapes layer, nothing to save
            print('No ROIs found to export')
            return

        # get the image size
        imageLayer=self.findImageLayer()
        if imageLayer is None:
            print('Quick export failed')
            return
        s=imageLayer.data.shape

        # export a mask image from the rois
        labels = roiLayer.to_labels([s[0], s[1]])
        labelLayer = self.viewer.add_labels(labels, name='labelled_mask')
        labelLayer.visible = False

        # save the mask image
        savedSuccess=labelLayer.save(maskFileName,plugin='napari-annotatorj')
        
        if not savedSuccess:
            # failed
            print('Could not save exported image')
        else:
            # success
            pass

        # delete this label layer
        self.viewer.layers.remove(labelLayer)

        # bring the ROI layer forward
        self.viewer.layers.selection.add(roiLayer)


    def closeActiveWindows(self):
        doClose=False
        # ask for confirmation
        response=QMessageBox.question(self, 'Save before quit', 'Do you want to save current contours?\nThis will overwrite any previously\nsaved annotation for this image.',QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Yes)
        if response==QMessageBox.No:
            # just quit
            print('No button clicked (Save before quit)')
            doClose=True
            return doClose

        elif response==QMessageBox.Yes:
            print('Yes button clicked (Save before quit)')
            # save rois first
            if self.started:
                print('  started')
            else:
                print('  ! started')

            roiLayer=self.findROIlayer()
            if roiLayer is not None:
                print('  roiLayer is not None')
            else:
                print('  roiLayer = None')
            

            if self.started and roiLayer is not None:
                #self.imp=self.findImageLayer().data
                if len(roiLayer.data)!=0:
                    print('  >> starting save...')
                    # save using a separate fcn instead:
                    self.saveData()
                    
                    print('in close confirm after save finished')
            elif self.selectedAnnotationType.equals('semantic'):
                #self.imp=self.findImageLayer().data
                print('  >> starting semantic save...')
                self.saveData()
                print('in close confirm after save finished')
            
            doClose=True
            
        elif response == QMessageBox.Cancel:
            # do nothing
            print("Cancel button clicked (Save before quit)")
            doClose=False
            return doClose

        elif response == QMessageBox.Close:
            # do nothing
            print("Closed close confirm (Save before quit)")
            doClose=False
            return doClose

        return doClose
    

    def closeWindowsAndSave(self):
        if self.started:
            # check if there are rois added:
            # see if the roi layer is empty
            roiLayer=self.findROIlayer()
            if (roiLayer is not None and len(roiLayer.data)>0): # or (selectedAnnotationType=='semantic' and imp is not None and imp.getOverlay is not None)
                # offer to save current roi set
                contClosing=self.closeActiveWindows()

                if self.stepping:
                    if not self.finishedSaving: #!contClosing || !finishedSaving
                        # wait
                        print('Not done yet')
                        #return;

                if contClosing:
                    # close roimanager
                    if roiLayer is not None:
                        # delete the ROI layer
                        self.viewer.layers.remove(roiLayer)
                        self.manager=None

                    self.inAssisting=False
                    self.startedEditing=False
                    self.origEditedROI=None
                    #if editManager is not None:
                    #    editManager.reset()


            # check if image was passed as input argument
            if not self.imageFromArgs:
                # close image too if open
                imageLayer=self.findImageLayer()
                while imageLayer is not None:
                    self.viewer.layers.remove(imageLayer)
                    imageLayer=self.findImageLayer()

            # clear the log window
            # there is no log window in napari
            


    def prevImage(self):
        print('Function not implemented yet')
        pass

        self.closeingOnPurpuse=True
        if not self.started:
            warnings.warn('Use Open to select an image in a folder first')
            return

        # check if there is a list of images and if we can have a previous image
        if self.curFileList is not None and len(self.curFileList)>1:
            # more than 1 images in the list
            if self.curFileIdx>0:
                # current image is not the first, we can go back
                self.stepping=True

                # save current annotation first
                # this is done in openNew() fcn

                # open previous image with Open fcn:
                # set image name:
                self.curFileIdx-=1
                self.defFile=self.curFileList[self.curFileIdx]
                self.openNew()
                #self.imp=self.findImageLayer().data

                # check if auto mask load is enabled
                if self.enableMaskLoad and self.autoMaskLoad and self.maskFolderInited:
                    # load the mask from the selected folder automatically
                    self.loadROIs()

                return


        # this should not happen due to button inactivation, but handle it anyway:
        # if we get here there is no previous image to open, show message
        warnings.warn('There is no previous image in the current folder')
        return


    def nextImage(self):
        print('Function not implemented yet')
        pass

        self.closeingOnPurpuse=True
        if not self.started:
            warnings.warn('Use Open to select an image in a folder first')
            return

        # check if there is a list of images and if we can have a previous image
        if self.curFileList is not None and len(self.curFileList)>1:
            # more than 1 images in the list
            if self.curFileIdx<len(self.curFileList)-1:
                # current image is not the last, we can go forward
                self.stepping=True

                # save current annotation first
                # this is done in openNew() fcn

                # open next image with Open fcn:
                # set image name:
                self.curFileIdx+=1
                self.defFile=self.curFileList[self.curFileIdx]
                self.openNew()
                #self.imp=self.findImageLayer().data

                # check if auto mask load is enabled
                if self.enableMaskLoad and self.autoMaskLoad and self.maskFolderInited:
                    # load the mask from the selected folder automatically
                    self.loadROIs()

                return


        # this should not happen due to button inactivation, but handle it anyway:
        # if we get here there is no previous image to open, show message
        warnings.warn('There is no previous image in the current folder')
        return



    @thread_worker(start_thread=False)
    def startUnetLoading(self):
        importMode=None
        if self.modelFolder is None:
            print(f'Model folder not set, please set it to the folder containing the model {self.modelJsonFile} [.json and _weights.h5, or .hdf5]')
            return None
        if self.modelJsonFile is None:
            print(f'Model file [.json and _weights.h5, or .hdf5] not set, please set it')
            return None
        else:
            if os.path.isfile(os.path.join(self.modelFolder,self.modelJsonFile+'.json')) and os.path.isfile(os.path.join(self.modelFolder,self.modelJsonFile+'_weights.h5')):
                print('  >> importing from json config + weights .h5 files...')
                importMode=0
            elif os.path.isfile(os.path.join(self.modelFolder,self.modelJsonFile+'.hdf5')):
                print('  >> importing from a single .hdf5 file...')
                importMode=1
            else:
                print(f'Model file {self.modelJsonFile} [.json and _weights.h5, or .hdf5] does not exist in model folder {self.modelFolder}')
                return None
        #from .predict_unet import callPredictUnet,callPredictUnetLoaded,loadUnetModel
        #from .predict_unet import loadUnetModel
        # help the manual startup script import:
        try:
            from .predict_unet import loadUnetModelSetGpu
        except ImportError as e:
            try:
                from predict_unet import loadUnetModelSetGpu
            except Exception as e:
                print(e)
                return

        #model=loadUnetModel(os.path.join(self.modelFolder,self.modelJsonFile),importMode=importMode)
        model=loadUnetModelSetGpu(os.path.join(self.modelFolder,self.modelJsonFile),importMode=importMode,gpuSetting=self.gpuSetting)
        print('  >> importing done...')
        print('  >> no exception in loading the model...')
        return model


    def setUnetModel(self,model):
        if model is not None:
            self.trainedUNetModel=model
            print('Successfully loaded pretrained U-Net model for contour correction')
        else:
            print('>>>> Failed, model is None')


    def startUnet(self):
        #from .predict_unet import callPredictUnet,callPredictUnetLoaded#,loadUnetModel
        try:
            threadWorker=self.startUnetLoading()
            #threadWorker=loadUnetModelHere(os.path.join(self.modelFolder,self.modelJsonFile))
            #threadWorker=create_worker(loadUnetModelHere,os.path.join(self.modelFolder,self.modelJsonFile),_start_thread=False)
            threadWorker.started.connect(lambda: print('>>>> Started loading U-Net model...'))
            threadWorker.returned.connect(lambda x: self.setUnetModel(x))
            threadWorker.start()
        except Exception as e:
            print(f'Could not load model {self.modelFolder}/{self.modelJsonFile}')
            print(e)
            raise(e)


    # contour assist using U-Net
    def contourAssistUNet(self,imp,initROI,initROIdata,intensityThresh,distanceThresh,modelJsonFile,modelWeightsFile):
        assistedROI=None
        self.invertedROI=None
        self.ROIpositionX=0
        self.ROIpositionY=0
        print('  >> started assisting...')

        width=imp.shape[0]
        height=imp.shape[1]
        ch=imp.shape[2] if len(imp.shape)>2 else 0

        # see if the image is RGB or not
        origImageType=imp.dtype
        maxOrigVal=0
        colourful=False
        if (origImageType==numpy.dtype('uint8') or origImageType==numpy.dtype('int8')) and ch==0:
            maxOrigVal=255
            print('Image type: GRAY8')
        elif (origImageType==numpy.dtype('uint16') or origImageType==numpy.dtype('int16')) and ch==0:
            maxOrigVal=65535
            print('Image type: GRAY16')
        elif (origImageType==numpy.dtype('float32') or origImageType==numpy.dtype('float64')) and ch==0:
            maxOrigVal=int(1.0)
            warnings.warn('Current image is of type float in range [0,1].\nType not supported in suggestion mode.')
            print('Image type: GRAY32')
            return None
        elif (origImageType==numpy.dtype('uint8') or origImageType==numpy.dtype('int8')) and ch==3:
            # 8-bit indexed image
            maxOrigVal=255;
            print('Image type: COLOR_256')
        elif (origImageType==numpy.dtype('uint32') or origImageType==numpy.dtype('int32')) and ch==3:
            # 32-bit RGB colour image
            maxOrigVal=255
            colourful=True
            print('Image type: COLOR_RGB')
        else:
            maxOrigVal=255
            print('Image type: default')


        # get the bounding box of the current roi
        initBbox=[]
        x,y,w,h=cv2.boundingRect(initROIdata) # how to add this bbox: rect=shapes.add_rectangles([[y,x],[y+h,x+w]])
        initBbox.append(x)
        initBbox.append(y)
        initBbox.append(w)
        initBbox.append(h)
        #if initBbox[2]<2 or initBbox[3]<2:
            # 1-pixel width or height, fall back to the orig roi

        print(f'initBbox bounds: ({initBbox[0]},{initBbox[1]}) {initBbox[2]}x{initBbox[3]}')
        # allow x pixel growth for the new suggested contour
        tmpBbox=[max(x-int(self.distanceThreshVal/2),0),max(y-int(self.distanceThreshVal/2),0),min(w+int(self.distanceThreshVal/2),width),min(h+int(self.distanceThreshVal/2),height)]
        '''
        tmpROI=self.scaleContour(initROIdata,self.calcScale(w,self.distanceThreshVal)) # grow by distance thresh pixels
        tmpBbox=[]
        x,y,w,h=cv2.boundingRect(tmpROI)
        tmpBbox.append(x)
        tmpBbox.append(y)
        tmpBbox.append(w)
        tmpBbox.append(h)
        '''
        print(f'tmpROI bounds: ({tmpBbox[0]},{tmpBbox[1]}) {tmpBbox[2]}x{tmpBbox[3]}')


        #from .predict_unet import callPredictUnet,callPredictUnetLoaded,loadUnetModel
        try:
            from .predict_unet import callPredictUnet,callPredictUnetLoadedNoset,loadUnetModelSetGpu
        except ImportError as e:
            try:
                from predict_unet import callPredictUnet,callPredictUnetLoadedNoset,loadUnetModelSetGpu
            except Exception as e:
                print(e)
                return
        # load trained unet model
        if self.trainedUNetModel is not None:
            # model already loaded
            pass
        else:
            # load model
            
            # model loading was here, moved to its own fcn now
            #self.trainedUNetModel=loadUnetModel(os.path.join(self.modelFolder,self.modelJsonFile),importMode=0)
            self.trainedUNetModel=loadUnetModelSetGpu(os.path.join(self.modelFolder,self.modelJsonFile),importMode=0,gpuSetting=self.gpuSetting)

            # check if model was loaded
            if self.trainedUNetModel is None:
                print('Failed to load U-Net model for Contour assist')
                return None
        
        maskImage=None

        # check if this image has a valid prediction
        if not (self.curPredictionImage is None or self.curOrigImage is None):
            # check current image for equality too
            for xLayer in self.viewer.layers:
                if (xLayer.__class__ is Image):
                    if xLayer.name=='title':
                        # temp image, ignore it
                        continue
                    elif xLayer.name=='Image':
                        # current image window
                        curImageTmp=xLayer.data
                        if not self.imageFromArgs:
                            # normal image
                            if numpy.array_equal(curImageTmp,self.curOrigImage):
                                # it is the same, no changes applied, we can continue using the previous prediction on it
                                print('  >> using previous predicted image')

                                maskImage=self.viewer.add_image(self.curPredictionImage,name='title')
                                tmpLayer=self.findImageLayerName(layerName='title')
                                if tmpLayer is not None:
                                    tmpLayer.visible=True
                            else:
                                print('  >> current image does not match the previous predicted original image (image)')

                        else:
                            # stack
                            if numpy.array_equal(curImageTmp,self.curOrigImage):
                                # it is the same, no changes applied, we can continue using the previous prediction on it
                                print('  >> using previous predicted image')

                                maskImage=self.viewer.add_image(self.curPredictionImage,name='title')
                                tmpLayer=self.findImageLayerName(layerName='title')
                                if tmpLayer is not None:
                                    tmpLayer.visible=True
                            else:
                                print('  >> current image does not match the previous predicted original image (stack)')
                                self.curPredictionImage=None
                                #return contourAssistUNet(imp,initROI,intensityThresh,distanceThresh,modelJsonFile,modelWeightsFile)
                                return None

        else:
            # need to predict

            # show a dialog informing the user that prediction is being executed and wait
            # false to make in non-modal
            predictionStartedDialog=QDialog(self)
            predictionStartedDialog.setWindowTitle('Suggesting contour, please wait...')
            #predictionStartedDialog.setText('Creating suggested contour, please wait...')
            predictionStartedDialog.setModal(False)
            tmpContainer=QWidget(predictionStartedDialog)
            tmpContainer.setGeometry(QRect(0,0,250,80))
            dialogText=QLabel(tmpContainer)
            dialogText.setText('Creating suggested contour, please wait...')
            predictionStartedDialog.show()


            tmpLayer=self.findImageLayerName(layerName='Image')
            if tmpLayer is not None:
                self.curOrigImage=tmpLayer.data
            else:
                self.curOrigImage=None
            if self.curOrigImage is None:
                warnings.warn("Error","Cannot find image")
                return None

            print('  >> input image prepared...')

            # divide image by 255!!!!!!!!!!!!!!!
            #thisImage=self.curOrigImage/255

            # divide image by 255! and resize to 256x256 is handled in the predict_unet script


            # debug:
            #print('  >> input image size: '+thisImage.shape[0]+' x '+thisImage.shape[1])
            #print('  >> input array size: '+inputs.length)
            
            # expects rank 4 array with shape [miniBatchSize,layerInputDepth,inputHeight,inputWidth]
            #predictedImage=callPredictUnetLoaded(self.trainedUNetModel,self.curOrigImage,gpuSetting=self.gpuSetting)
            predictedImage=callPredictUnetLoadedNoset(self.trainedUNetModel,self.curOrigImage)
            print('  >> prediction done...')

            maskImage=self.viewer.add_image(predictedImage,name='title')

            print('  >> predicted image processed...')
            tmpLayer=self.findImageLayerName(layerName='title')
            if tmpLayer is not None:
                tmpLayer.visible=True

            if predictionStartedDialog is not None:
                predictionStartedDialog.done(1)
                



            # store prediction image until this image is closed/ new image is opened
            self.curPredictionImage=deepcopy(maskImage.data)
            self.curPredictionImageName=self.defFile

            # -----
            # debug:
            #ImagePlus predIm2show=new ImagePlus("prediction",predIm2showProc);
            #predIm2show.show();
            #return null;
            # ----


        # -------- here we have a valid prediction image and file name


        # crop initROI + distanceTresh pixels bbox of the predmask
        #cropMask=maskImage.data[x:x+w,y:y+h]
        #cropMask=maskImage.data[y:y+h,x:x+w]
        cropMask=maskImage.data[tmpBbox[1]:tmpBbox[1]+tmpBbox[3],tmpBbox[0]:tmpBbox[0]+tmpBbox[2]]
        t=skimage.filters.threshold_otsu(cropMask)
        masked=cropMask>t

        # see if the mask needs to be inverted:
        if self.checkIJMatrixCorners(maskImage.data):
            # need to invert it
            print('  >> need to invert mask: true')
            masked=cropMask<=t



        # -------- active contour method starts here ------------
        # moved to its own fcn

        # after everything is done: the new binary image must be converted to selection (Roi) and displayed on the image
        # create selection command, ThresholdToSelection class
        intermediateRoi,hierarchy=cv2.findContours(masked.astype(numpy.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if intermediateRoi is not None:
            pass
        else:
            print('intermediateRoi is None')
        
        # run active contour fitting:
        # not here!

        # store objects needed to run active contour fitting for later
        # TODO:
        #acObjects=new ACobjectDump(new ImagePlus(maskImage.getTitle(),maskImage.getProcessor().duplicate()),intermediateRoi,tmpBbox,new ImagePlus(imp.getTitle(),imp.getProcessor().duplicate()));

        # check if there is an output from ac as a roi
        if (assistedROI is None):# or not(assistedROI.getType()==Roi.FREEROI or assistedROI.getType()==Roi.COMPOSITE or assistedROI.getType()==Roi.TRACED_ROI)):
            # failed to produce a better suggested contour with AC than we had with unet before, revert to it
            print('Failed to create new contour with active contours, showing U-Net prediction')
            imp=None # placeholder
            assistedROI=deepcopy(intermediateRoi)
            assistedROI=self.postProcessAssistedROI(assistedROI,tmpBbox,maskImage.data,True,imp,True)
            # also reset the inverted roi
            if numpy.array_equal(assistedROI,self.invertedROI):
                print('Failed to invert current roi (same)')
            if self.invertedROI is None:
                print('  null ROI on line #3822')

        
        # roi positioning was done here, moved to its own fcn

        tmpLayer=self.findImageLayerName(layerName='title')
        if tmpLayer is not None:
            self.viewer.layers.remove(tmpLayer)
        # set main imwindow var to the original image

        return assistedROI


        #return initROI
        #return tmpROI



    def scaleContour(self,cnt,scale):
        M=cv2.moments(cnt)
        cx=int(M['m10']/M['m00']) if M['m00']>0.0 else 0
        cy=int(M['m01']/M['m00']) if M['m00']>0.0 else 0
        print(f'cx: {cx}')
        print(f'cy: {cy}')

        cnt_norm=cnt-[cx, cy]
        cnt_scaled=cnt_norm*scale
        cnt_scaled=cnt_scaled+[cx, cy]
        cnt_scaled=cnt_scaled.astype(numpy.int32)

        return cnt_scaled


    def calcScale(self,width,dist):
        return (width+dist)/width


    # to check if the mask needs to be inverted count the corners marked
    # it at least 2 corners are marked, the mask should be inverted
    def checkIJMatrixCorners(self,matrix):
        need2invert=False
        w=matrix.shape[0]
        h=matrix.shape[1]

        cornerCount=0
        if matrix[0][0]>0:
            cornerCount+=1
        if matrix[0][h-1]>0:
            cornerCount+=1
        if matrix[w-1][0]>0:
            cornerCount+=1
        if matrix[w-1][h-1]>0:
            cornerCount+=1

        if cornerCount>1:
            # need to invert the image as the background is white
            need2invert=True
        return need2invert


    def validateROI(self,assistedROI,maskImage):

        if assistedROI is not None and len(assistedROI)>1:
            # select the largest found object and delete all others
            assistedROI=self.selectLargestROI(assistedROI)

        # check if we have a valid roi now, else return null
        if assistedROI is not None:

            curBbox=[]
            if type(assistedROI) is tuple:
                assistedROI=assistedROI[0]
            x,y,w,h=cv2.boundingRect(assistedROI) # how to add this bbox: rect=shapes.add_rectangles([[y,x],[y+h,x+w]])
            curBbox.append(x)
            curBbox.append(y)
            curBbox.append(w)
            curBbox.append(h)
            #debug:
            print('curBbox: ')
            print(curBbox)

            # check if the corner points are included
            cornerCount=0
            if [0.0,0.0] in assistedROI:
                # top left corner
                cornerCount+=1
                print('     (0,0) corner')
            
            if [0.0,curBbox[2]] in assistedROI:
                # ? top right corner
                cornerCount+=1
                print('     (0,+) corner')
            
            if [curBbox[3],0.0] in assistedROI:
                # ? lower left corner
                cornerCount+=1
                print('     (+,0) corner')
            
            if [curBbox[3],curBbox[2]] in assistedROI:
                # ? lower right corner
                cornerCount+=1
                print('     (+,+) corner')

            if cornerCount>1:
                # at least 2 corners of the crop are included in the final roi, invert it!
                # store an inverted roi for later option to change

                #assistedROI=assistedROI.getInverse(maskImage)
                #invertedROI=assistedROI.getInverse(maskImage)
                print('  >> inverted ROI') # <-- TODO
                if numpy.array_equal(assistedROI,self.invertedROI):
                    print('Failed to invert current roi (same)')

                if self.invertedROI is None:
                    print('  null ROI on line #4137')
            


            # select the largest found object and delete all others
            assistedROI=self.selectLargestROI(assistedROI)

        
        return assistedROI
    


    # get the largest roi if multiple objects were detected on the mask
    def selectLargestROI(self,ROI2check):
        if type(ROI2check) is tuple:
            pass
        elif type(ROI2check) is numpy.ndarray:
            print('no need to select, already an ndarray')
            return ROI2check
        maxSize=0
        for idx,x in enumerate(ROI2check):
            if x.shape[0]>maxSize:
                maxSize=x.shape[0]
                maxIdx=idx

        # here we have the largest object index as maxIdx
        ROI2checkRet=ROI2check[maxIdx]
        return ROI2checkRet


    def postProcessAssistedROI(self,assistedROI,tmpBbox,maskImage,closeMaskIm,imp,storeRoiCoords):

        # validate current ROI and check if it needs to be inverted
        assistedROI=self.validateROI(assistedROI,maskImage)

        if assistedROI is None:
            print('  >> failed to create new contour')
            if closeMaskIm:
                # close image window
                tmpLayer=self.findImageLayerName(layerName='title')
                if tmpLayer is not None:
                    self.viewer.layers.remove(tmpLayer)
            self.invertedROI=None
            
            print('  null ROI on line #3909')
            
        else:
            assistedBbox=[]
            x,y,w,h=cv2.boundingRect(assistedROI)
            assistedBbox.append(x)
            assistedBbox.append(y)
            assistedBbox.append(w)
            assistedBbox.append(h)
            print(f'assistedROI bounds: ({assistedBbox[0]},{assistedBbox[1]}) {assistedBbox[2]}x{assistedBbox[3]}')

            # store an inverted roi for later option to change
            # TODO: ˇˇ
            #invertedROI=assistedROI.getInverse(maskImage);
            print('Stored inverse ROI')
            if self.invertedROI is None:
                print('  null ROI on line #4309')
            else:
                if numpy.array_equal(assistedROI,self.invertedROI):
                    print('Failed to invert current roi (same)')
                

            # store the coordinates of the roi's bounding box
            if storeRoiCoords:
                
                self.ROIpositionX=tmpBbox[0]
                self.ROIpositionY=tmpBbox[1]
                print('ROIposition (X,Y): '+str(self.ROIpositionX)+','+str(self.ROIpositionY))


            if self.invertedROI is None:
                print('  null ROI on line #4328')
            

            if closeMaskIm:
                # close image window
                tmpLayer=self.findImageLayerName(layerName='title')
                if tmpLayer is not None:
                    self.viewer.layers.remove(tmpLayer)


            # place new ROI on the new mask
            # added in contAssistROI fcn
            '''
            roiLayer=self.findROIlayer()
            shape=numpy.array(numpy.fliplr(numpy.squeeze(assistedROI)))
            roiLayer.add_polygons(shape)
            roiLayer.refresh()
            '''
            # flip x,y here because they will be flipped later <-- nope
            assistedROI=assistedROI+[self.ROIpositionX,self.ROIpositionY]

        #return assistedROI
        return numpy.array(numpy.fliplr(numpy.squeeze(assistedROI)))


    def addContAssistLayer(self):
        shapesLayer=self.findROIlayer()
        curColour=shapesLayer._data_view._edge_color[-1] if len(shapesLayer.data)>0 else 'white'
        # create temp layer for contour assist
        contAssistLayer=self.findROIlayer(layerName='contourAssist')
        if contAssistLayer is not None:
            #self.viewer.layers.remove(contAssistLayer)
            return contAssistLayer
        else:
            shapesLayer2=Shapes(name='contourAssist',shape_type='polygon',edge_width=2,edge_color=curColour,face_color=[0,0,0,0])
            self.viewer.add_layer(shapesLayer2)
            return shapesLayer2


    def classifyROI(self,layer,event):
        if not self.classMode:
            return
        if layer.mode=='add_polygon':
            msg='Cannot start classifying when {} is selected. Please select {}'.format(
                '\'Add polygons(P)\'','\'Select shapes(5)\'')
            warnings.warn(msg)
            return
        elif layer.mode!='select':
            msg='Cannot start classifying. Please select {}'.format('\'Select shapes(5)\'')
            warnings.warn(msg)
            return

        yield

        # assign a class (group) to the selected ROI
        if self.editMode or self.addAuto or self.inAssisting:
            # cannot classify in edit mode
            msg='Cannot classify objects if selected:\n edit mode\n contour assist\n add automatically'
            warnings.warn(msg)
            return

        # find current image
        imageLayer=self.findImageLayer()
        if imageLayer is None:
            print('No image layer found')
            return
        elif imageLayer.data is None:
            print('No image opened')
            return
        else:
            # we have an image
            s=imageLayer.data.shape
            self.imgSize=s

            # get clicked coordinates relative to the source component
            pos=layer.world_to_data(event.position)

            if pos[0]<=0 or pos[1]<=0 or pos[0]>s[1] or pos[1]>s[0]:
                print('(Class mode) not on the image')
            else:
                print('(Class mode) click on {}'.format(pos))

                print(f'found {self.roiCount} rois')

                # check if the user clicked on a shape
                if len(layer.selected_data)>0:
                    # clicked on a shape
                    # get the index of the shape and remove the selection in one go
                    curIdx=layer.selected_data.pop()
                    print('Selected \'{}\' ROI for classification'.format(layer.properties['name'][curIdx]))
                    # remove selection bbox around roi
                    # simulate mouse move somewhere else
                    # TODO!
                    '''
                    layer.mode='transform'
                    layer.mode='select'
                    '''

                    # fetch currently selected class info
                    # currently selected class we used as group:
                    curGroup=layer.properties['class'][curIdx]
                    if curGroup==self.selectedClassNameNumber:
                        # already in the target group
                        # --> unclassify it!
                        layer.properties['class'][curIdx]=0
                        layer._data_view.update_face_color(curIdx,[0,0,0,0])
                        layer._data_view.update_edge_color(curIdx,self.colourString2Float(self.defColour)) # this is the default contour colour
                        
                        print(f'Selected {layer.properties["name"][curIdx]} ROI to unclassify (0)')

                    else:
                        self.startedClassifying=True
                        # store the current class name idx as group for saving check
                        if not self.selectedClassNameNumber in self.usedClassNameNumbers:
                            self.usedClassNameNumbers.append(self.selectedClassNameNumber)

                        layer.properties['class'][curIdx]=self.selectedClassNameNumber
                        # its colour:
                        #debug:
                        print(f'currently selected class colour: {self.selectedClassColourIdx}')
                        layer._data_view.update_edge_color(curIdx,self.colourString2Float(self.selectedClassColourIdx))
                        
                        print(f'Selected {layer.properties["name"][curIdx]} ROI to class {self.selectedClassNameNumber}')

                        #debug:
                        checkColour=layer._data_view._edge_color[curIdx]
                        print(f'set stroke color to: {self.getClassColourIdx(checkColour)}')

                    # refresh text props
                    layer.refresh()
                    layer.refresh_text()
                    #layer.mode='select'
                    
                else:
                    # failed to find the currently clicked point's corresponding ROI
                    print('Could not find the ROI associated with the selected point on the image.')


    def colourString2Float(self,colourString):
        self.colourFloats=[[1,0,0,1],[0,1,0,1],[0,0,1,1],[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,0.65,0,1],[1,1,1,1],[0,0,0,1]]
        curColourIdx=self.colours.index(colourString)
        return self.colourFloats[curColourIdx]


    # fetch class color idx from the classidxlist
    def getClassColourIdx(self,curColour):
        curColourIdx=-1

        if curColour in numpy.array(self.colourFloats):
            def checkElement(i,j):
                return i==j
            checkElementVectorized=numpy.vectorize(checkElement)
            elementCheck=checkElementVectorized(curColour,self.colourFloats)
            curColourIdx=[idx for idx,val in enumerate(elementCheck) if val.all()]
            if curColourIdx is not None and len(curColourIdx)>0:
                curColourIdx=curColourIdx[0]
        else:
            print('Unexpected Color, no matching class colour index')

        return curColourIdx


    def setEditMode(self,state):
        shapesLayer=self.findROIlayer()
        if state == Qt.Checked:
            self.editMode=True
            print('Edit mode selected')

            # set the "select shapes" mode
            shapesLayer.mode = 'select'

            self.contAssist=False
            self.chckbxContourAssist.setChecked(False)
            self.chckbxContourAssist.setEnabled(False)
            self.chckbxContourAssist.setStyleSheet("color: gray")
            self.chckbxClass.setChecked(False)
            self.chckbxClass.setEnabled(False)
            self.chckbxClass.setStyleSheet("color: gray")
            self.classMode=False

        else:
            # close remaining editing temp layer if present
            editLayer=self.findLabelsLayerName()
            if editLayer is not None:
                self.rejectEdit2(editLayer)
                #self.viewer.layers.remove(editLayer)
            else:
                print('editLayer is None')

            #self.startedEditing=False
            #self.origEditedROI=None

            # make the ROI layer the active one
            #roiLayer=self.findROIlayer()
            #self.viewer.layers.selection.add(roiLayer)

            self.editMode=False
            print('Edit mode cleared')
            # set the "add polygon" mode
            shapesLayer.mode = 'add_polygon'

            self.chckbxContourAssist.setEnabled(True)
            self.chckbxContourAssist.setStyleSheet("color: white")
            self.chckbxClass.setEnabled(True)
            self.chckbxClass.setStyleSheet("color: white")


    def showCnt(self,state):
        shapesLayer=self.findROIlayer()
        if state == Qt.Checked:
            print('Show contours selected')
            shapesLayer.visible=True

        else:
            print('Show contours cleared')
            shapesLayer.visible=False



    def setContourAssist(self,state):
        shapesLayer=self.findROIlayer()
        if state == Qt.Checked:
            self.contAssist=True
            print('Contour assist selected')
            
            # disable auto adding was here, but it is already on by default

            # should set boolean vars to:
                    # first start freehand selection tool for drawing -->
                        # on mouse release start contour correction -->
                            # user can check it visually -->
                                # set brush selection tool for contour modification -->
                                    # detect pressing "q" when they add the new contour -->
                                        # reset freehand selection tool

            # add drawing layer for contour assist
            shapesLayer2=self.addContAssistLayer()
            self.addFreeROIdrawingCA(shapesLayer2)

            shapesLayer2.mode='add_polygon'

            self.chkEdit.setChecked(False)
            self.chkEdit.setEnabled(False)
            self.chkEdit.setStyleSheet("color: gray")
            self.editMode=False
            #self.chckbxStepThroughContours.setChecked(False)
            #self.chckbxStepThroughContours.setEnabled(False)

            self.chckbxClass.setChecked(False)
            self.chckbxClass.setEnabled(False)
            self.chckbxClass.setStyleSheet("color: gray")
            self.classMode=False

        else:
            self.contAssist=False
            print('Contour assist cleared')

            # can enable auto add again
            #self.chckbxAddAutomatically.setEnabled(True)
            #self.chckbxStepThroughContours.setEnabled(True)
            self.chckbxClass.setEnabled(True)
            self.chckbxClass.setStyleSheet("color: white")
            self.chkEdit.setEnabled(True)
            self.chkEdit.setStyleSheet("color: white")

            # close remaining editing temp layer if present
            editLayer=self.findLabelsLayerName()
            if editLayer is not None:
                self.viewer.layers.remove(editLayer)
            else:
                print('editLayer is None')

            # close remaining contour assist temp layer if present
            contAssistLayer=self.findROIlayer(layerName='contourAssist')
            if contAssistLayer is not None:
                self.viewer.layers.remove(contAssistLayer)

            # reset vars
            self.inAssisting=False
            self.invertedROI=None
            self.ROIpositionX=0
            self.ROIpositionY=0
            self.acObjects=None
            self.startedEditing=False
            self.origEditedROI=None

            # make the ROI layer the active one
            self.viewer.layers.selection.add(shapesLayer)


    def setClassMode(self,state):
        shapesLayer=self.findROIlayer()
        if state == Qt.Checked:
            self.classMode=True
            print('Class mode selected')

            # disable automatic adding to list and contour assist while editing
            print('Switching automatic adding and contour assist off')
            self.addAuto=False
            #self.chckbxAddAutomatically.setChecked(False)
            #self.chckbxAddAutomatically.setEnabled(False)

            self.contAssist=False
            self.chckbxContourAssist.setChecked(False)
            self.chckbxContourAssist.setEnabled(False)

            self.editMode=False
            #self.chckbxStepThroughContours.setChecked(False)
            #self.chckbxStepThroughContours.setEnabled(False)

            # start the classes frame
            #debug:
            print(f'class list: {self.listModelClasses}')
            print(f'selected col: {self.selectedClassColourIdx}')
            self.classesFrame=ClassesFrame(self.viewer,self)

            # add mouse click callback to classify ROIs
            if self.classifyROI not in shapesLayer.mouse_drag_callbacks:
                shapesLayer.mouse_drag_callbacks.append(self.classifyROI)

            shapesLayer.mode='select'

        else:
            self.classMode=False
            print('Class mode cleared')

            self.chckbxContourAssist.setEnabled(True)
            if self.classesFrame is not None:
                try:
                    self.viewer.window.remove_dock_widget(self.classesFrame)
                except Exception as e:
                    print(e)

            shapesLayer.mode='add_polygon'
            


# -------------------------------------
# end of class AnnotatorJ
# -------------------------------------

class BackupROI():
    def __init__(self,shape,idx=0,edgeColour=numpy.array([1.,1.,1.,1.]),faceColour=numpy.array([0.,0.,0.,0.]),edgeWidth=0.5):
        self.shape=shape
        self.idx=idx
        self.edgeColour=edgeColour
        self.faceColour=faceColour
        self.edgeWidth=edgeWidth

    def setShape(self,shape):
        self.shape=shape

    def setIdx(self,idx):
        self.idx=idx

    def setEdgeColour(self,colour):
        self.edgeColour=colour

    def setFaceColour(self,colour):
        self.faceColour=colour

    def setEdgeWidth(self,width):
        self.edgeWidth=width

    def toString(self):
        print('shape: {}\nidx: {}\nedgeColour: {}\nfaceColour: {}\nedgeWidth: {}'.format(self.shape,self.idx,self.edgeColour,self.faceColour,self.edgeWidth))

    def toStringNoShape(self):
        print('idx: {}\nedgeColour: {}\nfaceColour: {}\nedgeWidth: {}'.format(self.idx,self.edgeColour,self.faceColour,self.edgeWidth))

# -------------------------------------
# end of class BackupROI
# -------------------------------------


# new frame for classes when selecting the "Class mode" checkbox in the main frame
class ClassesFrame(QWidget):
    def __init__(self,napari_viewer,annotatorjObj):
        super().__init__()
        self.viewer = napari_viewer
        self.annotatorjObj=annotatorjObj

        # check if there is opened instance of this frame
        # the main plugin is: 'napari-annotatorj: Annotator J'
        if self.annotatorjObj.classesFrame is not None:
            # already inited once, load again
            print('detected that Classes widget has already been initialized')
            if self.annotatorjObj.classesFrame.isVisible():
                print('Classes widget is visible')
                return
            else:
                print('Classes widget is not visible')
                # rebuild the widget

        #self.classesFrame=QWidget()
        #self.classesFrame.setWindowTitle('Classes')
        #self.classesFrame.resize(350, 200)
        '''
        self.classesFrame.addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {
                self.classesFrame.dispose();
                self.classesFrame=null;
            }
        });
        '''

        '''
        self.bsize=self.annotatorjObj.bsize2
        self.classNameLUT=self.annotatorjObj.classNameLUT
        self.listModelClasses=self.annotatorjObj.listModelClasses
        self.classFrameNames=self.annotatorjObj.classFrameNames
        self.classListSelectionHappened=False
        self.defaultClassNumber=self.annotatorjObj.defaultClassNumber
        self.defColour=self.annotatorjObj.defColour
        '''
        
        self.lblClasses=QLabel('Classes')
        self.lblCurrentClass=QLabel('Current:')

        self.btnAddClass=QPushButton('+')
        self.btnAddClass.setToolTip('Add a new class')
        self.btnAddClass.clicked.connect(self.addNewClass)
        self.btnAddClass.setStyleSheet(f"max-width: {int(self.annotatorjObj.bsize2/2)}px")

        self.btnDeleteClass=QPushButton('-')
        self.btnDeleteClass.setToolTip('Delete current class')
        self.btnDeleteClass.clicked.connect(self.deleteClass)
        self.btnDeleteClass.setStyleSheet(f"max-width: {int(self.annotatorjObj.bsize2/2)}px")

        self.classListList=QListWidget()

        # list of colours
        self.rdbtnGroup=QComboBox()
        self.rdbtnGroup.addItem('Red')
        self.rdbtnGroup.addItem('Green')
        self.rdbtnGroup.addItem('Blue')
        self.rdbtnGroup.addItem('Cyan')
        self.rdbtnGroup.addItem('Magenta')
        self.rdbtnGroup.addItem('Yellow')
        self.rdbtnGroup.addItem('Orange')
        self.rdbtnGroup.addItem('White')
        self.rdbtnGroup.addItem('Black')
        

        # add default class option as a combo box
        self.lblClassDefault=QLabel('Default:')
        
        self.comboBoxDefaultClass=QComboBox()
        self.comboBoxDefaultClass.setToolTip('Assign this class to all objects by default.')


        curColourName='red'
        self.annotatorjObj.selectedClassColourIdx='red'
        if self.annotatorjObj.listModelClasses is None or len(self.annotatorjObj.listModelClasses)==0:
            self.classListList.insertItem(0,'Class_01')
            self.classListList.insertItem(1,'Class_02')
            self.annotatorjObj.listModelClasses=[]
            self.annotatorjObj.listModelClasses.append('Class_01')
            self.annotatorjObj.listModelClasses.append('Class_02')
            self.annotatorjObj.classFrameColours=[0,1]
            self.annotatorjObj.classFrameNames=['Class_01','Class_02']

            self.annotatorjObj.selectedClassNameNumber=1
            #self.classListList.setCurrentItem(self.classListList.item(0))

            # set default roi group to 0
            #pluginInstance=AnnotatorJ(self.viewer)
            roiLayer=self.annotatorjObj.findROIlayer()
            for idx,roi in enumerate(roiLayer.data):
                roiLayer.properties['class'][idx]=0

            self.comboBoxDefaultClass.addItem('(none)')
            self.comboBoxDefaultClass.addItem('Class_01')
            self.comboBoxDefaultClass.addItem('Class_02')

            self.comboBoxDefaultClass.setCurrentIndex(0)

            self.annotatorjObj.defaultClassNumber=0 #-1

            self.annotatorjObj.classNumberCounter=2

            self.annotatorjObj.classNameLUT['Class_01']=1
            self.annotatorjObj.classNameLUT['Class_02']=2

        else:
            # set default class selection list
            self.comboBoxDefaultClass.addItem('(none)')

            for i,el in enumerate(self.annotatorjObj.classFrameNames):
                curClassName=el
                if curClassName is None:
                    continue
                self.annotatorjObj.listModelClasses.append(curClassName)
                self.comboBoxDefaultClass.addItem(curClassName)
                self.classListList.insertItem(i,curClassName)

                curClassNum=int(curClassName[curClassName.find("_")+1:])
                print(f'classFrameNames.[i]: {self.annotatorjObj.classFrameNames[i]}, curClassNum: {curClassNum}')
                print(self.annotatorjObj.classNameLUT)

                self.annotatorjObj.classNameLUT[self.annotatorjObj.classFrameNames[i]]=curClassNum

            # set selected colour for the previously selected class and colour
            selectedClassNameVar='Class_{:02d}'.format(self.annotatorjObj.selectedClassNameNumber)
            selectedClassIdxList=self.annotatorjObj.classFrameNames.index(selectedClassNameVar)
            if selectedClassIdxList==-1:
                # could not find the selected class in the list e.g. if reimport didnt work well
                print('Could not find the selected class in the list of classes, using the first.')
                selectedClassIdxList=0
                self.classListList.setCurrentItem(self.classListList.item(0))
                tmpString=classListList.currentItem().text()
                if tmpString is None or tmpString=='None':
                    # failed to find the selected item in the list
                    self.annotatorjObj.selectedClassNameNumber=-1
                else:
                    self.annotatorjObj.selectedClassNameNumber=self.annotatorjObj.classNameLUT[tmpString]

            curColourName=self.setColourRadioButton(selectedClassNameVar,selectedClassIdxList)
            self.classListList.setCurrentItem(self.classListList.item(selectedClassIdxList))

            self.comboBoxDefaultClass.setCurrentIndex(0)
            self.annotatorjObj.defaultClassNumber=0 #-1


        self.classListList.setCurrentItem(self.classListList.item(0))

        # set default class (group) for all unclassified objects
        if self.annotatorjObj.classFrameNames is None and self.annotatorjObj.classFrameColours is None:
            self.setDefaultClass4objects()
        else:
            # it should already be set
            pass

        self.classListList.setSelectionMode(QAbstractItemView.SingleSelection)

        self.classListList.currentItemChanged.connect(self.classListSelectionChanged)
        self.lblCurrentClass.setText(f'<html>Current: <font color="{curColourName}">{self.classListList.currentItem().text()}</font></html>')

        self.rdbtnGroup.currentIndexChanged.connect(self.classColourBtnChanged)
        self.comboBoxDefaultClass.currentIndexChanged.connect(self.defaultClassSelectionChanged)

        # set default class (group) for all unclassified objects
        if self.annotatorjObj.classFrameNames is None and self.annotatorjObj.classFrameColours is None:
            self.setDefaultClass4objects()
        else:
            # it should already be set
            pass


        self.classMainVBox=QVBoxLayout()
        self.classHeaderHBox=QHBoxLayout()
        self.classContentHBox=QHBoxLayout()
        self.classRightVBox=QVBoxLayout()
        self.classRightHBox=QHBoxLayout()
        self.classHeaderInnerLeftHBox=QHBoxLayout()
        self.classHeaderInnerRightHBox=QHBoxLayout()

        self.classHeaderInnerLeftHBox.addWidget(self.lblClasses)
        # +/- buttons
        self.classHeaderInnerLeftHBox.addWidget(self.btnAddClass)
        self.classHeaderInnerLeftHBox.addWidget(self.btnDeleteClass)
        self.classHeaderInnerRightHBox.addWidget(self.lblCurrentClass)
        self.classContentHBox.addWidget(self.classListList)
        self.classRightVBox.addWidget(self.rdbtnGroup)
        self.classRightHBox.addWidget(self.lblClassDefault)
        self.classRightHBox.addWidget(self.comboBoxDefaultClass)

        self.classHeaderHBox.addLayout(self.classHeaderInnerLeftHBox)
        self.classHeaderInnerRightHBox.setAlignment(Qt.AlignRight)
        self.classHeaderHBox.addLayout(self.classHeaderInnerRightHBox)
        self.classContentHBox.setAlignment(Qt.AlignTop)
        self.classContentHBox.addLayout(self.classRightVBox)
        self.classRightVBox.addLayout(self.classRightHBox)
        self.classMainVBox.addLayout(self.classHeaderHBox)
        #self.classMainVBox.addLayout(self.classRightVBox)
        self.classMainVBox.addLayout(self.classContentHBox)

        #self.classesFrame.setLayout(self.classMainVBox)
        #self.classesFrame.show()
        self.setLayout(self.classMainVBox)
        #self.show()
        self.viewer.window.add_dock_widget(self,name='Classes')


    def addNewClass(self):
        # add new class to list
        lastClassName=self.annotatorjObj.classFrameNames[-1]
        lastClassNum=-1
        if lastClassName is None:
            lastClassNum=len(self.annotatorjObj.classNameLUT)
        else:
            lastClassNum=int(lastClassName[lastClassName.index("_")+1:len(lastClassName)])
        
        lastClassNum=-1
        for key in self.annotatorjObj.classNameLUT:
            tmpClassNum=self.annotatorjObj.classNameLUT[key]
            lastClassNum=tmpClassNum if tmpClassNum>lastClassNum else lastClassNum

        newClassName='Class_{:02d}'.format(lastClassNum+1)
        self.annotatorjObj.classFrameNames.append(newClassName)
        self.annotatorjObj.listModelClasses.append(newClassName)
        self.classListList.addItem(newClassName) # insert item to the end of the list
        self.comboBoxDefaultClass.addItem(newClassName)

        #classNameLUT.put(newClassName,classNameLUT.size()+1); # <-- if loaded and classes are not numbered from 1, this will mess the numbering up
        self.annotatorjObj.classNameLUT[newClassName]=lastClassNum+1

        # assign a free colour to the new class
        if len(self.annotatorjObj.classFrameNames)<=8:
            for i in range(8):
                if not (i in self.annotatorjObj.classFrameColours):
                    # found first free colour, take it
                    self.annotatorjObj.classFrameColours.append(i)
                    break
        else:
            # assign the first colour
            self.annotatorjObj.classFrameColours.append(0)


    def deleteClass(self):
        # see if there are classes in the list

        if self.classListList.count()>0:
            # remove selected class from list
            # fetch selected class from the list
            selectedClassName=self.classListList.currentItem().text()
            # find it in the classnames list
            try:
                selectedClassIdxList=self.annotatorjObj.classFrameNames.index(selectedClassName)
            except ValueError as e:
                # didnt find it
                print(e)
                print('Could not find the currently selected class name in the list for deletion. Please try again.')
                return

            self.annotatorjObj.classFrameNames.pop(selectedClassIdxList)
            self.annotatorjObj.classFrameColours.pop(selectedClassIdxList)
            self.annotatorjObj.listModelClasses.remove(selectedClassName)


            # reset group attribute of all ROIs in this class if any
            # set all currently assigned ROIs of this group to have the new contour colour
            # loop through all slices
            # TODO: when z-stack images are opened and annotated, the shapes layer is either multi-D or there are multiple shapes layers by slices; if so: loop the slices, run self.unClassifyClass(self.annotatorjObj.classNameLUT[selectedClassName]) and update the roi manager/layer and its visibility if needed; else: just this command is needed to run once
            self.unClassifyClass(self.annotatorjObj.classNameLUT[selectedClassName])
            
            # see if the default class is being deleted
            if self.annotatorjObj.defaultClassNumber==self.annotatorjObj.classNameLUT[selectedClassName]:
                # reset the default class to (none)
                print('>>>> deleting the default class --> unclassify these ROIs')
                self.comboBoxDefaultClass.setCurrentText('(none)')
                #unClassifyClass(defaultClassNumber);
                self.annotatorjObj.defaultClassNumber=0 #-1
                self.defaultClassColour=None

            #comboBoxDefaultClass.removeItemAt(selectedClassIdxList+1); # default class selector has "(none)" as the first element
            self.comboBoxDefaultClass.removeItem(self.comboBoxDefaultClass.findText(selectedClassName))
            self.classListList.takeItem(self.classListList.row(self.classListList.currentItem()))

            print(f'Deleted class "{selectedClassName}" from the class list')

            # select the first class as default
            if self.classListList.count()>0:
                self.classListList.setCurrentRow(0)
                selectedClassNameVar=self.annotatorjObj.listModelClasses[0]
                print(f'Selected class "{selectedClassNameVar}"')

                # store currently selected class's number for ROI grouping
                self.annotatorjObj.selectedClassNameNumber=int(selectedClassNameVar[selectedClassNameVar.find("_")+1:])
                self.selectedClassIdxList=self.annotatorjObj.classFrameNames.index(selectedClassNameVar)

                self.setColourRadioButton(selectedClassNameVar,self.selectedClassIdxList)
            else:
                # deleted the last class
                # allow this?
                pass


        else:
            # no classes in the list
            print('No classes left to delete.')


    # class list change listener
    def classListSelectionChanged(self,item):
        '''
        print(f'selected class: {item.text()}')
        curColourName=self.getCurColourName()
        self.lblCurrentClass.setText(f'<html>Current: <font color="{curColourName}">{item.text()}</font></html>')
        '''
        
        selectedClassNameIdx=self.classListList.currentRow()
        selectedClassNameVar=''

        self.classListSelectionHappened=True

        if selectedClassNameIdx<0:
            # selection is empty
            selectedClassNameVar=None
        else:
            selectedClassNameVar=self.annotatorjObj.listModelClasses[selectedClassNameIdx]
            print(f'Selected class "{selectedClassNameVar}"')

            # store currently selected class's number for ROI grouping
            self.annotatorjObj.selectedClassNameNumber=self.annotatorjObj.classNameLUT[selectedClassNameVar]

            # find its colour
            
            # find it in the classnames list
            selectedClassIdxList=self.annotatorjObj.classFrameNames.index(selectedClassNameVar)
            #debug:
            print(f'({selectedClassIdxList+1}/{len(self.annotatorjObj.classFrameNames)}) classes')
            if selectedClassIdxList<0:
                # didnt find it
                print('Could not find the newly selected class name in the list. Please try again.')
                return
            
            # moved radio button setting to its own fcn
            self.setColourRadioButton(selectedClassNameVar,selectedClassIdxList)

        self.classListSelectionHappened=False


    def getCurColourName(self):
        return self.rdbtnGroup.currentText()


    def classColourBtnChanged(self):
        # TODO
        rbText=self.rdbtnGroup.currentText()
        print(f'selected class colour: "{rbText}" for class {self.classListList.currentItem().text()}')

        # set vars according to radio buttons
        selectedClassColourCode=-1
        curColourName=None
        if rbText=='Red':
            selectedClassColourCode=0
            curColourName='red'
            self.annotatorjObj.selectedClassColourIdx='red'
        elif rbText=='Green':
            selectedClassColourCode=1
            curColourName='green'
            self.annotatorjObj.selectedClassColourIdx='green'
        elif rbText=='Blue':
            selectedClassColourCode=2
            curColourName='blue'
            self.annotatorjObj.selectedClassColourIdx='blue'
        elif rbText=='Cyan':
            selectedClassColourCode=3
            curColourName='cyan'
            self.annotatorjObj.selectedClassColourIdx='cyan'
        elif rbText=='Magenta':
            selectedClassColourCode=4
            curColourName='magenta'
            self.annotatorjObj.selectedClassColourIdx='magenta'
        elif rbText=='Yellow':
            selectedClassColourCode=5
            curColourName='yellow'
            self.annotatorjObj.selectedClassColourIdx='yellow'
        elif rbText=='Orange':
            selectedClassColourCode=6
            curColourName='orange'
            self.annotatorjObj.selectedClassColourIdx='orange'
        elif rbText=='White':
            selectedClassColourCode=7
            curColourName='white'
            self.annotatorjObj.selectedClassColourIdx='white'
        elif rbText=='Black':
            selectedClassColourCode=8
            curColourName='black'
            self.annotatorjObj.selectedClassColourIdx='black'
        else:
            print('Unexpected radio button value')

        selectedClassName=self.classListList.currentItem().text()
        # find it in the classnames list
        selectedClassIdxList=self.annotatorjObj.classFrameNames.index(selectedClassName)
        if (selectedClassIdxList<0):
            # didnt find it
            print("Could not find the currently selected class name in the list. Please try again.")
            return

        self.annotatorjObj.classFrameColours[selectedClassIdxList]=selectedClassColourCode
        print("Set selected class (\""+selectedClassName+"\") colour to "+rbText)
        # display currently selected class colour on the radiobuttons and label
        self.lblCurrentClass.setText(f'<html>Current: <font color="{curColourName}">{selectedClassName}</font></html>')

        # set all currently assigned ROIs of this group to have the new contour colour
        roiLayer=self.annotatorjObj.findROIlayer()
        n=len(roiLayer.data)
        print(f'self.annotatorjObj.selectedClassNameNumber: {self.annotatorjObj.selectedClassNameNumber}')
        for i in range(n):
            # selectGroup selects all ROIs if none belong the the arg class classNum --> check again
            if roiLayer.properties['class'][i]!=self.annotatorjObj.selectedClassNameNumber:
                continue

            roiLayer._data_view.update_face_color(i,[0,0,0,0])
            roiLayer._data_view.update_edge_color(i,self.annotatorjObj.colourString2Float(curColourName))

        roiLayer.refresh()
        roiLayer.refresh_text()


    # set radio button for class frame
    def setColourRadioButton(self,className,classIdx):
        if self is None:
            # class selection frame is not opened
            return

        curColourIdx=self.annotatorjObj.classFrameColours[classIdx]
        curColourName=None
        #debug:
        print(f'>>>coloridx: {curColourIdx}')

        # set radio buttons
        if curColourIdx==0:
            self.rdbtnGroup.setCurrentIndex(curColourIdx)
            curColourName='red'
            self.annotatorjObj.selectedClassColourIdx='red'
        elif curColourIdx==1:
            self.rdbtnGroup.setCurrentIndex(curColourIdx)
            curColourName='green'
            self.annotatorjObj.selectedClassColourIdx='green'
        elif curColourIdx==2:
            self.rdbtnGroup.setCurrentIndex(curColourIdx)
            curColourName='blue'
            self.annotatorjObj.selectedClassColourIdx='blue'
        elif curColourIdx==3:
            self.rdbtnGroup.setCurrentIndex(curColourIdx)
            curColourName='cyan'
            self.annotatorjObj.selectedClassColourIdx='cyan'
        elif curColourIdx==4:
            self.rdbtnGroup.setCurrentIndex(curColourIdx)
            curColourName='magenta'
            self.annotatorjObj.selectedClassColourIdx='magenta'
        elif curColourIdx==5:
            self.rdbtnGroup.setCurrentIndex(curColourIdx)
            curColourName='yellow'
            self.annotatorjObj.selectedClassColourIdx='yellow'
        elif curColourIdx==6:
            self.rdbtnGroup.setCurrentIndex(curColourIdx)
            curColourName='orange'
            self.annotatorjObj.selectedClassColourIdx='orange'
        elif curColourIdx==7:
            self.rdbtnGroup.setCurrentIndex(curColourIdx)
            curColourName='white'
            self.annotatorjObj.selectedClassColourIdx='white'
        elif curColourIdx==8:
            self.rdbtnGroup.setCurrentIndex(curColourIdx)
            curColourName='black'
            self.annotatorjObj.selectedClassColourIdx='black'
        else:
            print('Unexpected radio button value')
        
        # display currently selected class colour on the radiobuttons and label
        self.lblCurrentClass.setText(f'<html>Current: <font color="{curColourName}">{className}</font></html>')

        print("(\""+className+"\")'s colour: "+curColourName)
        return curColourName


    def defaultClassSelectionChanged(self,idx):
        # a default class was selected, assign all unassigned objects to this class

        # first save current classes
        '''
        if self.started and self.classMode and (self.imageFromArgs): #&& managerList.size()>0))
            # TODO: select current slice's rois
            #managerList.set(currentSliceIdx-1,manager)
            pass
        '''

        selectedClassName=self.comboBoxDefaultClass.currentText()
        print(f'Selected "{selectedClassName}" as default class')
        if selectedClassName=="(none)":
            # set no defaults
            self.annotatorjObj.defaultClassNumber=0 #-1
        else:
            # a useful class is selected
            self.annotatorjObj.defaultClassNumber=self.annotatorjObj.classNameLUT[selectedClassName]

            # set all unassigned objects to this class
            self.runDefaultClassSetting4allSlices()

            # show the latest opened roi stack again
            #if self.imageFromArgs: #(managerList!=null && managerList.size()>0)
                #updateROImanager(managerList.get(currentSliceIdx-1),showCnt)
                # TODO
                #pass


    # sets all objects with no currently assigned class (group) to the default class
    def setDefaultClass4objects(self):
        if not self.annotatorjObj.started or self.annotatorjObj.roiCount==0:
            print("Cannot find objects for the current image")
            return
        elif self.annotatorjObj.defaultClassNumber<1:
            print(f'Cannot set the default class to "{defaultClassNumber}". Must be >0.')
            return

        else:
            selectedClassNameVar=None
            # find its colour
            selectedClassNameVar='Class_{:02d}'.format(self.annotatorjObj.defaultClassNumber)
            tmpIdx=self.annotatorjObj.classFrameColours[self.annotatorjObj.classFrameNames.index(selectedClassNameVar)]
            self.annotatorjObj.defaultClassColour=self.getClassColour(tmpIdx)
            print(f'default class colour: {tmpIdx}')

            roiLayer=self.annotatorjObj.findROIlayer()
            n=len(roiLayer.data)

            for i in range(n):
                if roiLayer.properties['class'][i]<1:
                    # unclassified ROI
                    roiLayer.properties['class'][i]=self.annotatorjObj.defaultClassNumber
                    roiLayer._data_view.update_edge_color(i,self.annotatorjObj.colourString2Float(self.annotatorjObj.defaultClassColour))
            print(f'added them to the default class: {selectedClassNameVar}')

            # refresh text props
            roiLayer=self.annotatorjObj.findROIlayer()
            roiLayer.refresh()
            roiLayer.refresh_text()
        

    # loops through all sclies of the stack and sets the default class for all objects in all roi sets
    def runDefaultClassSetting4allSlices(self):
        # save the latest opened roi stack internally
        if self.annotatorjObj.started and self.annotatorjObj.classMode and self.annotatorjObj.imageFromArgs: #(managerList!=null and managerList.size()>0))
            pass

        # loop through all slices
        if self.annotatorjObj.imageFromArgs: #(managerList!=null and managerList.size()>0){
            # loop through them, then self.setDefaultClass4objects()
            pass
        else:
            self.setDefaultClass4objects()


    # unclassify all instances of currently selected class
    def unClassifyClass(self,classNum):
        #pluginInstance=AnnotatorJ(self.viewer)
        roiLayer=self.annotatorjObj.findROIlayer()
        n=len(roiLayer.data)
        print(f'found {n} rois in class {classNum}')
        for i in range(n):
            # selectGroup selects all ROIs if none belong the the arg class classNum --> check again
            if roiLayer.properties['class'][i]!=classNum:
                continue

            # --> unclassify it!
            roiLayer.properties['class'][i]=0
            roiLayer._data_view.update_face_color(i,[0,0,0,0])
            roiLayer._data_view.update_edge_color(i,self.annotatorjObj.colourString2Float(self.annotatorjObj.defColour)) # this is the default contour colour

            print(f'Selected "{roiLayer.properties["name"][i]}" ROI to unclassify (0)')

        # deselect the current ROI so the true class colour contour can be shown
        # no need, it doesn't get selected by default

        # refresh text props
        roiLayer=self.annotatorjObj.findROIlayer()
        roiLayer.refresh()
        roiLayer.refresh_text()


    # fetch color from the classidxlist
    def getClassColour(self,curColourIdx):
        curColour=''
        if curColourIdx==0:
            curColour='red'
        elif curColourIdx==1:
            curColour='green'
        elif curColourIdx==2:
            curColour='blue'
        elif curColourIdx==3:
            curColour='cyan'
        elif curColourIdx==4:
            curColour='magenta'
        elif curColourIdx==5:
            curColour='yellow'
        elif curColourIdx==6:
            curColour='orange'
        elif curColourIdx==7:
            curColour='white'
        elif curColourIdx==8:
            curColour='black'
        else:
            print('Unexpected class colour index')

        return curColour

# -------------------------------------
# end of class ClassesFrame
# -------------------------------------


# exporter frame
class ExportFrame(QWidget):
    def __init__(self,napari_viewer,annotatorjObj=None):
        super().__init__()
        self.viewer = napari_viewer
        self.annotatorjObj=annotatorjObj

        # check if there is opened instance of this frame
        # the main plugin is: 'napari-annotatorj: Annotator J'
        if self.annotatorjObj is not None and self.annotatorjObj.ExportFrame is not None:
            # already inited once, load again
            print('detected that Export widget has already been initialized')
            if self.annotatorjObj.ExportFrame.isVisible():
                print('Export widget is visible')
                return
            else:
                print('Export widget is not visible')
                # rebuild the widget



        # browse buttons
        self.btnBrowseOrig=QPushButton('Browse original ...')
        self.btnBrowseOrig.setToolTip('Browse folder of original images')
        self.btnBrowseOrig.clicked.connect(self.browseOriginal)
        self.btnBrowseROI=QPushButton('Browse annot ...')
        self.btnBrowseROI.setToolTip('Browse folder of annotation zip files')
        self.btnBrowseROI.clicked.connect(self.browseROI)

        # text fields
        self.textFieldOrig=QLineEdit()
        self.textFieldOrig.setToolTip('original images folder')
        self.textFieldOrig.editingFinished.connect(self.browseOriginal)
        self.textFieldROI=QLineEdit()
        self.textFieldROI.setToolTip('annotation zips folder')
        self.textFieldROI.editingFinished.connect(self.browseROI)

        # export options
        self.lblNewLabel=QLabel('Export options')
        self.chckbxMultiLabel=QCheckBox('Multi-label (instances)')
        self.chckbxMultiLabel.setChecked(True)
        self.chckbxMultiLabel.setToolTip('Single 1-channel image with increasing labels for instances')
        self.chckbxMultiLabel.stateChanged.connect(self.setMultiLabel)
        self.chckbxMultiLayer=QCheckBox('Multi-layer (stack)')
        self.chckbxMultiLayer.setToolTip('Single image (stack) with separate layers for instances')
        self.chckbxMultiLayer.stateChanged.connect(self.setMultiLayer)
        self.chckbxSemantic=QCheckBox('Semantic (binary)')
        self.chckbxSemantic.setToolTip('Single binary image (foreground-background)')
        self.chckbxSemantic.stateChanged.connect(self.setSemantic)

        self.chckbxCoordinates=QCheckBox('Coordinates')
        self.chckbxCoordinates.setToolTip('Bounding box coordinates')
        self.chckbxCoordinates.stateChanged.connect(self.setCoordinates)
        #add right click menu to choose bbox format: COCO/YOLO
        self.chckbxCoordinates.setContextMenuPolicy(Qt.CustomContextMenu)
        self.chckbxCoordinates.customContextMenuRequested.connect(self.addCoordsContextMenu)

        self.chckbxOverlay=QCheckBox('Overlay')
        self.chckbxOverlay.setToolTip('Annotations overlayed as outlines on the original image')
        self.chckbxOverlay.stateChanged.connect(self.setOverlay)


        # annot options
        self.lblObjectToExport=QLabel('Object to export')
        self.rdbtnRoi=QRadioButton('ROI')
        self.rdbtnRoi.setChecked(True)
        self.rdbtnRoi.toggled.connect(self.setAnnotRoi)
        self.rdbtnSemantic=QRadioButton('semantic')
        self.rdbtnSemantic.toggled.connect(self.setAnnotSemantic)
        self.rdbtnBoundingBox=QRadioButton('bounding box')
        self.rdbtnBoundingBox.toggled.connect(self.setAnnotBbox)


        # export buttons
        self.btnExportMasks=QPushButton('Export masks')
        self.btnExportMasks.setToolTip('Start exporting mask images')
        self.btnExportMasks.clicked.connect(self.startExport)
        self.btnCancel=QPushButton('Cancel')
        self.btnCancel.clicked.connect(self.cancelExport)


        self.ExportMainVBox=QVBoxLayout()
        self.ExportHeaderVBox=QVBoxLayout()
        self.ExportContentHBox=QHBoxLayout()
        self.ExportHeaderUpperHBox=QHBoxLayout()
        self.ExportHeaderLowerVBox=QVBoxLayout()
        self.ExportContentLeftVBox=QVBoxLayout()
        self.ExportContentRightVBox=QVBoxLayout()
        self.ExportContentRightObjVBox=QVBoxLayout()
        self.ExportContentLeftButtonHBox=QHBoxLayout()

        # add browse elements to self.ExportHeaderVBox
        self.ExportHeaderVBox.addWidget(self.textFieldOrig)
        self.ExportHeaderVBox.addWidget(self.textFieldROI)
        # add browse elements to self.ExportHeaderLowerVBox
        self.ExportHeaderLowerVBox.addWidget(self.btnBrowseOrig)
        self.ExportHeaderLowerVBox.addWidget(self.btnBrowseROI)

        # add label + checkboxes to self.ExportContentLeftVBox
        self.ExportContentLeftVBox.addWidget(self.lblNewLabel)
        self.ExportContentLeftVBox.addWidget(self.chckbxMultiLabel)
        self.ExportContentLeftVBox.addWidget(self.chckbxMultiLayer)
        self.ExportContentLeftVBox.addWidget(self.chckbxSemantic)
        self.ExportContentLeftVBox.addWidget(self.chckbxCoordinates)
        self.ExportContentLeftVBox.addWidget(self.chckbxOverlay)

        # add label + radio buttons to self.ExportContentRightObjVBox
        self.ExportContentRightObjVBox.addWidget(self.lblObjectToExport)
        self.ExportContentRightObjVBox.addWidget(self.rdbtnRoi)
        self.ExportContentRightObjVBox.addWidget(self.rdbtnSemantic)
        self.ExportContentRightObjVBox.addWidget(self.rdbtnBoundingBox)

        # add buttons to self.ExportContentLeftButtonHBox
        self.ExportContentLeftButtonHBox.addWidget(self.btnCancel)
        self.ExportContentLeftButtonHBox.addWidget(self.btnExportMasks)


        self.ExportContentRightObjVBox.setAlignment(Qt.AlignTop)
        self.ExportContentRightVBox.setAlignment(Qt.AlignRight)

        self.ExportContentRightVBox.addLayout(self.ExportContentRightObjVBox)
        self.ExportContentRightVBox.addLayout(self.ExportContentLeftButtonHBox)

        self.ExportHeaderUpperHBox.addLayout(self.ExportHeaderVBox)
        self.ExportHeaderUpperHBox.addLayout(self.ExportHeaderLowerVBox)
        #self.ExportHeaderVBox.addLayout(self.ExportHeaderUpperHBox)
        #self.ExportHeaderVBox.addLayout(self.ExportHeaderLowerVBox)

        self.ExportContentHBox.addLayout(self.ExportContentLeftVBox)
        self.ExportContentHBox.addLayout(self.ExportContentRightVBox)
        
        self.ExportMainVBox.addLayout(self.ExportHeaderUpperHBox)
        self.ExportMainVBox.addLayout(self.ExportContentHBox)

        self.setLayout(self.ExportMainVBox)
        #self.show()

        if self.annotatorjObj is not None:
            self.viewer.window.add_dock_widget(self,name='Export')



    def browseOriginal(self):
        # TODO
        return

    def browseROI(self):
        # TODO
        return

    def setMultiLabel(self):
        # TODO
        return

    def setMultiLayer(self):
        # TODO
        return

    def setSemantic(self):
        # TODO
        return

    def setCoordinates(self):
        # TODO
        return

    def addCoordsContextMenu(self):
        coordsContextMenu=QMenu()
        coco=coordsContextMenu.addAction('(default) COCO format')
        coco.triggered.connect(self.setCoco)
        yolo=coordsContextMenu.addAction('YOLO format')
        yolo.triggered.connect(self.setYolo)
        coordsContextMenu.exec_(QCursor.pos())


    def setCoco(self):
        self.chckbxCoordinates.setText('Coordinates (COCO)')
        self.bboxFormat=0
        print('COCO selected')

    def setYolo(self):
        self.chckbxCoordinates.setText('Coordinates (YOLO)')
        self.bboxFormat=1
        print('YOLO selected')


    def setOverlay(self):
        # TODO
        return


    def setAnnotRoi(self):
        # TODO
        return

    def setAnnotSemantic(self):
        # TODO
        return

    def setAnnotBbox(self):
        # TODO
        return


    def startExport(self):
        # TODO
        return

    def cancelExport(self):
        # TODO
        return

# -------------------------------------
# end of class ExportFrame
# -------------------------------------