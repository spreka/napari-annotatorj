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
import cv2
from copy import deepcopy,copy
#from napari.qt import create_worker #thread_worker
from napari.qt.threading import thread_worker #create_worker
from tqdm import tqdm

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
        self.annotEdgeWidth=0.5
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
        #self.modelFolder=os.path.join(os.path.dirname(__file__),'models')
        # init model folder opening
        self.modelFolder=self.initModelFolder()
        self.modelReleaseUrl='https://github.com/spreka/annotatorj/releases/download/v0.0.2-model/models.zip'
        self.DownloadProgressBar=None
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

        self.ColourSelector=None
        self.overlayColour='black'

        self.selectedClass=None
        self.prevSelectedClass=None
        self.propsClassString=['normal','cancerous']
        self.finishedSelection=False
        self.cancelledSaving=False
        self.newClassActive=False

        # options
        self.optionsFrame=None
        self.selectedAnnotationType='instance'
        self.semanticSaving=False
        self.paramFile='napari-annotatorj_config.json'
        self.rememberAnnotType=True
        self.saveOutlines=False
        self.saveAnnotTimes=False

        # read options from file if exists
        self.params=None
        self.initParams()

        # for dock tabs
        self.firstDockWidget=None
        self.firstDockWidgetName=None
        self.optionsWidget=None
        self.coloursWidget=None
        self.classesWidget=None
        self.ExportWidget=None


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
        self.btnOverlay.clicked.connect(self.setOverlay)

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
        self.buttonOptions.clicked.connect(self.openOptionsFrame)

        self.btnColours=QPushButton('Colours')
        self.btnColours.setToolTip('Set colour for annotations or overlay')
        self.btnColours.clicked.connect(self.addColourWidget)

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
        self.chkShowOverlay.stateChanged.connect(self.showOverlay)
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
        #self.vBoxRightDummy.addSpacing(62)
        self.btnDummy1=QPushButton()
        self.btnDummy2=QPushButton()
        self.btnDummy1.setEnabled(False)
        self.btnDummy2.setEnabled(False)
        self.vBoxRightDummy.addWidget(self.btnDummy1)
        self.vBoxRightDummy.addWidget(self.btnDummy2)
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
                    self.destNameRaw=os.path.join(self.defDir,self.defFile)
                    self.stepping=False
                else:
                    self.destNameRaw,_=QFileDialog.getOpenFileName(
                        self,"Select an image",
                        str(os.path.join(self.defDir,self.defFile)),"Images (*.png *.bmp *.jpg *.jpeg *.tif *.tiff *.gif)")

                print(self.destNameRaw)
                if os.path.exists(self.destNameRaw):
                    self.defDir=os.path.dirname(self.destNameRaw)
                    self.defFile=os.path.basename(self.destNameRaw)
                    img=skimage.io.imread(self.destNameRaw)
                    print('Opened file: {}'.format(self.destNameRaw))
                else:
                    print('Could not open file: {}'.format(self.destNameRaw))
                    return

                self.curPredictionImageName=self.defFile
                self.curPredictionImage=None
                self.curOrigImage=None

        imageLayer = self.viewer.add_image(img,name='Image')

        # check if a shapes layer already exists for the rois
        # if so, bring it forward
        roiLayer=self.findROIlayer(True)
        if roiLayer is None:
            # set roi edge width for visibility
            s=imageLayer.data.shape
            if (s[0]<=300) and (s[1]<=300):
                self.annotEdgeWidth=0.5
            elif (s[0]>300 and s[0]<=500) or (s[1]>300 and s[1]<=500):
                self.annotEdgeWidth=1.0
            elif (s[0]>500 and s[0]<=1000) or (s[1]>500 and s[1]<=1000):
                self.annotEdgeWidth=1.5
            elif (s[0]>1000 and s[0]<=1500) or (s[1]>1000 and s[1]<=1500):
                self.annotEdgeWidth=2.0
            elif (s[0]>1500 and s[0]<=2000) or (s[1]>1500 and s[1]<=2000):
                self.annotEdgeWidth=3.0
            elif (s[0]>2000 and s[0]<=3000) or (s[1]>2000 and s[1]<=3000):
                self.annotEdgeWidth=5.0
            else:
                self.annotEdgeWidth=7.0

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


        # fetch annotation type from settings
        types=['instance','bbox','semantic']
        validType=False
        if self.rememberAnnotType and (self.selectedAnnotationType is not None and self.selectedAnnotationType!=""):
            if self.selectedAnnotationType in types:
                # valid annot type, can continue
                validType=True
                print(f'Fetched annotation type: {self.selectedAnnotationType}')

        if not self.rememberAnnotType or not validType:
            # use default instance
            self.selectedAnnotationType='instance'

        if self.rememberAnnotType:
            # save it
            self.SaveNewProp('selectedAnnotationType',self.selectedAnnotationType)

        # instance annotation type
        if self.selectedAnnotationType=='instance':
            # set freehand selection tool by default
            roiLayer=self.findROIlayer()
            if roiLayer is None:
                self.initRoiManager()
                roiLayer=self.findROIlayer()
            roiLayer.mode='add_polygon'
            if self.freeHandROI not in roiLayer.mouse_drag_callbacks:
                roiLayer.mouse_drag_callbacks.append(self.freeHandROI)
            if not self.contAssist:
                # contAssist is off
                if not self.editMode:
                    # edit mode is off
                    if not self.classMode:
                        # class mode is off
                        # enable contour correction
                        #self.chckbxAddAutomatically.setEnabled(True)
                        #chckbxStepThroughContours.setEnabled(True)
                        self.chckbxContourAssist.setEnabled(True)
                    else:
                        # class mode is on
                        # disable the others
                        #self.chckbxAddAutomatically.setChecked(False)
                        #self.chckbxAddAutomatically.setEnabled(False)
                        #self.chckbxStepThroughContours.setChecked(False)
                        #self.chckbxStepThroughContours.setEnabled(False)
                        self.chckbxContourAssist.setChecked(False)
                        self.chckbxContourAssist.setEnabled(False)

                        self.chckbxClass.setEnabled(True)

                        self.editMode=False
                        self.addAuto=False
                        contAssist=False
                    
                else:
                    # edit mode is on
                    # disable contour correction
                    #self.chckbxAddAutomatically.setChecked(False)
                    #self.chckbxAddAutomatically.setEnabled(False)
                    #self.chckbxStepThroughContours.setEnabled(True)
                    self.chckbxContourAssist.setChecked(False)
                    self.chckbxContourAssist.setEnabled(False)

                    self.chckbxClass.setChecked(False)
                    self.chckbxClass.setEnabled(False)

                    self.addAuto=False
                    self.classMode=False

            else:
                # contAssist is on
                #self.chckbxAddAutomatically.setChecked(False)
                #self.chckbxAddAutomatically.setEnabled(False)
                #self.chckbxStepThroughContours.setChecked(False)
                #self.chckbxStepThroughContours.setEnabled(False)
                self.chckbxContourAssist.setEnabled(True)

                self.chckbxClass.setChecked(False)
                self.chckbxClass.setEnabled(False)

                self.editMode=False
                self.addAuto=False
                self.classMode=False
        
        # semantic painting annotation type
        elif self.selectedAnnotationType=='semantic':
            # disable contour correction
            self.addAuto=False
            self.editMode=False
            self.contAssist=False
            self.classMode=False
            #self.chckbxStepThroughContours.setChecked(False)
            self.chckbxContourAssist.setChecked(False)
            #self.chckbxAddAutomatically.setChecked(False)
            #self.chckbxAddAutomatically.setEnabled(False)
            self.chckbxContourAssist.setEnabled(False)
            #self.chckbxStepThroughContours.setEnabled(False)
            self.chckbxClass.setChecked(False)
            self.chckbxClass.setEnabled(False)

            # remove default ROI layer if present
            roiLayer=self.findROIlayer()
            if roiLayer is not None:
                self.viewer.layers.remove(roiLayer)
            # add labels layer for painting
            labelLayer=self.findLabelsLayerName(layerName='semantic')
            if labelLayer is None:
                imageLayer=self.findImageLayer()
                if imageLayer is None:
                    # this should never happen
                    print('No image opened yet')
                    return
                else:
                    s=imageLayer.data.shape
                    labelImage=numpy.zeros((s[0],s[1]),dtype='uint8')
                    labelLayer=self.viewer.add_labels(labelImage,name='semantic')
            labelLayer.mode='paint'
            labelLayer.brush_size=self.brushSize
            labelLayer.opacity=0.5


        # bounding box annotation
        elif self.selectedAnnotationType=='bbox':
            # set rectangle selection tool by default
            roiLayer=self.findROIlayer()
            if roiLayer is None:
                self.initRoiManager()
                roiLayer=self.findROIlayer()
            roiLayer.mode='add_rectangle'
            if self.freeHandROI in roiLayer.mouse_drag_callbacks:
                roiLayer.mouse_drag_callbacks.remove(self.freeHandROI)

            # disable contour correction
            self.editMode=False
            self.contAssist=False
            #self.chckbxAddAutomatically.setEnabled(True)
            #self.chckbxStepThroughContours.setChecked(False)
            self.chckbxContourAssist.setChecked(False)
            #self.chckbxStepThroughContours.setEnabled(False)
            self.chckbxContourAssist.setEnabled(False)
            self.chckbxClass.setEnabled(True)


        # TODO: add missing settings


        # reset contour assist layer
        self.inAssisting=False
        if self.contAssist:
            self.setContourAssist(Qt.Checked)
        elif not self.contAssist and self.selectedAnnotationType!='semantic':
            self.setContourAssist(False)

        self.overlayAdded=False

        self.startedEditing=False
        self.origEditedROI=None
        self.closeingOnPurpuse=False

        # when open function finishes:
        self.started=True


    def loadROIs(self):
        # temporarily load a test ImageJ ROI.zip file with contours created in ImageJ and saved with AnnotatorJ
        # later this will start a browser dialog to select the annotation file
        if self.testMode==True:
            if os.path.exists(self.test_rois):
                rois=ImagejRoi.fromfile(self.test_rois)
                print('Test roi file read successfully')

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
        print('After loading we have '+str(curROInum)+' contours')
        self.roiCount=curROInum

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
        shapesLayer=Shapes(data=numpy.array([[0,0],[1,1]]),shape_type='polygon',name='ROI',edge_width=self.annotEdgeWidth,edge_color='white',face_color=[0,0,0,0],properties=roiProps,text=roiTextProps)
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


    def extractROIdata(self,rois,layerName='ROI'):
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
                if layerName=='ROI':
                    curColour=None
                    if self.classColourLUT is None:
                        self.initClassColourLUT(rois)
                    
                    curColour=self.classColourLUT[curClass]
                    roiColours.append(curColour)

                    # store class info for the Classes widget
                    if self.listModelClasses is None:
                        self.listModelClasses=[]
                        self.selectedClassNameNumber=1
                        self.classFrameColours=[]
                    newName='Class_{:02d}'.format(curClass)
                    if newName not in self.listModelClasses and newName not in self.classFrameNames:
                        self.listModelClasses.append(newName)
                        self.classFrameNames.append(newName)
                        self.classFrameColours.append(curClass-1)
                else:
                    roiColours.append(self.overlayColour)
                roiProps['class'].append(curClass)
            else:
                if layerName=='ROI':
                    roiColours.append(self.defColour)
                else:
                    roiColours.append(self.overlayColour)
                roiProps['class'].append(0)

            # store the roi's name
            roiProps['name'].append(curROI.name)
            roiProps['nameInt'].append(int(curROI.name))

            # TODO: fetch more data from the rois

        # rename any existing ROI layers so that this one is the new default
        self.renameROIlayers(layerName=layerName)

        # fill (face) colour of rois is transparent by default, only the contours are visible
        if layerName=='ROI':
            fillColour=[0,0,0,0]
        else:
            fillColour=self.colourString2Float(self.overlayColour)
            fillColour[-1]=0.5
        # edge_width=0.5 actually sets it to 1
        shapesLayer = Shapes(data=roiList,shape_type=roiType,name=layerName,edge_width=self.annotEdgeWidth,edge_color=roiColours,face_color=fillColour,properties=roiProps,text=roiTextProps)

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


    def initModelFolder(self):
        # return a path where Keras U-Net models are located
        defModelPath=os.path.join(os.path.dirname(__file__),'models')
        if os.path.isdir(defModelPath):
            pass
        else:
            # try to use a user folder
            userHome=os.path.expanduser("~") 
            defModelPath=os.path.join(userHome,'.napari_annotatorj','models')
            if os.path.isdir(defModelPath):
                pass
            else:
                # create it
                try:
                    os.makedirs(defModelPath)
                except Exception as e:
                    print(e)
                if not os.path.isdir(defModelPath):
                    print(f'Failed to create model folder: {defModelPath}')
                    defModelPath=None
                else:
                    print(f'Created model folder: {defModelPath}')
                    pass
        return defModelPath


    def downloadModelRelease(self,):
        # download models.zip from repo
        import urllib.request
        #from napari.utils import progress
        modelFolderRoot=os.path.dirname(self.modelFolder)
        print(f'Downloading model file...')
        try:
            with DownloadProgressBar(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
              desc='model download') as t:
                urllib.request.urlretrieve(self.modelReleaseUrl,filename=os.path.join(modelFolderRoot,'models.zip'),reporthook=t.update_to)
                t.total=t.n
        except Exception as e:
            print(e)

        # extract archive
        import zipfile
        modelZipFile=os.path.join(modelFolderRoot,'models.zip')
        if os.path.isfile(modelZipFile):
            print(f'Extracting model files from .zip archive...')
            with zipfile.ZipFile(modelZipFile, 'r') as modelZip:
                modelZip.extractall(modelFolderRoot)
        else:
            print(f'Model zip file "models.zip" does not exist in location {modelFolderRoot}')
            return None,None

        modelName='model_real'
        importMode=-1
        # check if the correct files exist in the model folder
        if os.path.isfile(os.path.join(self.modelFolder,modelName+'.json')) and os.path.isfile(os.path.join(self.modelFolder,modelName+'_weights.h5')):
            print('  >> importing from json config + weights .h5 files...')
            importMode=0
        elif os.path.isfile(os.path.join(self.modelFolder,modelName+'.hdf5')):
            print('  >> importing from a single .hdf5 file...')
            importMode=1
        else:
            return None,None
        print(f'Downloaded pre-trained model successfully to: {self.modelFolder}')
        return modelName,importMode


    def initParams(self):
        optionsFile=os.path.join(self.modelFolder,self.paramFile)
        import json
        if not os.path.isfile(optionsFile):
            print('Params file doesn\'t exist, loading default config:')
            # defaults are already set in the class __init__ fcn
            pass
        else:
            print(f'Found params file {optionsFile}, reading config...')
            self.collectParams(setParams=True,readParams=True,paramsFile=optionsFile)

        self.printConfig()


    def collectParams(self,setParams=False,readParams=False,paramsFile=None):
        if readParams and paramsFile is not None:
            import json
            try:
                with open(paramsFile) as paramFile:
                    jsonString=json.load(paramFile)
                    #debug:
                    print(jsonString)
                    params=copy(jsonString)
                    self.setParams(params)
                    print('Read params:')
            except Exception as e:
                print(e)
                print(f'Could not read json file {paramsFile}')
                params=None
        else:
            params={
                'defaultAnnotType': self.selectedAnnotationType,
                'rememberAnnotTyperememberAnnotType': self.rememberAnnotType,
                'defColour':self.defColour,
                'overlayColor':self.overlayColour,
                'classes':self.propsClassString,
                'intensityThreshVal':self.intensityThreshVal,
                'intensityThreshValR':self.intensityThreshValR,
                'intensityThreshValG':self.intensityThreshValG,
                'intensityThreshValB':self.intensityThreshValB,
                'selectedCorrMethod':self.selectedCorrMethod,
                'distanceThreshVal':self.distanceThreshVal,
                'modelFullFile':self.modelFullFile,
                'modelJsonFile':self.modelJsonFile,
                'modelFolder':self.modelFolder,
                'modelWeightsFile':self.modelWeightsFile,
                'correctionBrushSize':self.correctionBrushSize,
                'semanticBrushSize':self.semanticBrushSize,
                'saveAnnotTimes':self.saveAnnotTimes,
                'autoMaskLoad':self.autoMaskLoad,
                'enableMaskLoad':self.enableMaskLoad,
                'saveOutlines':self.saveOutlines,
                'gpuSetting':self.gpuSetting
            }

        if setParams:
            self.params=params
        return params


    def setParams(self,params):
        if 'defaultAnnotType' in params:
            validTypes=['instance','bbox','semantic']
            if params['defaultAnnotType'] in validTypes:
                self.selectedAnnotationType=params['defaultAnnotType']
            else:
                # set to default
                self.selectedAnnotationType='instance'
        if 'rememberAnnotTyperememberAnnotType' in params and isinstance(params['rememberAnnotTyperememberAnnotType'],bool):
            self.rememberAnnotType=params['rememberAnnotTyperememberAnnotType']
        if 'defColour' in params:
            self.defColour=params['defColour']
        if 'overlayColor' in params:
            self.overlayColour=params['overlayColor']
        if 'classes' in params:
            self.propsClassString=params['classes']
        if 'intensityThreshVal' in params:
            self.intensityThreshVal=params['intensityThreshVal']
        if 'intensityThreshValR' in params:
            self.intensityThreshValR=params['intensityThreshValR']
        if 'intensityThreshValG' in params:
            self.intensityThreshValG=params['intensityThreshValG']
        if 'intensityThreshValB' in params:
            self.intensityThreshValB=params['intensityThreshValB']
        if 'selectedCorrMethod' in params:
            self.selectedCorrMethod=params['selectedCorrMethod']
        if 'distanceThreshVal' in params:
            self.distanceThreshVal=params['distanceThreshVal']
        if 'modelFullFile' in params:
            self.modelFullFile=params['modelFullFile']
        if 'modelJsonFile' in params:
            self.modelJsonFile=params['modelJsonFile']
        if 'modelFolder' in params:
            self.modelFolder=params['modelFolder']
        if 'modelWeightsFile' in params:
            self.modelWeightsFile=params['modelWeightsFile']
        if 'correctionBrushSize' in params:
            self.correctionBrushSize=params['correctionBrushSize']
        if 'semanticBrushSize' in params:
            self.semanticBrushSize=params['semanticBrushSize']
        if 'saveAnnotTimes' in params and isinstance(params['saveAnnotTimes'],bool):
            self.saveAnnotTimes=params['saveAnnotTimes']
        if 'autoMaskLoad' in params and isinstance(params['autoMaskLoad'],bool):
            self.autoMaskLoad=params['autoMaskLoad']
        if 'enableMaskLoad' in params and isinstance(params['enableMaskLoad'],bool):
            self.enableMaskLoad=params['enableMaskLoad']
        if 'saveOutlines' in params and isinstance(params['saveOutlines'],bool):
            self.saveOutlines=params['saveOutlines']
        if 'gpuSetting' in params:
            self.gpuSetting=params['gpuSetting']
        return


    def printConfig(self):
        print(self.params)
        return


    def SaveNewProp(self,prop,val):
        if self.params is None:
            self.collectParams(setParams=True)

        self.writeParams2File()


    def writeParams2File(self):
        optionsFile=os.path.join(self.modelFolder,self.paramFile)
        import json
        if not os.path.isfile(optionsFile):
            # create it
            print('Cannot find config file, createing it now...')
        else:
            # rewrite it
            print(f'Found config file {optionsFile}, updating it now...')

        with open(optionsFile,'w') as paramFile:
            # write params to file
            try:
                json.dump(self.collectParams(setParams=True), paramFile,indent=4)
                print('Successfully wrote config file with current settings')
            except Exception as e:
                print(e)
                print('Could not write configs to file')


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

    def renameROIlayers(self,layerName='ROI'):
        for x in self.viewer.layers:
            if (x.__class__ is Shapes and x.name==layerName):
                print('{} was a ROI shapes layer'.format(x.name))
                # rename it
                newName=layerName+'_prev'
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
            layer.add(data=yx,shape_type=roiType,edge_width=self.annotEdgeWidth,edge_color=curColour,face_color=[0,0,0,0])#,properties=roiProps,text=roiTextProps)

            # TODO: fetch more data from the rois

            # add the new roi's props to the object:
            prevProps['class'].append(roiProps['class'])
            prevProps['name'].append(roiProps['name'])
            prevProps['nameInt'].append(roiProps['nameInt'])

        # set the props after adding the rois
        layer.properties=prevProps


        

    def setTestMode(self,mode=False):
        self.testMode=mode
        if mode==True:
            self.defDir=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),'demo')
            print(f'Set input image folder to {self.defDir} in testMode')
            self.defFile='img.png'

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
                    self.selectedClass='masks'

                else:

                    # ask class name in dialog box
                    self.popClassSelection()
                    #self.selectedClass='masks'
            else:
                # create new frame for optional extra element adding manually by the user (for new custom class):
                self.popClassSelection()
                #self.selectedClass='masks'

            
        else:
            # roi stack was imported, save to mask names
            # TODO: do this branch
            self.selectedClass='masks'


        if self.cancelledSaving:
            # abort saving
            return

        print(f'Set class: {self.selectedClass}')

        # set output folder and create it
        destMaskFolder2=os.path.join(self.defDir,self.selectedClass)
        os.makedirs(destMaskFolder2,exist_ok=True)
        print('Created output folder: {}'.format(destMaskFolder2))

        # set output file name according to annotation type:
        # TODO: add the others
        if self.selectedAnnotationType=='instance':
            # now we only have instance
            roiFileName=str(os.path.join(destMaskFolder2,'{}_ROIs.zip'.format(os.path.splitext(os.path.basename(self.destNameRaw))[0])))
        elif self.selectedAnnotationType=='bbox':
            roiFileName=str(os.path.join(destMaskFolder2,'{}_bboxes.zip'.format(os.path.splitext(os.path.basename(self.destNameRaw))[0])))
        elif self.selectedAnnotationType=='semantic':
            roiFileName=str(os.path.join(destMaskFolder2,'{}_semantic.tiff'.format(os.path.splitext(os.path.basename(self.destNameRaw))[0])))
            self.semanticSaving=True

        if not self.semanticSaving:
            print('Set output ROI.zip name: {}'.format(roiFileName))
        else:
            print('Set output binary image name: {}'.format(roiFileName))

        # check if annotation already exists for this image with this class
        if os.path.exists(roiFileName):
            # TODO: pop dialog to overwrite,rename,cancel
            newFileNum=0
            while os.path.exists(roiFileName):
                newFileNum+=1
                if self.selectedAnnotationType=='instance':
                    roiFileName=str(os.path.join(destMaskFolder2,'{}_ROIs_{}.zip'.format(os.path.splitext(os.path.basename(self.destNameRaw))[0],newFileNum)))
                elif self.selectedAnnotationType=='bbox':
                    roiFileName=str(os.path.join(destMaskFolder2,'{}_bboxes_{}.zip'.format(os.path.splitext(os.path.basename(self.destNameRaw))[0],newFileNum)))
                elif self.selectedAnnotationType=='semantic':
                    roiFileName=str(os.path.join(destMaskFolder2,'{}_semantic_{}.tiff'.format(os.path.splitext(os.path.basename(self.destNameRaw))[0],newFileNum)))

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

        if not self.semanticSaving:
            # new way with roi list
            rois2save=self.fetchShapes2ROIs()
            if rois2save is None:
                # nothing to save
                print('Failed to save ROI: {}'.format(roiFileName))
            else:
                roiwrite(roiFileName,rois2save)
                print('Saved ROI: {}'.format(roiFileName))

            if self.stepping:
                self.finishedSaving=True
        # save a semantic annotation image
        else:
            labelLayer=self.findLabelsLayerName(layerName='semantic')
            if labelLayer is not None and len(labelLayer.data)>0:
                try:
                    import tifffile
                    tifffile.imwrite(roiFileName,labelLayer.data.astype(bool),photometric='minisblack')
                    print(f'Saved binary image: {roiFileName}')
                except Exception as e:
                    print(e)
                    print(f'Failed to save binary image: {roiFileName}')


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
                layer.add(data=freeCoords,shape_type='polygon',edge_width=self.annotEdgeWidth,edge_color=self.defColour,face_color=[0,0,0,0])
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
        
        if pos[0]<=0 or pos[1]<=0 or pos[0]>s[0] or pos[1]>s[1]:
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
                        # try to find the largest contour in the list
                        try:
                            f=(lambda x: len(x))
                            lengths=[f(c) for c in contour]
                            contour=contour[lengths.index(max(lengths))]
                        except Exception as e:
                            print(e)
                            warnings.warn(msg)
                            return None
                        msg=msg+' - selecting largest contour'
                        print(msg)
                        #return None
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
                        # try to find the largest contour in the list
                        try:
                            f=(lambda x: len(x))
                            lengths=[f(c) for c in contour]
                            contour=contour[lengths.index(max(lengths))]
                        except Exception as e:
                            print(e)
                            warnings.warn(msg)
                            return None
                        msg=msg+' - selecting largest contour'
                        print(msg)
                        #return None
                else:
                    msg='Cannot create roi from this label'
                    warnings.warn(msg)
                    return None

            shape=numpy.array(numpy.fliplr(numpy.squeeze(contour)))
            roiLayer=self.findROIlayer()
            roiLayer.add_polygons(shape,edge_color=self.defColour,face_color=None,edge_width=self.annotEdgeWidth)
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
        #selectedClass='masks'
        mainExportFolder='labelled_masks'
        exportFolder=os.path.join(self.defDir,self.selectedClass,mainExportFolder)
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

                    # close overlay if any
                    overlayLayer=self.findROIlayer(layerName='overlay')
                    if overlayLayer is not None:
                        # delete it
                        self.viewer.layers.remove(overlayLayer)


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
                # try to download it from the original AnnotatorJ repo's releases
                self.modelJsonFile,importMode=self.downloadModelRelease()
                if self.modelJsonFile is None:
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
            from .predict_unet import callPredictUnet,callPredictUnetLoadedNoset,loadUnetModelSetGpu,callPredictUnetLoadedNosetCustomSize
        except ImportError as e:
            try:
                from predict_unet import callPredictUnet,callPredictUnetLoadedNoset,loadUnetModelSetGpu,callPredictUnetLoadedNosetCustomSize
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
                warnings.warn('Cannot find image')
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
            #predictedImage=callPredictUnetLoadedNoset(self.trainedUNetModel,self.curOrigImage)
            predictedImage=callPredictUnetLoadedNosetCustomSize(self.trainedUNetModel,self.curOrigImage)
            
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

            if pos[0]<=0 or pos[1]<=0 or pos[0]>s[0] or pos[1]>s[1]:
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
                    #layer._set_highlight(force=True)
                    
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

            shapesLayer.mode='select'
            # add mouse click callback to classify ROIs
            if self.customShapesLayerSelect not in shapesLayer.mouse_drag_callbacks and mouse_bindings.select in shapesLayer.mouse_drag_callbacks:
                shapesLayer.mouse_drag_callbacks.append(self.customShapesLayerSelect)
                shapesLayer.mouse_drag_callbacks.remove(mouse_bindings.select)
            if self.classifyROI not in shapesLayer.mouse_drag_callbacks:
                shapesLayer.mouse_drag_callbacks.append(self.classifyROI)


        else:
            self.classMode=False
            print('Class mode cleared')

            self.chckbxContourAssist.setEnabled(True)
            if self.classesFrame is not None:
                self.classesFrame.closeClassesFrame()

            # restore default 'select' mouse click callback
            if mouse_bindings.select not in shapesLayer.mouse_drag_callbacks and self.customShapesLayerSelect in shapesLayer.mouse_drag_callbacks:
                shapesLayer.mouse_drag_callbacks.append(mouse_bindings.select)
                shapesLayer.mouse_drag_callbacks.remove(self.customShapesLayerSelect)
            if self.classifyROI in shapesLayer.mouse_drag_callbacks:
                shapesLayer.mouse_drag_callbacks.remove(self.classifyROI)

            shapesLayer.mode='add_polygon'


    def showOverlay(self,state):
        overlayLayer=self.findROIlayer(layerName='overlay')
        if state == Qt.Checked:
            self.showOvl=True
            print('Show overlay selected')

            if self.overlayAdded and overlayLayer is not None:
                overlayLayer.visible=True
        else:
            self.showOvl=False
            print('Show overlay cleared')

            if self.overlayAdded and overlayLayer is not None:
                overlayLayer.visible=False


    def addColourWidget(self):
        if self.ColourSelector is None:
            self.ColourSelector=ColourSelector(self.viewer,annotatorjObj=self)
        else:
            # try to find 'x'-button closed version
            if 'Colours' in self.viewer.window._dock_widgets.data:
                print('Colour widget already open')
            else:
                # open it
                self.ColourSelector=ColourSelector(self.viewer,annotatorjObj=self)


    def setOverlay(self):
        # browse an ImageJ ROI zip file as an 'overlay' (actually also a shapes layer in napari)
        if not self.started or (self.findImageLayer() is None or self.findImageLayer().data is None):
            warnings.warn('Open an image and annotate it first')
            return

        # check if we have annotations in the list before loading anything to it
        roiLayer=self.findROIlayer(layerName='overlay')
        curROInum=len(roiLayer.data) if roiLayer is not None else 0
        print('Before loading we had '+str(curROInum)+' contours on overlay');
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
            shapesLayer=self.extractROIdata(rois,layerName='overlay')
            if shapesLayer is not None:
                self.viewer.add_layer(shapesLayer)
                print('Loaded {} ROIs successfully on overlay'.format(len(rois)))
                self.overlayAdded=True

            roiLayer=self.findROIlayer(True)

            # select the "select shape" mode from the controls by default
            #shapesLayer.mode = 'select'
            # select the "add polygon" mode from the controls by default to enable freehand ROI drawing
            roiLayer.mode = 'add_polygon'

            self.viewer.reset_view()

        self.showOvl=True
        self.chkShowOverlay.setChecked(True)


    def popClassSelection(self):
        self.prevSelectedClass=deepcopy(self.selectedClass)

        self.classSelectionDialog=QDialog()
        self.classSelectionDialog.setModal(True)
        self.classSelectionDialog.setWindowTitle('Select class')

        self.classNameLabel=QLabel('Select class of objects')
        self.classLabel=QLabel('class:')
        self.classLabel.setToolTip('Class of your annotated objects')
        self.newLabel=QLabel('new:')
        self.classLabel.setToolTip('New custom class to create')
        self.newTextField=QLineEdit()
        self.newTextField.setEnabled(False)
        self.newTextField.setToolTip('Name of the new class')

        baseClassArray=['normal','cancerous','other...']
        self.classList=QComboBox()
        if self.propsClassString==baseClassArray[0:2]:
            for el in baseClassArray:
                self.classList.addItem(el)
        else:
            for el in self.propsClassString:
                self.classList.addItem(el)
            self.classList.addItem('other...')

        self.classList.setCurrentIndex(0)

        self.classList.currentIndexChanged.connect(self.saveClassSelectionChanged)

        self.btnClassSelectionOk=QPushButton('OK')
        self.btnClassSelectionOk.setToolTip('Continue to save')
        self.btnClassSelectionOk.clicked.connect(self.saveClassSelectionOk)
        self.btnClassSelectionCancel=QPushButton('Cancel')
        self.btnClassSelectionCancel.setToolTip('Cancel saving')
        self.btnClassSelectionCancel.clicked.connect(self.saveClassSelectionCancel)

        self.popMainVbox=QVBoxLayout()
        self.popContentHboxSelector=QHBoxLayout()
        self.popContentHboxBtns=QHBoxLayout()
        self.popLeftVbox=QVBoxLayout()
        self.popRightVbox=QVBoxLayout()


        self.popLeftVbox.addWidget(self.classLabel)
        self.popLeftVbox.addWidget(self.newLabel)

        self.popRightVbox.addWidget(self.classList)
        self.popRightVbox.addWidget(self.newTextField)

        self.popContentHboxSelector.addLayout(self.popLeftVbox)
        self.popContentHboxSelector.addLayout(self.popRightVbox)

        self.popContentHboxBtns.addWidget(self.btnClassSelectionOk)
        self.popContentHboxBtns.addWidget(self.btnClassSelectionCancel)

        self.popMainVbox.addWidget(self.classNameLabel)
        self.popMainVbox.addLayout(self.popContentHboxSelector)
        self.popMainVbox.addLayout(self.popContentHboxBtns)

        self.classSelectionDialog.setLayout(self.popMainVbox)
        self.classSelectionDialog.exec()


    def saveClassSelectionChanged(self,idx):
        #
        if idx==self.classList.count()-1:
            self.newTextField.setEnabled(True)
            self.newClassActive=True
        else:
            self.newTextField.setEnabled(False)
            self.newClassActive=False


    def saveClassSelectionOk(self):
        if self.newClassActive:
            # read from the text field
            textVal=self.newTextField.text()
            if textVal is None or textVal=='null' or len(textVal)==0:
                # empty string, warn the user to type something
                self.finishedSelection=False
                warnings.warn('Please enter a new class name or select one from the list to continue.')
            else:
                self.selectedClass=textVal
                self.finishedSelection=True
                self.cancelledSaving=False
                self.propsClassString.append(textVal)
                self.SaveNewProp('classes',textVal)
                self.closeClassSelectionFrame()

        else:
            # get from the list
            self.selectedClass=self.classList.currentText()
            self.finishedSelection=True
            self.cancelledSaving=False
            self.closeClassSelectionFrame()


    def saveClassSelectionCancel(self):
        self.cancelClassSelection()


    def cancelClassSelection(self):
        # remember that saving was cancelled here!
        self.cancelledSaving=True
        print('Cancelled saving')

        self.selectedClass=deepcopy(self.prevSelectedClass)
        self.closeClassSelectionFrame()


    def closeClassSelectionFrame(self):
        # close the progress window
        if self.classSelectionDialog is not None:
            self.classSelectionDialog.done(1)
            self.classSelectionDialog=None


    # modified version of napari.layers.shapes._shapes_mouse_bindings.select that disables the bounding box around selected shapes when clicked
    def customShapesLayerSelect(self,layer,event):
        """Select shapes or vertices either in select or direct select mode.

        Once selected shapes can be moved or resized, and vertices can be moved
        depending on the mode. Holding shift when resizing a shape will preserve
        the aspect ratio.
        """
        shift = 'Shift' in event.modifiers
        # on press
        value = layer.get_value(event.position, world=True)
        layer._moving_value = copy(value)
        shape_under_cursor, vertex_under_cursor = value
        if vertex_under_cursor is None:
            if shift and shape_under_cursor is not None:
                if shape_under_cursor in layer.selected_data:
                    layer.selected_data.remove(shape_under_cursor)
                else:
                    if len(layer.selected_data):
                        # one or more shapes already selected
                        layer.selected_data.add(shape_under_cursor)
                    else:
                        # first shape being selected
                        layer.selected_data = {shape_under_cursor}
            elif shape_under_cursor is not None:
                if shape_under_cursor not in layer.selected_data:
                    layer.selected_data = {shape_under_cursor}
            else:
                layer.selected_data = set()
        #layer._set_highlight()

        # we don't update the thumbnail unless a shape has been moved
        update_thumbnail = False
        yield

        # on move
        while event.type == 'mouse_move':
            coordinates = layer.world_to_data(event.position)
            # ToDo: Need to pass moving_coordinates to allow fixed aspect ratio
            # keybinding to work, this should be dropped
            layer._moving_coordinates = coordinates
            # Drag any selected shapes
            if len(layer.selected_data) == 0:
                mouse_bindings._drag_selection_box(layer, coordinates)
            else:
                mouse_bindings._move(layer, coordinates)

            # if a shape is being moved, update the thumbnail
            if layer._is_moving:
                update_thumbnail = True
            yield

        # only emit data once dragging has finished
        if layer._is_moving:
            layer.events.data(value=layer.data)

        # on release
        shift = 'Shift' in event.modifiers
        if not layer._is_moving and not layer._is_selecting and not shift:
            if shape_under_cursor is not None:
                layer.selected_data = {shape_under_cursor}
            else:
                layer.selected_data = set()
        elif layer._is_selecting:
            layer.selected_data = layer._data_view.shapes_in_box(layer._drag_box)
            layer._is_selecting = False
            #layer._set_highlight()

        layer._is_moving = False
        layer._drag_start = None
        layer._drag_box = None
        layer._fixed_vertex = None
        layer._moving_value = (None, None)
        #layer._set_highlight()

        if update_thumbnail:
            layer._update_thumbnail()


    def openOptionsFrame(self):
        self.optionsFrame=OptionsFrame(self.viewer,self)


    def findDockWidgets(self,newWidgetName):
        wCount=0
        widgets=[]
        for w in self.viewer.window._dock_widgets.data:
            # only care for the extra widgets apart from AnnotatorJ and AnnotatorJExport
            if w=='napari-annotatorj: AnnotatorJ':
                continue
            elif w=='napari-annotatorj: AnnotatorJExport':
                continue
            else:
                wCount+=1
                widgets.append(w)

        if newWidgetName in widgets: #and self.firstDockWidgetName==newWidgetName:
            widgets.remove(newWidgetName)
        else:
            print(f'Cannot find widget {newWidgetName} in napari')
            #debug:
            print(widgets)
        if len(widgets)>0:
            newName=widgets[0]
            self.firstDockWidgetName=newName
            print(f'Resetting firstDockWidgetName to {newName}')
            if newName=='Classes':
                self.firstDockWidget=self.classesWidget
            elif newName=='Options':
                self.firstDockWidget=self.optionsWidget
            elif newName=='Colours':
                self.firstDockWidget=self.coloursWidget
            elif newName=='Export' or newName=='napari-annotatorj: AnnotatorJExport':
                # this should never happen
                self.firstDockWidget=self.ExportWidget
            else:
                self.firstDockWidget=None
                self.firstDockWidgetName=None
                print('Resetting firstDockWidgetName to None')
        else:
            self.firstDockWidget=None
            self.firstDockWidgetName=None
            print('Resetting firstDockWidgetName to None')
            


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
            if self.annotatorjObj.classesFrame.isVisible() or 'Classes' in self.viewer.window._dock_widgets.data:
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

            self.classListList.setCurrentItem(self.classListList.item(0))

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
        dw=self.viewer.window.add_dock_widget(self,name='Classes')
        self.annotatorjObj.classesWidget=dw
        if self.annotatorjObj.firstDockWidget is None:
            self.annotatorjObj.firstDockWidget=dw
            self.annotatorjObj.firstDockWidgetName='Classes'
        else:
            try:
                self.viewer.window._qt_window.tabifyDockWidget(self.annotatorjObj.firstDockWidget,dw)
            except Exception as e:
                print(e)
                # RuntimeError: wrapped C/C++ object of type QtViewerDockWidget has been deleted
                # try to reset the firstDockWidget manually
                self.annotatorjObj.findDockWidgets('Classes')
                try:
                    if self.annotatorjObj.firstDockWidget is None:
                        self.annotatorjObj.firstDockWidget=dw
                        self.annotatorjObj.firstDockWidgetName='Classes'
                    else:
                        self.viewer.window._qt_window.tabifyDockWidget(self.annotatorjObj.firstDockWidget,dw)
                except Exception as e:
                    print(e)
                    print('Failed to add widget Classes')


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


    def closeEvent(self, event):
        event.ignore()
        self.closeClassesFrame()
        #event.accept()


    def closeClassesFrame(self):
        try:
            if self.annotatorjObj.firstDockWidgetName=='Classes':
                self.annotatorjObj.findDockWidgets('Classes')
            self.viewer.window.remove_dock_widget(self.annotatorjObj.classesFrame)
            self.annotatorjObj.classesWidget=None
        except Exception as e:
            print(e)
            try:
                if self.annotatorjObj.firstDockWidgetName=='Classes':
                    self.annotatorjObj.findDockWidgets('Classes')
                self.viewer.window.remove_dock_widget('Classes')
                self.annotatorjObj.classesWidget=None
            except Exception as e:
                print(e)
                print('Failed to remove widget named Classes')

# -------------------------------------
# end of class ClassesFrame
# -------------------------------------


# colour selection widget
class ColourSelector(QWidget):
    def __init__(self,napari_viewer,annotatorjObj=None):
        super().__init__()
        self.viewer = napari_viewer
        self.annotatorjObj=annotatorjObj

        # check if there is opened instance of this frame
        # the main plugin is: 'napari-annotatorj: Annotator J'
        if self.annotatorjObj is not None and self.annotatorjObj.ColourSelector is not None:
            # already inited once, load again
            print('detected that Colour widget has already been initialized')
            if self.annotatorjObj.ColourSelector.isVisible() or 'Colours' in self.viewer.window._dock_widgets.data:
                print('Colour widget is visible')
                return
            else:
                print('Colour widget is not visible')
                # rebuild the widget

        # UI elements
        self.setWindowTitle('Select contour colours')
        self.annotLabel=QLabel('annotation:')
        self.overlayLabel=QLabel('overlay:')
        self.annotColourBox=QComboBox()
        self.overlayColourBox=QComboBox()
        self.addColours(self.annotColourBox)
        self.addColours(self.overlayColourBox)
        self.annotColourBox.setCurrentText(self.annotatorjObj.defColour)
        self.overlayColourBox.setCurrentText(self.annotatorjObj.overlayColour)
        self.annotColourBox.currentIndexChanged.connect(self.updateAnnotColour)
        self.overlayColourBox.currentIndexChanged.connect(self.updateOverlayColour)

        self.btnOk=QPushButton('Ok')
        self.btnOk.clicked.connect(self.updateColour)
        self.btnCancel=QPushButton('Cancel')
        self.btnCancel.clicked.connect(self.closeWidget)


        self.classMainVBox=QVBoxLayout()
        self.classHeaderHBox=QHBoxLayout()
        self.classContentHBox=QHBoxLayout()
        self.labelLeftVBox=QVBoxLayout()
        self.comboRightVBox=QVBoxLayout()

        self.labelLeftVBox.addWidget(self.annotLabel)
        self.labelLeftVBox.addWidget(self.overlayLabel)

        self.comboRightVBox.addWidget(self.annotColourBox)
        self.comboRightVBox.addWidget(self.overlayColourBox)

        self.classContentHBox.addWidget(self.btnOk)
        self.classContentHBox.addWidget(self.btnCancel)

        self.labelLeftVBox.setAlignment(Qt.AlignRight)
        self.classHeaderHBox.addLayout(self.labelLeftVBox)
        self.classHeaderHBox.addLayout(self.comboRightVBox)

        self.classMainVBox.addLayout(self.classHeaderHBox)
        self.classMainVBox.addLayout(self.classContentHBox)

        self.setLayout(self.classMainVBox)
        #self.show()
        dw=self.viewer.window.add_dock_widget(self,name='Colours')
        self.annotatorjObj.coloursWidget=dw
        if self.annotatorjObj.firstDockWidget is None:
            self.annotatorjObj.firstDockWidget=dw
            self.annotatorjObj.firstDockWidgetName='Colours'
        else:
            try:
                self.viewer.window._qt_window.tabifyDockWidget(self.annotatorjObj.firstDockWidget,dw)
            except Exception as e:
                print(e)
                # RuntimeError: wrapped C/C++ object of type QtViewerDockWidget has been deleted
                # try to reset the firstDockWidget manually
                self.annotatorjObj.findDockWidgets('Colours')
                try:
                    if self.annotatorjObj.firstDockWidget is None:
                        self.annotatorjObj.firstDockWidget=dw
                        self.annotatorjObj.firstDockWidgetName='Colours'
                    else:
                        self.viewer.window._qt_window.tabifyDockWidget(self.annotatorjObj.firstDockWidget,dw)
                except Exception as e:
                    print(e)
                    print('Failed to add widget Colours')


    def addColours(self,comboBox):
        comboBox.addItem('red')
        comboBox.addItem('green')
        comboBox.addItem('blue')
        comboBox.addItem('cyan')
        comboBox.addItem('magenta')
        comboBox.addItem('yellow')
        comboBox.addItem('orange')
        comboBox.addItem('white')
        comboBox.addItem('black')


    def updateAnnotColour(self,idx):
        colour=self.annotColourBox.currentText()
        print(f'Set annotation colour: {colour}')
        #self.updateColour(colour,0)


    def updateOverlayColour(self,idx):
        colour=self.overlayColourBox.currentText()
        print(f'Set overlay colour: {colour}')
        #self.updateColour(colour,1)


    def updateColour(self,colour,toUpdate=2):
        if toUpdate==0:
            # annot colour update
            self.annotatorjObj.defColour=colour
            self.updateROIcolours(0)
        elif toUpdate==1:
            # overlay colour update
            self.annotatorjObj.overlayColour=colour
            self.updateROIcolours(1)
        elif toUpdate==2:
            # update both
            self.annotatorjObj.defColour=self.annotColourBox.currentText()
            self.annotatorjObj.overlayColour=self.overlayColourBox.currentText()
            self.updateROIcolours(0)
            self.updateROIcolours(1)

            # also destroy the widget
            self.closeWidget()


    def updateROIcolours(self,toUpdate):
        if toUpdate==0:
            # annot colour update
            shapeLayer=self.annotatorjObj.findROIlayer()
            if shapeLayer is None:
                print('Cannot update ROI colours, layer not found (ROI)')
                return
            for i in range(len(shapeLayer.data)):
                if shapeLayer.properties['class'][i]==0:
                    # only update unclassified rois' edge colour
                    shapeLayer._data_view.update_edge_color(i,self.annotatorjObj.colourString2Float(self.annotatorjObj.defColour))
            shapeLayer.refresh()

        elif toUpdate==1:
            # overlay colour update
            shapeLayer=self.annotatorjObj.findROIlayer(layerName='overlay')
            if shapeLayer is None:
                print('Cannot update overlay colours, layer not found (overlay)')
                return
            for i in range(len(shapeLayer.data)):
                # all rois' edge colour is updated
                shapeLayer._data_view.update_edge_color(i,self.annotatorjObj.colourString2Float(self.annotatorjObj.overlayColour))
                # also update face colour for overlays
                overlayCol=self.annotatorjObj.colourString2Float(self.annotatorjObj.overlayColour)
                overlayCol[-1]=0.5
                shapeLayer._data_view.update_face_color(i,overlayCol)
            shapeLayer.refresh()


    def closeWidget(self):
        if self.annotatorjObj.ColourSelector is not None:
            try:
                if self.annotatorjObj.firstDockWidgetName=='Colours':
                    self.annotatorjObj.findDockWidgets('Colours')
                self.viewer.window.remove_dock_widget(self.annotatorjObj.ColourSelector)
                self.coloursWidget=None
                self.annotatorjObj.ColourSelector=None
            except Exception as e:
                print(e)


    def closeEvent(self, event):
        event.ignore()
        self.closeWidget()
        #event.accept()


# -------------------------------------
# end of class ColourSelector
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
            if self.annotatorjObj.ExportFrame.isVisible() or 'napari-annotatorj: AnnotatorJExport' in self.viewer.window._dock_widgets.data:
                print('Export widget is visible')
                return
            else:
                print('Export widget is not visible')
                # rebuild the widget


        # supported image formats
        self.imageExsts=['.png','.bmp','.jpg','.jpeg','.tif','.tiff']
        self.defDir=None
        self.defFile=None
        self.started=False
        self.startedOrig=False
        self.startedROI=False
        self.curFileList=[]
        self.annotExsts=['.zip','.tiff','.tif']
        self.curROIList=[]
        self.selectedObjectType='ROI'
        self.finished=True
        self.multiLabel=True
        self.multiLayer=False
        self.semantic=False
        self.coordinates=False
        self.overlay=False
        self.bboxFormat=0; # COCO
        self.exportDone=False
        self.originalFolder=None
        self.annotationFolder=None
        self.annotEdgeWidth=1.0

        # browse buttons
        self.btnBrowseOrig=QPushButton('Browse original ...')
        self.btnBrowseOrig.setToolTip('Browse folder of original images')
        self.btnBrowseOrig.clicked.connect(self.browseOrig)
        self.btnBrowseROI=QPushButton('Browse annot ...')
        self.btnBrowseROI.setToolTip('Browse folder of annotation zip files')
        self.btnBrowseROI.clicked.connect(self.browseROI)

        # text fields
        self.textFieldOrig=QLineEdit()
        self.textFieldOrig.setToolTip('original images folder')
        self.textFieldOrig.editingFinished.connect(self.setOrigFolder)
        self.textFieldROI=QLineEdit()
        self.textFieldROI.setToolTip('annotation zips folder')
        self.textFieldROI.editingFinished.connect(self.setROIFolder)

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
        self.btnExportMasks.clicked.connect(self.startExportProgress)
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
            dw=self.viewer.window.add_dock_widget(self,name='Export')
            if self.annotatorjObj.firstDockWidget is None:
                self.annotatorjObj.firstDockWidget=dw
                self.annotatorjObj.ExportWidget=dw
                self.annotatorjObj.firstDockWidgetName='Export'
            else:
                try:
                    self.viewer.window._qt_window.tabifyDockWidget(self.annotatorjObj.firstDockWidget,dw)
                except Exception as e:
                    print(e)
                    # RuntimeError: wrapped C/C++ object of type QtViewerDockWidget has been deleted
                    # try to reset the firstDockWidget manually
                    self.annotatorjObj.findDockWidgets('Export')
                    try:
                        if self.annotatorjObj.firstDockWidget is None:
                            self.annotatorjObj.firstDockWidget=dw
                            self.annotatorjObj.firstDockWidgetName='Export'
                        else:
                            self.viewer.window._qt_window.tabifyDockWidget(self.annotatorjObj.firstDockWidget,dw)
                    except Exception as e:
                        print(e)
                        print('Failed to add widget Export')


    def browseOrig(self):
        # open folder loading dialog box

        if self.started:
            # check if there are rois added:
            if not self.exportDone:
                # check how far the current export has progressed
                #self.closeActiveWindows(self.curFileIdx,len(self.curFileList))
                pass

        # open folder dialog
        self.originalFolder=QFileDialog.getExistingDirectory(
                self,"Select original image folder",
                self.defDir,QFileDialog.ShowDirsOnly)

        print(self.originalFolder)
        if os.path.isdir(self.originalFolder):
            print('Opened original image folder: {}'.format(self.originalFolder))
            self.textFieldOrig.setText(self.originalFolder)
        else:
            print('Failed to open original image folder')
            return

        self.initializeOrigFolderOpening(self.originalFolder)


    def browseROI(self):
        if self.started:
            # check if there are rois added:
            if not self.exportDone:
                # check how far the current export has progressed
                #self.closeActiveWindows(self.curFileIdx,len(self.curFileList))
                pass

        # open folder dialog
        self.annotationFolder=QFileDialog.getExistingDirectory(
                self,"Select annotation folder",
                self.defDir,QFileDialog.ShowDirsOnly)

        print(self.annotationFolder)
        if os.path.isdir(self.annotationFolder):
            print('Opened annotation folder: {}'.format(self.annotationFolder))
            self.textFieldROI.setText(self.annotationFolder)
        else:
            print('Failed to open annotation folder')
            return

        self.initializeROIFolderOpening(self.annotationFolder)


    def initializeOrigFolderOpening(self,originalFolder):
        self.defDir=originalFolder


        # get a list of files in the current directory
        self.curFileList=[f for f in os.listdir(originalFolder) if os.path.isfile(os.path.join(originalFolder,f)) and os.path.splitext(f)[1] in self.imageExsts]
        fileListCount=len(self.curFileList)

        # check if there are correct files in the selected folder
        if fileListCount<1:
            print('No original image files found in current folder')
            warnings.warn('Could not find original image files in selected folder')
            self.started=False
            return

        print(f'Found {fileListCount} images in current folder')

        self.startedOrig=True
        if self.startedROI:
            self.started=True


    def setOrigFolder(self):
        self.originalFolder=self.textFieldOrig.text()
        if os.path.isdir(self.originalFolder):
            print('Opened original image folder: {}'.format(self.originalFolder))
        else:
            print('Failed to open original image folder')
            return
        self.initializeOrigFolderOpening(self.originalFolder)


    def setROIFolder(self):
        self.annotationFolder=self.textFieldROI.text()
        if os.path.isdir(self.annotationFolder):
            print('Opened annotation folder: {}'.format(self.annotationFolder))
            self.textFieldROI.setText(self.annotationFolder)
        else:
            print('Failed to open annotation folder')
            return
        self.initializeROIFolderOpening(self.annotationFolder)


    def initializeROIFolderOpening(self,annotationFolder):
        # get a list of files in the current directory
        listOfROIs=[f for f in os.listdir(annotationFolder) if os.path.isfile(os.path.join(annotationFolder,f)) and os.path.splitext(f)[1] in self.annotExsts]
        ROIListCount=0
        #String[] curFileList;

        # get number of useful files
        # see which object type is selected
        annotNameReg=None
        annotExt=None
        if self.selectedObjectType=='ROI':
            #
            annotNameReg='_ROIs'
            annotExt='.zip'
        elif self.selectedObjectType=='semantic':
            #
            annotNameReg='_semantic'
            annotExt='.tiff'
        elif self.selectedObjectType=='bbox':
            #
            annotNameReg='_bboxes'
            annotExt='.zip'
        else:
            #
            pass

        self.curROIList=[]
        for i in range(len(listOfROIs)):
            # new, for any type of object we support
            curFileName=listOfROIs[i]
            if os.path.splitext(curFileName)[1]==annotExt and annotNameReg in curFileName:
                self.curROIList.append(curFileName)
                ROIListCount+=1


        # check if there are correct files in the selected folder
        if ROIListCount<1:
            print('No annotation files found in current folder')
            warnings.warn('Could not find annotation files in selected folder')
            self.started=False
            return

        print(f'Found {ROIListCount} annotation files in current folder')

        self.startedROI=True
        if self.startedOrig:
            self.started=True


    def setMultiLabel(self,state):
        if state==Qt.Checked:
            self.multiLabel=True
            print('Multi-label (instances) selected')
        else:
            self.multiLabel=False
            print('Multi-label (instances) cleared')

    def setMultiLayer(self,state):
        if state==Qt.Checked:
            self.multiLayer=True
            print('Multi-layer (stack) selected')
        else:
            self.multiLayer=False
            print('Multi-layer (stack) cleared')

    def setSemantic(self,state):
        if state==Qt.Checked:
            self.semantic=True
            print('Semantic (binary) selected')
        else:
            self.semantic=False
            print('Semantic (binary) cleared')

    def setCoordinates(self,state):
        if state==Qt.Checked:
            self.coordinates=True
            print('Coordinates selected')
        else:
            self.coordinates=False
            print('Coordinates cleared')

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


    def setOverlay(self,state):
        if state==Qt.Checked:
            self.overlay=True
            print('Overlay selected')
        else:
            self.overlay=False
            print('Overlay cleared')


    def setAnnotRoi(self,state):
        if state==Qt.Checked:
            # ROI object type selected, default
            # set var for object
            self.selectedObjectType='ROI'
            print(f'Selected annotation object type: {self.selectedObjectType}')

            # enable all checkboxes
            self.chckbxMultiLabel.setEnabled(True)
            self.chckbxMultiLayer.setEnabled(True)
            self.chckbxSemantic.setEnabled(True)
            self.chckbxCoordinates.setEnabled(True)
            self.chckbxOverlay.setEnabled(True)

            self.initializeOrigFolderOpening(self.textFieldOrig.text())
            self.initializeROIFolderOpening(self.textFieldROI.text())
        else:
            # handle in next button's selected option
            pass

    def setAnnotSemantic(self,state):
        if state==Qt.Checked:
            # semantic selected, set everything to semantic instead
            # set var for object
            self.selectedObjectType="semantic"
            print(f'Selected annotation object type: {self.selectedObjectType}')

            # enable all checkboxes
            self.chckbxMultiLabel.setEnabled(True)
            self.chckbxMultiLayer.setEnabled(True)
            self.chckbxSemantic.setEnabled(True)
            self.chckbxCoordinates.setEnabled(True)
            self.chckbxOverlay.setEnabled(True)

            self.initializeOrigFolderOpening(self.textFieldOrig.text())
            self.initializeROIFolderOpening(self.textFieldROI.text())

            warnings.warn('Semantic annotation type selected.\nExported images (instance, stack) might\ncontain multiple touching objects as one!')
        else:
            # handle in next button's selected option
            pass

    def setAnnotBbox(self):
        if state==Qt.Checked:
            # bbox selected, set everything to bbox instead
            # set var for object
            self.selectedObjectType="bbox"
            print(f'Selected annotation object type: {self.selectedObjectType}')

            # enable only coordinates checkbox
            self.chckbxMultiLabel.setEnabled(False)
            self.chckbxMultiLayer.setEnabled(False)
            self.chckbxSemantic.setEnabled(False)
            self.chckbxCoordinates.setEnabled(True)
            self.chckbxOverlay.setEnabled(True)
            # also reset the others to False
            self.chckbxMultiLabel.setChecked(False)
            self.chckbxMultiLayer.setChecked(False)
            self.chckbxSemantic.setChecked(False)

            self.initializeOrigFolderOpening(self.textFieldOrig.text())
            self.initializeROIFolderOpening(self.textFieldROI.text())
        else:
            # handle in next button's selected option
            pass


    def startExportProgress(self):
        if not self.started:
            print('Open an image and annotation folder first')
            warnings.warn('Click Browse buttons to initialize folders')
            return


        # check that at least one export option is selected:
        if not self.multiLabel and not self.multiLayer and not self.semantic and not self.coordinates and not self.overlay:
            print('Select export option')
            print('No export option is selected')
            warnings.warn('Select at least one export option')
            return

        self.finished=False

        # this doesnt work yet:
        self.openExportProgressFrame()

        self.viewer.window._status_bar._toggle_activity_dock(True)

        # check if the folders have the same number of files
        # check if every annotation file has a corresponding original image file
        origFileCount=len(self.curFileList)
        annotFileCount=len(self.curROIList)
        self.skipFileList=[]
        curAnnotType=None
        if origFileCount!=annotFileCount:
            print('Different number of files in folders')
        else:
            print('Same number of files in folders')

        # set progressbar length
        from napari.utils import progress
        #from time import sleep
        #self.progressBar.setMaximum(annotFileCount)

        # for annot time saving
        # TODO later

        #debug:
        print(f'orig folder: {self.originalFolder}')
        print(f'annot folder: {self.annotationFolder}')
        print(f'origs: {self.curFileList}')
        print(f'annots: {self.curROIList}')

        print('---- starting export ----')
        # check for annot file correspondance
        with progress(range(len(self.curROIList))) as progressBar:
            for i in progressBar:
                self.curAnnotFileName=self.curROIList[i]
                progressText=f'({i+1}/{annotFileCount}): {self.curAnnotFileName}'
                print(progressText)
                progressBar.set_description(progressText)
                #debug:
                print(progressText)

                # check if this file is in the skip list by being a multiple-annotated file
                if self.curAnnotFileName in self.skipFileList:
                    # should check if the skipped annot file matches another original file name better !!!!!!!!
                    # TODO

                    # it is, skip it
                    print(f'skipping annotation file {self.curAnnotFileName}')
                    progressBar.update(1)
                    #continue
                    # without for loop, put the other steps in else below

                else:
                    curAnnotFileRaw=None
                    # find annotation type by name:
                    if '_ROIs' in self.curAnnotFileName:
                        # roi file
                        curAnnotFileRaw=self.curAnnotFileName[0:self.curAnnotFileName.rfind('_ROIs')]
                        curAnnotType='ROI'
                    elif '_bboxes' in self.curAnnotFileName:
                        # bbox file
                        curAnnotFileRaw=self.curAnnotFileName[0:self.curAnnotFileName.rfind('_bboxes')]
                        curAnnotType='bbox'
                    elif '_semantic' in self.curAnnotFileName:
                        # semantic file
                        curAnnotFileRaw=self.curAnnotFileName[0:self.curAnnotFileName.rfind('_semantic')]
                        curAnnotType='binary'
                    else:
                        print(f'could not determine type of annotation file: {self.curAnnotFileName}')
                        # use default ROI in this case
                        print('using default annotation type: ROI')
                        curAnnotType='ROI'
                        curAnnotFileRaw=self.curAnnotFileName[0:self.curAnnotFileName.rfind('.')]


                    # check if multiple annotation files exist for this image/annot file:
                    multipleAnnots=False
                    self.multipleList=[]
                    for e in range(annotFileCount):
                        if e==i:
                            # current annot file
                            continue
                        else:
                            tmpAnnotFileName=self.curROIList[e]
                            tmpIdx=tmpAnnotFileName.find(curAnnotFileRaw)
                            if tmpIdx==-1:
                                # not found, continue
                                pass
                            else:
                                # another annot file for this image found!
                                # store this name or idx
                                # TODO
                                self.multipleList.append(e)
                                multipleAnnots=True

                    if multipleAnnots:
                        # show dialog to choose which annotation file they want for the image
                        # ask annotation type in dialog box
                        self.multiNum=len(self.multipleList)
                        self.annotNames=[]
                        self.annotNames=[self.curAnnotFileName]
                        for e in range(self.multiNum):
                            self.annotNames.append(self.curROIList[self.multipleList[e]])

                        self.annotNameChooserDialog=QDialog()
                        self.annotNameChooserDialog.setModal(True)
                        self.annotNameChooserDialog.setWindowTitle('Multiple instances found')

                        annotNamesLabel=QLabel(f'Select which annotation file to use for {curAnnotFileRaw} :')
                        self.annotNamesBox=QComboBox()
                        for el in self.annotNames:
                            self.annotNamesBox.addItem(el)
                        self.annotNamesBox.setCurrentIndex(0)

                        annotNamesOK=QPushButton('Ok')
                        annotNamesOK.clicked.connect(self.okMultiSelection)
                        annotNamesCancel=QPushButton('Cancel')
                        annotNamesCancel.clicked.connect(self.cancelMultiSelection)

                        boxLayout=QHBoxLayout()
                        boxLayout.addWidget(annotNamesLabel)
                        boxLayout.addWidget(self.annotNamesBox)
                        boxLayout.addWidget(annotNamesOK)
                        boxLayout.addWidget(annotNamesCancel)
                        self.annotNameChooserDialog.setLayout(boxLayout)
                        #self.annotNameChooserDialog.show()
                        self.annotNameChooserDialog.exec()


                    # find if its original image exists
                    #Arrays.asList(curFileList).contains(curAnnotFileRaw);
                    foundIt=False
                    curOrigFileName=None
                    for j in range(origFileCount):
                        curOrigFileName=self.curFileList[j]
                        curOrigFileNameRaw=curOrigFileName[0:curOrigFileName.rfind('.')]
                        if curAnnotFileRaw==curOrigFileNameRaw:
                            # found it
                            foundIt=True
                            break
                        else:
                            # continue searching
                            pass

                    if foundIt:
                        # check annotation type:
                        # so far only instance segmentation is supported
                        if not curAnnotType=='ROI':
                            # implemented now
                            pass


                        # ---------------------
                        # call export function:
                        # ---------------------
                        self.startExport(self.curAnnotFileName,curAnnotFileRaw,curOrigFileName,annotFileCount,i,curAnnotType,progressBar)

                    else:
                        # no original image for it --> skip or throw error?
                        # cannot generate mask for sure as we need the image dims to create a mask of the same size!
                        print(f'No original image found for annotation file "{self.curAnnotFileName}" --> skipping it')
                        print('---------------------')
                        continue

        print('---- finished export ----')
        
        #self.progressBar.setValue(annotFileCount)
        
        # finished every image in the folder
        if hasattr(self,'btnOk') and hasattr(self,'btnCancelProgress') and hasattr(self,'lblExportingImages'):
            self.btnOk.setEnabled(True)
            self.btnCancelProgress.setEnabled(False)
            self.lblExportingImages.setText('Finished exporting images')
        
        # save annot time in file
        # TODO later

        # make the progress bar activity panel invisible again
        self.viewer.window._status_bar._toggle_activity_dock(False)

        # when open functions finish:
        self.started=True

        # check export options
        # call export function

        self.finished=True
        self.exportDone=True

        # pop message: export finished
        self.showExportFinished()


    def cancelMultiSelection(self):
        # selection was cancelled --> abort
        self.finished=True
        self.exportDone=True
        if hasattr(self,'btnOk') and hasattr(self,'btnCancelProgress') and hasattr(self,'lblExportingImages'):
            self.lblExportingImages.setText('Exporting was cancelled')
            self.btnOk.setEnabled(True)
            self.btnCancelProgress.setEnabled(False)
        self.annotNameChooserDialog.done(QDialog.Rejected)
        return

    def okMultiSelection(self):
        choiceIdx=self.annotNamesBox.currentIndex()

        # add all other options to skipFileList
        for c in range(self.multiNum+1):
            if c==0:
                # first item was the original name
                self.skipFileList.append(self.curAnnotFileName)
            else:
                # any other item from the list
                self.skipFileList.append(self.curROIList[self.multipleList[c-1]])

        # can set selected file name now:
        self.curAnnotFileName=self.annotNames[choiceIdx]
        print(f'Selected annotation file: {self.curAnnotFileName}')
        self.annotNameChooserDialog.done(QDialog.Accepted)


    def startExport(self,curAnnotFileName,curAnnotFileRaw,curOrigFileName,annotFileCount,i,curAnnotType,progressBar):
        # start a progress bar for logging

        # ------------ for logging progress: -----------------
        # (i+1/annotFileCount) <-- to log

        # update file name tag on main window to check which image we are annotating
        displayedName=curAnnotFileRaw
        maxLength=20
        # check how long the file name is (if it can be displayed)
        nameLength=len(curAnnotFileRaw)
        if nameLength>maxLength:
            displayedName=f'{curAnnotFileRaw[0:maxLength-3]}...tiff'

        # display this in the progress bar too:
        # not yet
        # set the labels and progress
        if hasattr(self,'lblExportingImages'):
            self.lblExportingImages.setText("Exporting images...")
        if hasattr(self,'lblCurrentImage'):
            self.lblCurrentImage.setText(f' ({i}/{annotFileCount}): {displayedName}')

        # ------------- logging end -------------------------
        
        

        # export
        # ----------------------------------------------------------------
        # instance segmentation

        # create masks folder in current annotation folder
        # create output folder with the class name
        

        dimensions=[]
        width=0
        height=0

        shapesLayer=None

        # this annotation file has a corresponding image
        # read original image to get the dimensions of it:
        origImage=skimage.io.imread(os.path.join(self.originalFolder,curOrigFileName))
        if origImage is not None:
            # read it successfully
            # if it is opened by default, close the window after getting the size info from it
            # dimensions is an array of (width, height, nChannels, nSlices, nFrames)
            dimensions=origImage.shape
            width=dimensions[0]
            height=dimensions[1]
        else:
            print(f'Could not open original image: {curOrigFileName}')
            warnings.warn('Could not open original image: {curOrigFileName}')
            # allow to continue the processing?
            # if not, close progressbar too

            return


        # check annot type first
        if curAnnotType=='ROI' or curAnnotType=='bbox':
            # only need to load the .zip file
            try:
                #debug:
                print(os.path.join(self.annotationFolder,curAnnotFileName))
                rois=ImagejRoi.fromfile(os.path.join(self.annotationFolder,curAnnotFileName))
                shapesLayer=self.extractROIdataSimple(rois)
            except Exception as e:
                print(f'Failed to open ROI: {curAnnotFileName}');
                warnings.warn('Failed to open ROI .zip file')
                print(e)
                return
            print(f'Opened ROI: {curAnnotFileName}')

        elif curAnnotType=='binary':
            # need to convert the semantic binary image to rois
            semdimensions
            semwidth=0
            semheight=0
            semanticImage=skimage.io.imread(os.path.join(self.annotationFolder,curAnnotFileName))
            if semanticImage is not None:
                # read it successfully
                # if it is opened by default, close the window after getting the size info from it
                # dimensions is an array of (width, height, nChannels, nSlices, nFrames)
                semdimensions=semanticImage.shape
                semwidth=semdimensions[0]
                semheight=semdimensions[1]
                if semwidth!=width or semheight!=height:
                    print(f'Inconsistent size of semantic annotation image: {curAnnotFileName}, skipping it')
                    warnings.warn(f'Could not verify annotation image: {curAnnotFileName}')
                    return

                # create rois from it
                masked=semanticImage>0
                semanticRoi,hierarchy=cv2.findContours(masked.astype(numpy.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if semanticRoi is not None:
                    pass
                else:
                    print('semanticRoi is None')

                roiProps={'name':['0001'],'class':[0],'nameInt':[1]}
                roiTextProps={
                    'text': '{nameInt}: ({class})',
                    'anchor': 'center',
                    'size': 10,
                    'color': 'black',
                    'visible':False
                }
                # add an empty shapes layer
                shapesLayer=Shapes(data=numpy.array([[0,0],[1,1]]),shape_type='polygon',name='ROI',properties=roiProps,text=roiTextProps)
                shapesLayer._data_view.remove(0)
                if len(semanticROI)>1 and type(semanticROI) is tuple:
                    # multiple rois in the roi --> split them!
                    for idx,x in enumerate(semanticROI):
                        if type(x) is tuple:
                            x=x[0]
                            shapesLayer.add_polygon(numpy.array(numpy.fliplr(numpy.squeeze(x))))
                            shapesLayer.properties['class'][-1]=0
                else:
                    shapesLayer.add_polygon(semanticROI)
                    shapesLayer.properties['class'][-1]=0

            else:
                print(f'Could not open original image: {curOrigFileName}')
                warnings.warn(f'Could not open original image: {curOrigFileName}')
                # allow to continue the processing?
                # if not, close progressbar too

                return

        # check if there are annotated objects in the file:
        roiCount=len(shapesLayer.data) if shapesLayer is not None else 0
        print(f'annotated objects: {roiCount}')
        if roiCount<1:
            # this continue worked while this whole bunch of export code was in the EXPORT ------- for cycle
            #continue;
            return

        # create mask image
        # export a mask image from the rois
        labels=shapesLayer.to_labels([width, height])
        #labelLayer = self.viewer.add_labels(labels, name='labelled_mask')
        #labelLayer.visible = False


        # fill output mask image according to export type selected:
        outputFileName=None
        annotationFolder2=None
        exportFolder=None
        refreshMask=False

        # collect class info from the ROIs if any
        nonEmptyClassNameNumbers=[]
        for r in range(roiCount):
            # get r. instance
            curROIgroup=shapesLayer.properties['class'][r]
            if curROIgroup>0 and curROIgroup not in nonEmptyClassNameNumbers:
                nonEmptyClassNameNumbers.append(curROIgroup)
                #debug:
                print(f'Nonempty class idx+= {curROIgroup}')
        
        exportFolderClass=None
        outDir=None


        # check export type selected
        if self.multiLabel:
            # labelled masks on single layer of mask

            exportFolder='labelled_masks'

            # labels is already in increasing label format

            # create export folder:
            annotationFolder2=self.createExportFolder(exportFolder)
            # construct output file name:
            outputFileName=os.path.join(annotationFolder2,curAnnotFileRaw+'.tiff')
            # save output image:
            toSave=deepcopy(labels)
            toSave.astype('uint16')
            self.saveExportedImage(toSave,outputFileName)

            # also save class mask images if the ROIs have class info saved
            for c in range(len(nonEmptyClassNameNumbers)):

                curClassNum=nonEmptyClassNameNumbers[c]

                # create export folder:
                exportFolderClass=os.path.join(annotationFolder2,'Class_{:02d}'.format(curClassNum))

                os.makedirs(exportFolderClass,exist_ok=True)
                print(f'Created class output folder: {exportFolderClass}')

                maskImage2=self.createClassMask(roiCount,curClassNum,shapesLayer,True,labels)
                if maskImage2 is None:
                    return

                # construct output file name:
                outputFileName=os.path.join(exportFolderClass,curAnnotFileRaw+'.tiff')

                self.saveExportedImage(maskImage2,outputFileName)

            refreshMask=True


        if self.multiLayer:
            # multi-layer stack mask

            exportFolder='layered_masks'
            ok=False

            if refreshMask:
                # create new empty mask
                maskImage=deepcopy(labels)

            # create stack image:
            if roiCount>1:

                stack=numpy.zeros((width,height,roiCount),dtype=bool)
                try:
                    color1=1
                    vals=numpy.unique(maskImage)
                    vals=numpy.delete(vals,numpy.where(vals==0))
                    # start getting the objects from the annotation file
                    for s in range(roiCount):
                        # could already set the slices here instead of creating empty layers
                        tmpSlice=numpy.zeros((width,height),dtype=bool)

                        # get s. instance
                        # set fill value
                        tmpSlice[numpy.where(maskImage==vals[s])]=color1

                        stack[:,:,s]=tmpSlice
                    ok=True

                except MemoryError as e:
                    print('Out-of-memory error')
                    #stack.trim()
                    ok=False

                if stack.shape[2]>1:
                    maskImage=stack
                

                if not ok:
                    maskImage=None
                    print('Error creating stack image')
                    return

            # create export folder:
            annotationFolder2=self.createExportFolder(exportFolder)
            # construct output file name:
            outputFileName=os.path.join(annotationFolder2,curAnnotFileRaw+'.tiff')
            # save output image:
            self.saveExportedImage(maskImage,outputFileName)

            # also save class mask images if the ROIs have class info saved
            for c in range(len(nonEmptyClassNameNumbers)):

                curClassNum=nonEmptyClassNameNumbers[c]

                # create export folder:
                exportFolderClass=os.path.join(annotationFolder2,'Class_{:02d}'.format(curClassNum))

                os.makedirs(exportFolderClass,exist_ok=True)
                print(f'Created class output folder: {exportFolderClass}')

                maskImage2=self.createClassMaskStack(width,height,roiCount,curClassNum,shapesLayer,labels)
                if maskImage2 is None:
                    return

                # construct output file name:
                outputFileName=os.path.join(exportFolderClass,curAnnotFileRaw+'.tiff')

                self.saveExportedImage(maskImage2, outputFileName)

            refreshMask=True


        if self.semantic:
            # binary semantic segmentation image

            exportFolder='binary_masks'

            if refreshMask:
                # create new empty mask
                maskImage=deepcopy(labels)

            maskImage=maskImage.astype(bool)
            #maskImage[numpy.where(maskImage>0)]=1

            # create export folder:
            annotationFolder2=self.createExportFolder(exportFolder)
            # construct output file name:
            outputFileName=os.path.join(annotationFolder2,curAnnotFileRaw+'.tiff')
            # save output image:
            self.saveExportedImage(maskImage, outputFileName)

            # also save class mask images if the ROIs have class info saved
            for c in range(len(nonEmptyClassNameNumbers)):

                curClassNum=nonEmptyClassNameNumbers[c]

                # create export folder:
                exportFolderClass=os.path.join(annotationFolder2,'Class_{:02d}'.format(curClassNum))

                os.makedirs(exportFolderClass,exist_ok=True)
                print(f'Created class output folder: {exportFolderClass}')

                maskImage2=self.createClassMask(roiCount,curClassNum,shapesLayer,False,deepcopy(labels))
                if maskImage2 is None:
                    return

                # construct output file name:
                outputFileName=os.path.join(exportFolderClass,curAnnotFileRaw+'.tiff')

                self.saveExportedImage(maskImage2.astype(bool),outputFileName)

            refreshMask=True


        if self.coordinates:
            # bounding box coordinates of objects

            exportFolder='bounding_box_coordinates'

            # prepare a bbox array for export
            bboxList=None

            # check selected bounding box format
            badBbox=False
            if self.bboxFormat==0:
                # COCO format, default
                bboxList=self.fillBboxList(shapesLayer,roiCount)
            elif self.bboxFormat==1:
                # YOLO format
                #TODO
                bboxList=self.fillBboxListYOLO(shapesLayer,roiCount,width,height)
            else:
                # unknown case
                print('Unknown bounding box format selected, please try again.')
                badBbox=True

            # start getting the objects from the annotation file
            # moved to its own fcn fillBboxList();

            if badBbox or bboxList is None:
                print('Failed to create .csv file with bounding box coordinates.')
                # return
            else:
                # can continue
                pass

            # create export folder:
            annotationFolder2=self.createExportFolder(exportFolder)
            # construct output file name:
            outputFileName=os.path.join(annotationFolder2,curAnnotFileRaw+'.csv')
            # save output csv:
            # TODO: create fcn for this
            if (self.bboxFormat==0):
                self.saveExportedCSV(bboxList,outputFileName)
            elif (self.bboxFormat==1):
                self.saveExportedCSVyolo(bboxList,outputFileName)
            #saveExportedImage(maskImage, outputFileName);

            # also save class mask images if the ROIs have class info saved
            for c in range(len(nonEmptyClassNameNumbers)):

                curClassNum=nonEmptyClassNameNumbers[c]

                # create export folder:
                exportFolderClass=os.path.join(annotationFolder2,'Class_{:02d}'.format(curClassNum))

                os.makedirs(exportFolderClass,exist_ok=True)
                print(f'Created class output folder: {exportFolderClass}')

                bboxList2=self.createClassCSV(roiCount,curClassNum,shapesLayer,width,height,self.bboxFormat)
                if bboxList2 is None:
                    return

                # construct output file name:
                outputFileName=os.path.join(exportFolderClass,curAnnotFileRaw+'.csv')

                if (self.bboxFormat==0):
                    self.saveExportedCSV(bboxList2,outputFileName)
                elif (self.bboxFormat==1):
                    self.saveExportedCSVyolo(bboxList2,outputFileName)

            refreshMask=True


        if self.overlay:
            # outlines overlayed on original image

            exportFolder='outlined_images'

            # create export folder:
            annotationFolder2=self.createExportFolder(exportFolder)
            # construct output file name:
            outputFileName=os.path.join(annotationFolder2,curAnnotFileRaw+'.tiff')
            # save output image:
            self.saveOutlinedImage(origImage,shapesLayer,roiCount,outputFileName)

        
        # measure time
        # TODO: later


        # finished this image
        # not yet:
        #progressBar.set_description(progressText) # i+1
        progressBar.update(1)
        if i==annotFileCount:
            # finished every image in the folder
            if hasattr(self,'btnOk') and hasattr(self,'btnCancelProgress') and hasattr(self,'lblExportingImages'):
                self.btnOk.setEnabled(True)
                self.btnCancelProgress.setEnabled(False)
                self.lblExportingImages.setText('Finished exporting images')


    def createExportFolder(self,exportFolder):
        annotationFolder2=os.path.join(self.annotationFolder,exportFolder)
        os.makedirs(annotationFolder2,exist_ok=True)
        print(f'Created output folder: {annotationFolder2}')
        return annotationFolder2


    def createClassMask(self,roiCount,curClassNum,shapeLayer,useRealValue,labelArrayIn):
        count=0
        labelArray=deepcopy(labelArrayIn)
        vals=numpy.unique(labelArray)
        # remove the background value
        vals=numpy.delete(vals,numpy.where(vals==0))
        classArray=numpy.zeros_like(labelArray)

        if len(vals)!=roiCount:
            print('Inconsistent number of rois')
            return None
        # start getting the objects from the annotation file
        for r in range(roiCount):
            # get r. instance
            # see if it belongs to the current class (group)
            if shapeLayer.properties['class'][r]==curClassNum:
                # this class
                #debug:
                print(f'using ROI #{r}')

                # set fill value
                fillValue=-1
                if useRealValue:
                    fillValue=shapeLayer.properties['nameInt'][r]
                else:
                    fillValue=1
                classArray[numpy.where(labelArray==vals[r])]=fillValue

        return classArray


    def createClassMaskStack(self,width,height,roiCount,curClassNum,shapeLayer,labelArrayIn):
        count=0
        labelArray=deepcopy(labelArrayIn)

        stack=numpy.zeros((width,height,roiCount),dtype=bool)

        try:
            color1=1
            vals=numpy.unique(labelArrayIn)
            vals=numpy.delete(vals,numpy.where(vals==0))
            # start getting the objects from the annotation file
            for s in range(roiCount):
                # could already set the slices here instead of creating empty layers
                tmpSlice=numpy.zeros((width,height),dtype=bool)
                if tmpSlice is None:
                    print(f'could not create layer {s}')
                    return None

                # get s. instance
                # see if it belongs to the current class (group)
                if shapeLayer.properties['class'][s]==curClassNum:
                    # set fill value
                    tmpSlice[numpy.where(labelArrayIn==vals[s])]=color1

                stack[:,:,s]=tmpSlice
            ok=True

        except MemoryError as e:
            print('Out-of-memory error')
            #stack.trim()
            ok=False

        if not ok:
            stack=None
            print('Error creating stack image')
            return None

        return stack


    def saveExportedImage(self,img,path):
        if len(img.shape)==2:
            if img.dtype!=bool:
                skimage.io.imsave(path,img,check_contrast=False)
            else:
                import tifffile
                tifffile.imwrite(path,img,photometric='minisblack')
        elif len(img.shape)>2:
            # stack image
            import tifffile
            # img is like XxYxN where N: number of objects, X,Y: image size in xy
            # tifffile needs it like NxYxX
            img2=numpy.swapaxes(img,0,-1)
            img3=numpy.swapaxes(img2,-2,-1)
            tifffile.imwrite(path,img3,photometric='minisblack') #numpy.swapaxes(img,0,-1)
        print('Saved exported image: {}'.format(path))
        print('---------------------')


    # COCO format: [x,y,w,h] --> top left (x,y) + width, height
    def fillBboxList(self,shapeLayer,roiCount):
        if shapeLayer is None or roiCount<1 or len(shapeLayer.data)!=roiCount:
            print('Cannot find bounding boxes for export')
            return None

        bboxList=[]

        # start getting the objects from the annotation file
        for r in range(roiCount):
            # get r. instance
            bbox=shapeLayer.interaction_box(r)
            # bbox is a 10x2 ndarray of bbox coords from upper-left corner
            # [upper-left y,x], [middle-left y,x], [lower-left y,x], [lower middle y,x]
            # [lower-right y,x], [middle-right y,x], [upper-right y,x], [upper middle y,x]
            # [centre y,x], [rotation handle y,x]
            # 0.: [y,x], 2. [height], 6. width
            # get coordinates
            x=round(bbox[0,1])
            y=round(bbox[0,0])
            w=round(bbox[6,1]-bbox[0,1])
            h=round(bbox[2,0]-bbox[0,0])
            bboxList.append([x,y,w,h])
        return bboxList


    # YOLO format: [class,x,y,w,h] normalized to [0,1] --> center (x,y) + width, height
    def fillBboxListYOLO(self,shapeLayer,roiCount,width,height):
        if shapeLayer is None or roiCount<1 or len(shapeLayer.data)!=roiCount:
            print('Cannot find bounding boxes for export')
            return None

        bboxList=[]

        # start getting the objects from the annotation file
        for r in range(roiCount):
            c=shapeLayer.properties['class'][r]
            # get r. instance
            bbox=shapeLayer.interaction_box(r)
            # bbox is a 10x2 ndarray of bbox coords from upper-left corner
            # [upper-left y,x], [middle-left y,x], [lower-left y,x], [lower middle y,x]
            # [lower-right y,x], [middle-right y,x], [upper-right y,x], [upper middle y,x]
            # [centre y,x], [rotation handle y,x]
            # 0.: [y,x], 2. [height], 6. width
            # get coordinates normalized to width and height, from the center point
            y,x=bbox[8,:]
            x=x/width
            y=y/height
            w=(bbox[6,1]-bbox[0,1])/width
            h=(bbox[2,0]-bbox[0,0])/height
            bboxList.append([c,x,y,w,h])
        return bboxList


    def saveExportedCSV(self,bboxes,outPath):
        import csv
        with open(outPath,'w',newline='') as csvFile:
            csvWriter = csv.writer(csvFile, delimiter=',')
            # write header
            csvWriter.writerow(['x','y','width','height'])
            for row in bboxes:
                csvWriter.writerow(row)


    def saveExportedCSVyolo(self,bboxes,outPath):
        # YOLO data format: [class x_center y_center w h] normalized, white space delimited
        import csv
        if os.path.splitext(outPath)[1]==('.csv'):
            outPath=outPath[0:-4]+'.txt'

        with open(outPath,'w',newline='') as csvFile:
            csvWriter = csv.writer(csvFile, delimiter=' ')
            # write header
            csvWriter.writerow(['class','x','y','width','height'])
            for row in bboxes:
                csvWriter.writerow(row)


    def createClassCSV(self,roiCount,curClassNum,shapeLayer,imWidth,imHeight,bboxFormat):
        # count the number of objects in the current class first --> create the bboxList of this size
        thisClassCount=0
        lineNum=0
        # prepare a bbox array for export
        bboxList=[]
        for r in range(roiCount):
            c=shapeLayer.properties['class'][r]
            # see if it belongs to the current class (group)
            if c==curClassNum:
                thisClassCount+=1
                # get coordinates
                bbox=shapeLayer.interaction_box(r)
                if bboxFormat==0:
                    # COCO format, default
                    x=round(bbox[0,1])
                    y=round(bbox[0,0])
                    w=round(bbox[6,1]-bbox[0,1])
                    h=round(bbox[2,0]-bbox[0,0])
                    bboxList.append([x,y,w,h])
                elif bboxFormat==1:
                    # YOLO format
                    y,x=bbox[8,:]
                    x=x/imWidth
                    y=y/imHeight
                    w=(bbox[6,1]-bbox[0,1])/imWidth
                    h=(bbox[2,0]-bbox[0,0])/imHeight
                    bboxList.append([c,x,y,w,h])

                else:
                    # unknown case
                    print('Unknown bounding box format selected, please try again.')

        if thisClassCount==0:
            print(f'Class {curClassNum} contains 0 objects')
            return None

        return bboxList


    # save current annotations as outlines on the original image
    def saveOutlinedImage(self,image,shapeLayer,roiCount,outputFileName):
        outImage=numpy.zeros((image.shape[0],image.shape[1],3),dtype=numpy.uint8)
        rgb=False
        if len(image.shape)==2:
            # 2D image
            for ch in range(3):
                outImage[:,:,ch]=image
        elif len(image.shape)==3:
            # 2D RGB image
            rgb=True

        # LUT for class colours
        classColours=[[1,0,0,1],[0,1,0,1],[0,0,1,1],[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,0.65,0,1],[1,1,1,1],[0,0,0,1]]
        # add some more colours
        classColours2=deepcopy(classColours)
        for i in range(len(classColours)):
            classColours2[i]=[0.5*x for x in classColours[i]]
        classColours=classColours+classColours2

        # single mask image
        #overlay=shapeLayer.to_labels(labels_shape=(image.shape[0],image.shape[1]))
        #maskOutline=overlay-skimage.morphology.erosion(overlay,skimage.morphology.disk(1))

        # multiple mask images by objects
        overlays=shapeLayer.to_masks(mask_shape=(image.shape[0],image.shape[1]))
        overlayed=deepcopy(image)

        for r in range(roiCount):
            c=shapeLayer.properties['class'][r]
            curColour=classColours[c]
            curColour=curColour[0:3] # trim alpha
            maskOutlinex=numpy.bitwise_xor(overlays[r],skimage.morphology.erosion(overlays[r],skimage.morphology.disk(1)))
            for ch in range(3):
                overlayed[:, :, ch] = numpy.where(maskOutlinex[:, :] != 0, 255*curColour[ch], overlayed[:, :, ch])
        skimage.io.imsave(outputFileName, overlayed)


    def showExportFinished(self):
        # ask for confirmation
        response=QMessageBox.information(self, 'Export finished', 'Finished exporting annotations',QMessageBox.Ok, QMessageBox.Ok)
        if response==QMessageBox.Ok:
            # just quit
            print('Ok button clicked')

        elif response==QMessageBox.Close:
            # do nothing
            print("Closed close confirm")


    def cancelExport(self):
        # TODO
        return


    def openExportProgressFrame(self):
        # TODO
        return


    def closeActiveWindows(self,curFileIdx,curFileListLength):
        # TODO
        return


    # mock a similar function to AnnotatorJ class' extractROIdata fcn
    def extractROIdataSimple(self,rois):
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
                curColour='white'
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
        shapesLayer = Shapes(data=roiList,shape_type=roiType,name='ROI',edge_width=self.annotEdgeWidth,edge_color=roiColours,face_color=[0,0,0,0],properties=roiProps,text=roiTextProps)

        return shapesLayer

# -------------------------------------
# end of class ExportFrame
# -------------------------------------


class OptionsFrame(QWidget):
    def __init__(self,napari_viewer,annotatorjObj):
        super().__init__()
        self.viewer = napari_viewer
        self.annotatorjObj=annotatorjObj

        # check if there is opened instance of this frame
        # the main plugin is: 'napari-annotatorj: Annotator J'
        if self.annotatorjObj.optionsFrame is not None:
            # already inited once, load again
            print('detected that Options widget has already been initialized')
            if self.annotatorjObj.optionsFrame.isVisible() or 'Options' in self.viewer.window._dock_widgets.data:
                print('Options widget is visible')
                return
            else:
                print('Options widget is not visible')
                # rebuild the widget


        self.titleLabel=QLabel('Configurations are saved and loaded on plugin startup. When applying changes requires the restart of napari, a message will be shown.')
        self.titleLabel.setWordWrap(True)

        self.annotTypeLabel=QLabel('annotation type: ')
        self.annotTypeLabel.setToolTip('Select annotation type: instance for freehand drawing, bounding box ("bbox") for rectangles, semantic for painting with a brush (labels)')
        self.annotTypeBox=QComboBox()
        self.annotTypeBox.setToolTip('Select annotation type: instance for freehand drawing, bounding box ("bbox") for rectangles, semantic for painting with a brush (labels)')
        validTypes=['instance','bbox','semantic']
        for e in validTypes:
            self.annotTypeBox.addItem(e)
        #self.annotTypeBox.setCurrentIndex(0)
        if self.annotatorjObj.selectedAnnotationType in validTypes:
            # good to go
            pass
        else:
            self.annotatorjObj.selectedAnnotationType='instance'
            print('Setting annotation type to instance (default)')
        self.annotTypeBox.setCurrentText(self.annotatorjObj.selectedAnnotationType)
        self.annotTypeBox.currentIndexChanged.connect(self.annotTypeChanged)

        self.optionsOkBtn=QPushButton('Ok')
        self.optionsOkBtn.setToolTip('Apply changes')
        self.optionsOkBtn.clicked.connect(self.applyOptions)
        #self.optionsOkBtn.setStyleSheet(f"max-width: {int(self.annotatorjObj.bsize2)}px")
        self.optionsCancelBtn=QPushButton('Cancel')
        self.optionsCancelBtn.setToolTip('Revert changes')
        self.optionsCancelBtn.clicked.connect(self.cancelOptions)
        #self.optionsCancelBtn.setStyleSheet(f"max-width: {int(self.annotatorjObj.bsize2)}px")

        self.optionsMainVbox=QVBoxLayout()
        self.typeHBox=QHBoxLayout()
        self.btnHBox=QHBoxLayout()
        
        self.typeHBox.addWidget(self.annotTypeLabel)
        self.typeHBox.addWidget(self.annotTypeBox)
        self.typeHBox.setAlignment(Qt.AlignLeft)

        self.btnHBox.addWidget(self.optionsOkBtn)
        self.btnHBox.addWidget(self.optionsCancelBtn)
        #self.btnHBox.setAlignment(Qt.AlignRight)

        self.optionsMainVbox.addWidget(self.titleLabel)
        self.optionsMainVbox.addLayout(self.typeHBox)
        self.optionsMainVbox.addLayout(self.btnHBox)

        self.setLayout(self.optionsMainVbox)
        #self.show()
        dw=self.viewer.window.add_dock_widget(self,name='Options')
        self.annotatorjObj.optionsWidget=dw
        #dwOrig=self.annotatorjObj.findDockWidgets('Options')
        if self.annotatorjObj.firstDockWidget is None:
            self.annotatorjObj.firstDockWidget=dw
            self.annotatorjObj.firstDockWidgetName='Options'
        else:
            try:
                self.viewer.window._qt_window.tabifyDockWidget(self.annotatorjObj.firstDockWidget,dw)
            except Exception as e:
                print(e)
                # RuntimeError: wrapped C/C++ object of type QtViewerDockWidget has been deleted
                # try to reset the firstDockWidget manually
                self.annotatorjObj.findDockWidgets('Options')
                try:
                    if self.annotatorjObj.firstDockWidget is None:
                        self.annotatorjObj.firstDockWidget=dw
                        self.annotatorjObj.firstDockWidgetName='Options'
                    else:
                        self.viewer.window._qt_window.tabifyDockWidget(self.annotatorjObj.firstDockWidget,dw)
                except Exception as e:
                    print(e)
                    print('Failed to add widget Options')


    def closeEvent(self,event):
        event.ignore()
        print('in the closeEvent event')
        self.closeWidget()
        #event.accept()


    def annotTypeChanged(self,idx):
        #self.annotatorjObj.selectedAnnotationType=self.annotTypeBox.currentText()
        #print(f'Set annotation type: {self.annotatorjObj.selectedAnnotationType}')
        print(f'Selected annotation type: {self.annotTypeBox.currentText()} (not set yet, please apply changes with "Ok" or revert with "Cancel")')


    def applyOptions(self):
        # apply all settings
        self.setAnnotType()
        # write all settings to file
        self.annotatorjObj.writeParams2File()
        self.closeWidget()


    def setAnnotType(self):
        newAnnotType=self.annotTypeBox.currentText()
        newType=False
        if self.annotatorjObj.selectedAnnotationType==newAnnotType:
            # nothing changed
            pass
        else:
            # new annot type selected, update the ROI layer and settings
            newType=True
            if self.annotatorjObj.selectedAnnotationType=='instance' or self.annotatorjObj.selectedAnnotationType=='bbox':
                # shapes layer
                roiLayer=self.annotatorjObj.findROIlayer()
                if roiLayer is not None and len(roiLayer.data)>0:
                    # need to save current annotations first
                    # TODO
                    self.annotatorjObj.saveData()
                    # remove the layer
                    self.viewer.layers.remove(roiLayer)
                else:
                    # can continue
                    if self.annotatorjObj.selectedAnnotationType=='semantic':
                        # remove the layer
                        self.viewer.layers.remove(roiLayer)
                    else:
                        # maybe check type to avoid mismatched annot types in saved files
                        # TODO
                        warnings.warn(f'Set {newAnnotType} but current is {self.annotatorjObj.selectedAnnotationType}')
                        pass
            elif self.annotatorjObj.selectedAnnotationType=='semantic':
                # labels layer
                labelLayer=self.annotatorjObj.findLabelsLayerName(layerName='semantic')
                if labelLayer is not None and len(labelLayer.data)>0:
                    # need to save current annotations first
                    # TODO
                    self.annotatorjObj.saveData()
                    self.viewer.layers.remove(labelLayer)
                    pass
                elif labelLayer is not None:
                    # can continue
                    self.viewer.layers.remove(labelLayer)
                else:
                    # can continue
                    pass

        self.annotatorjObj.selectedAnnotationType=newAnnotType
        print(f'Set annotation type: {self.annotatorjObj.selectedAnnotationType}')

        if newType:
            if newAnnotType=='instance':
                roiLayer=self.annotatorjObj.findROIlayer()
                if roiLayer is None:
                    self.annotatorjObj.initRoiManager()
                    roiLayer=self.annotatorjObj.findROIlayer()
                roiLayer.mode='add_polygon'
                if self.annotatorjObj.freeHandROI not in roiLayer.mouse_drag_callbacks:
                    roiLayer.mouse_drag_callbacks.append(self.annotatorjObj.freeHandROI)
            elif newAnnotType=='bbox':
                roiLayer=self.annotatorjObj.findROIlayer()
                if roiLayer is None:
                    self.annotatorjObj.initRoiManager()
                    roiLayer=self.annotatorjObj.findROIlayer()
                roiLayer.mode='add_rectangle'
                if self.annotatorjObj.freeHandROI in roiLayer.mouse_drag_callbacks:
                    roiLayer.mouse_drag_callbacks.remove(self.annotatorjObj.freeHandROI)
            elif newAnnotType=='semantic':
                labelLayer=self.annotatorjObj.findLabelsLayerName(layerName='semantic')
                if labelLayer is None:
                    imageLayer=self.annotatorjObj.findImageLayer()
                    if imageLayer is None:
                        print('Will initialize semantic annotation layer upon image open function')
                        return
                    else:
                        s=imageLayer.data.shape
                        labelImage=numpy.zeros((s[0],s[1]),dtype='uint8')
                        labelLayer=self.viewer.add_labels(labelImage,name='semantic')
                labelLayer.mode='paint'
                labelLayer.brush_size=self.annotatorjObj.brushSize
                labelLayer.opacity=0.5


    def cancelOptions(self):
        print('Reverting all changes made in the Options widget')
        self.closeWidget()


    def closeWidget(self):
        if self.annotatorjObj.optionsFrame is not None:
            try:
                if self.annotatorjObj.firstDockWidgetName=='Options':
                    self.annotatorjObj.findDockWidgets('Options')
                        
                self.viewer.window.remove_dock_widget(self.annotatorjObj.optionsFrame)
                self.annotatorjObj.optionsWidget=None
                self.annotatorjObj.optionsFrame=None
            except Exception as e:
                print(e)
                try:
                    self.viewer.window.remove_dock_widget('Options')
                    self.annotatorjObj.optionsFrame=None
                    self.annotatorjObj.optionsWidget=None
                except Exception as e:
                    print(e)
                    print('Failed to remove widget named Options')


# -------------------------------------
# end of class OptionsFrame
# -------------------------------------


class DownloadProgressBar(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)  # also sets self.n = b * bsize

# -------------------------------------
# end of class DownloadProgressBar
# -------------------------------------