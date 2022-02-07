"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
from qtpy.QtWidgets import QWidget, QHBoxLayout,QVBoxLayout, QPushButton, QCheckBox,QLabel
from magicgui import magic_factory

import os
import skimage.io
from roifile import ImagejRoi,ROI_TYPE,roiwrite
from napari.layers import Shapes, Image
import numpy
from qtpy.QtWidgets import QFileDialog
from qtpy.QtCore import Qt
#from napari.layers.Shapes import mode
from napari.layers.shapes import _shapes_key_bindings as key_bindings
from napari.layers.shapes import _shapes_mouse_bindings as mouse_bindings
from napari.layers.labels import _labels_mouse_bindings as labels_mouse_bindings
import warnings
from cv2 import cv2
from copy import deepcopy


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

        # ---------------------------
        # add buttons and ui elements
        # ---------------------------
        btnOpen = QPushButton('Open')
        btnOpen.clicked.connect(self.openNew)

        btnLoad = QPushButton('Load')
        btnLoad.clicked.connect(self.loadROIs)

        btnSave = QPushButton('Save')
        btnSave.clicked.connect(self.saveROIs)

        # quick export
        btnExport = QPushButton('[^]')
        btnExport.clicked.connect(self.quickExport)

        # checkboxes
        # edit mode
        chkEdit = QCheckBox('Edit mode')
        chkEdit.setChecked(False)
        chkEdit.setToolTip('Allows switching to contour edit mode. Select with mouse click, accept with "q".')
        chkEdit.stateChanged.connect(self.setEditMode)

        # add auto mode
        chkAuto = QCheckBox('Add automatically')
        chkAuto.setChecked(True)
        chkAuto.setEnabled(False)
        chkAuto.setToolTip('Adds contours to annotations, always active (used in the ImageJ version)')
        chkAuto.setStyleSheet("color: gray")
        # smooth mode
        chkSmooth = QCheckBox('Smooth')
        chkSmooth.setToolTip('Applies smoothing to contour')
        chkSmooth.setChecked(False)
        #chkSmooth.stateChanged.connect(self.setSmooth)
        # show contours
        chkShowContours = QCheckBox('Show contours')
        chkShowContours.setChecked(True)
        chkShowContours.stateChanged.connect(self.showCnt)
        # assist mode
        chkAssist = QCheckBox('Contour assist')
        chkAssist.setChecked(False)
        #chkAssist.stateChanged.connect(self.setContourAssist)
        # show overlay
        chkShowOverlay = QCheckBox('Show overlay')
        chkShowOverlay.setChecked(False)
        #chkShowOverlay.stateChanged.connect(self.showOverlay)
        # class mode
        chkClass = QCheckBox('Class mode')
        chkClass.setChecked(False)
        #chkClass.stateChanged.connect(self.setClassMode)


        # add labels
        roiLabel=QLabel('ROIs')
        nameLabel=QLabel('(1/1) [image name]')

        # set layouts
        mainVbox=QVBoxLayout()
        hBoxTitle=QHBoxLayout()
        hBoxUp=QHBoxLayout()
        hBoxDown=QHBoxLayout()
        vBoxLeft=QVBoxLayout()
        hBoxRight=QHBoxLayout()
        vBoxRightReal=QVBoxLayout()
        vBoxRightDummy=QVBoxLayout()

        hBoxTitle.addWidget(roiLabel)

        vBoxLeft.setAlignment(Qt.AlignTop)
        vBoxLeft.addWidget(chkAuto)
        vBoxLeft.addWidget(chkSmooth)
        vBoxLeft.addWidget(chkShowContours)
        vBoxLeft.addWidget(chkAssist)
        vBoxLeft.addWidget(chkShowOverlay)
        vBoxLeft.addWidget(chkEdit)
        vBoxLeft.addWidget(chkClass)

        # add dummy buttons as spacers
        vBoxRightDummy.setAlignment(Qt.AlignTop)
        vBoxRightDummy.addSpacing(54)
        vBoxRightDummy.addWidget(btnExport)
        
        vBoxRightReal.setAlignment(Qt.AlignTop)
        vBoxRightReal.addWidget(btnOpen)
        vBoxRightReal.addWidget(btnLoad)
        vBoxRightReal.addWidget(btnSave)

        hBoxDown.addWidget(nameLabel)

        hBoxRight.addLayout(vBoxRightDummy)
        hBoxRight.addLayout(vBoxRightReal)

        hBoxUp.addLayout(vBoxLeft)
        hBoxUp.addLayout(hBoxRight)
        mainVbox.addLayout(hBoxTitle)
        mainVbox.addLayout(hBoxUp)
        mainVbox.addLayout(hBoxDown)

        self.setLayout(mainVbox)

        '''
        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btnOpen)
        self.layout().addWidget(btnLoad)
        self.layout().addWidget(btnSave)
        self.layout().addWidget(btnExport)
        self.layout().addWidget(chkEdit)
        '''

        # greeting
        #print('AnnotatorJ plugin is started | Happy annotations!')
        print('----------------------------\nAnnotatorJ plugin is started\nHappy annotations!\n----------------------------')

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
            # TODO
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

        imageLayer = self.viewer.add_image(img,name='Image')

        # check if a shapes layer already exists for the rois
        # if so, bring it forward
        self.findROIlayer(True)
        self.viewer.reset_view()



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
        roiProps={'name':[],'class':[],'nameInt':[]}
        roiTextProps={
            'text': '{nameInt}: ({class})',
            'anchor': 'center',
            'size': 10,
            'color': 'black',
            'visible':False
        }
        # add an empty shapes layer
        shapesLayer=Shapes(data=None,shape_type='polygon',name='ROI',edge_width=0.5,edge_color='white',face_color=[0,0,0,0],properties=None,text=roiTextProps)
        self.viewer.add_layer(shapesLayer)
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
        colours=['red','green','blue','cyan','magenta','yellow','orange','white','black']
        # TODO: add much more colours!

        self.classColourLUT={}
        for x in classIdxs:
            self.classColourLUT.update({x:colours[x-1]}) # classes are only considered when class>0

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


    def findROIlayer(self,setLayer=False,layerName='ROI'):
        for x in self.viewer.layers:
            if (x.__class__ is Shapes and x.name==layerName):
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


    def addROIdata(self,layer,rois):
        roiType='polygon' # default to this
        defColour='white'

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
                curColour=defColour
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

    def saveROIs(self):
        # open a save dialog and save the rois to an imagej compatible roi.zip file
        print('saving...')
        # TODO: rename rois

        # set output folder and create it
        selectedClass='masks'
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
    def freeHandROI(self,layer, event):
        #data_coordinates = layer.world_to_data(event.position)
        #cords = numpy.round(data_coordinates).astype(int)
        #print('Clicked at {} on layer {}'.format(cords,layer.name))
        yield
        #layer.data[0] is an array of points in the shape

        if layer.mode=='add_polygon':
            dragged=False
            freeCoords=[]
            defColour='white'
            # on move
            while event.type == 'mouse_move':
                dragged = True
                coords = list(layer.world_to_data(event.position))
                freeCoords.append(coords)
                yield
            # on release
            if dragged:
                # drag ended
                # mimic an 'esc' key press to quit the basic add_polygon method
                key_bindings.finish_drawing_shape(layer)
                print(freeCoords)
                # add the coords as a new shape
                layer.add(data=freeCoords,shape_type='polygon',edge_width=0.5,edge_color=defColour,face_color=[0,0,0,0])
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
            defColour='white'
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


    def updateNewROIprops(self,event):
        roiLayer=self.findROIlayer()
        #debug:
        print(f'---- {len(roiLayer.data)} rois on layer ----')
        print(f'---- {self.roiCount} rois in manager ----')

        # check the number of shapes on the layer
        n=len(roiLayer.data)
        if n==1:
            # empty shapes layer, init the props
            roiLayer.properties={'name':['0001'],'class':[0],'nameInt':[1]}
        elif self.roiCount<n:
            # the latest roi is the new one, rename it
            lastNumber=roiLayer.properties['nameInt'][-2] # second last in the list
            roiLayer.properties['nameInt'][-1]=lastNumber+1
            roiLayer.properties['name'][-1]='{:04d}'.format(lastNumber+1)
            # default class is 0 (no class)
            roiLayer.properties['class'][-1]=0
        elif self.roiCount>n:
            self.roiCount=n-1

        # update text properties for display option
        roiLayer.text.refresh_text(roiLayer.properties)
        self.roiCount+=1
        print(f'roiCount: {self.roiCount}')
        self.roiLayer=roiLayer


    def initROItextProps(self):
        roiLayer=self.findROIlayer()
        initProps={'name': array(['0001'], dtype='<U4'),'class': array([0]),'nameInt': array([1])}
        roiLayer.text.add(initProps,1)


    def quickExport(self):
        # save the ImageJ ROI files as when pressing the "Save" button
        self.saveROIs()

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


    def setEditMode(self,state):
        shapesLayer=self.findROIlayer()
        if state == Qt.Checked:
            self.editMode=True
            print('Edit mode selected')

            # set the "select shapes" mode
            shapesLayer.mode = 'select'

        else:
            self.editMode=False
            print('Edit mode cleared')
            # set the "add polygon" mode
            shapesLayer.mode = 'add_polygon'


    def showCnt(self,state):
        shapesLayer=self.findROIlayer()
        if state == Qt.Checked:
            print('Show contours selected')
            shapesLayer.visible=True

        else:
            print('Show contours cleared')
            shapesLayer.visible=False


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
