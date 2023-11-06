import napari_annotatorj
import pytest
#from .test_dock_widget import im_layer_dims
from napari.layers import Shapes, Image, Labels, Layer
from napari.layers.shapes import _shapes_mouse_bindings as mouse_bindings
from roifile import ImagejRoi
import cv2
import numpy
from qtpy.QtCore import Qt,QTimer,QPoint
from qtpy.QtWidgets import QApplication
from time import sleep
import os

# helper fcns
def countLayers(layers,layerType=Image):
    c=0
    for x in layers:
        if (x.__class__ is layerType):
            c+=1
    return c

@pytest.fixture
def startAnnotatorJwidget(qtbot,make_napari_viewer):
    def doStart():
        viewer = make_napari_viewer()
        pluginInstance=napari_annotatorj.AnnotatorJ(viewer)
        qtbot.addWidget(pluginInstance)
        return pluginInstance
    return doStart

def im_layer(dims):
    # dims is like (100,100,3)
    return Image(numpy.array(numpy.random.random(dims)),name='Image')

@pytest.fixture
def write_test_file(tmp_path):
    def write(filename):
        # tmp_path is a pytest fixture, it will be cleaned up when the tests are run
        testFile=str(tmp_path/filename)
        testData=numpy.random.random((20,20))
        cv2.imwrite(testFile,testData)

        return testFile,testData

    return write

def init_annotatorj_w_image(startAnnotatorJwidget):
    pluginInstance=startAnnotatorJwidget()
    dims=(100,100,3)
    dummy_image=im_layer(dims)
    imageLayer = pluginInstance.viewer.add_layer(dummy_image)
    # mock a name so init fcn doesn't fail
    pluginInstance.destNameRaw='dummy'
    return pluginInstance

@pytest.fixture
def save_roi_2_mask(tmp_path):
    def write_mask(filename,shapesLayer,w,h):
        labels=shapesLayer.to_labels([w,h])
        outputFileName=os.path.join(tmp_path,filename)
        napari_annotatorj.ExportFrame.saveExportedImage(labels,outputFileName)
        return outputFileName,labels
    return write_mask

@pytest.fixture
def save_roi_2_text(tmp_path):
    def write_text(filename,exporter,shapesLayer,format=0,w=256,h=256):
        if format==0:
            bboxes=exporter.fillBboxList(shapesLayer,len(shapesLayer.data))
            exporter.saveExportedCSV(bboxes,os.path.join(tmp_path,filename))
            return True
        elif format==1:
            bboxes=exporter.fillBboxListYOLO(shapesLayer,len(shapesLayer.data),w,h)
            exporter.saveExportedCSVyolo(bboxes,os.path.join(tmp_path,filename))
            return True
        else:
            return False
    return write_text


def test_reset_annot_type(startAnnotatorJwidget):
    pluginInstance=startAnnotatorJwidget()
    # reset annot type
    pluginInstance.selectedAnnotationType='instance'
    pluginInstance.writeParams2File()

def test_init_fcns(startAnnotatorJwidget):
    pluginInstance=startAnnotatorJwidget()
    pluginInstance.preInitChkBxs()
    assert pluginInstance.contAssist==False, f'expected contAssist to be {False} but was {pluginInstance.contAssist}'
    assert pluginInstance.editMode==False, f'expected editMode to be {False} but was {pluginInstance.editMode}'

    pluginInstance.initChkBoxes()
    assert pluginInstance.selectedAnnotationType=='instance',f'expected selectedAnnotationType to be instance but was {pluginInstance.selectedAnnotationType}'
    assert pluginInstance.classMode==False, f'expected classMode to be {False} but was {pluginInstance.classMode}'
    assert pluginInstance.addAuto==False, f'expected addAuto to be {False} but was {pluginInstance.addAuto}'
    assert pluginInstance.chkEdit.isEnabled()==True,f'expected chkEdit enabled to be {True} but was {pluginInstance.chkEdit.isEnabled()}'
    assert pluginInstance.chckbxContourAssist.isEnabled()==True,f'expected chckbxContourAssist enabled to be {True} but was {pluginInstance.chckbxContourAssist.isEnabled()}'
    assert pluginInstance.chckbxClass.isEnabled()==True,f'expected chckbxClass enabled to be {True} but was {pluginInstance.chckbxClass.isEnabled()}'

    print('init fcns tested')


def test_opening_image_randgen(tmp_path,startAnnotatorJwidget):
    pluginInstance=startAnnotatorJwidget()
    sleep(1)
    # fake a 100x100x3 random rgb image
    dims=(100,100,3)
    expected_width=0.5
    dummy_image=im_layer(dims)
    prev_count_im=countLayers(pluginInstance.viewer.layers)
    prev_count_shapes=countLayers(pluginInstance.viewer.layers,layerType=Shapes)
    # mock openNew fcn
    imageLayer = pluginInstance.viewer.add_layer(dummy_image)
    after_count_im=countLayers(pluginInstance.viewer.layers)
    assert after_count_im==prev_count_im+1,f'expected image layer count after image addition to be {prev_count_im+1} but was {after_count_im}'
    assert prev_count_shapes==1,f'expected shapes layer count to be 1 but was {prev_count_shapes}'
    s=imageLayer.data.shape
    assert s[0]==dims[0],f'expected image to have dim1 dims[0] but was {s[0]}'
    assert s[1]==dims[1],f'expected image to have dim1 dims[1] but was {s[1]}'
    assert s[2]==dims[2],f'expected image to have dim1 dims[2] but was {s[2]}'
    pluginInstance.imgSize=s
    assert pluginInstance.imgSize==imageLayer.data.shape,f'unexpected image layer size'
    pluginInstance.defDir=tmp_path
    pluginInstance.defFile='dummy_name.png'
    pluginInstance.curPredictionImageName=pluginInstance.defFile
    pluginInstance.curPredictionImage=None
    pluginInstance.curOrigImage=None

    pluginInstance.finishOpenNewInit()
    assert pluginInstance.findROIlayer is not None,f'expected to find at least 1 ROI layer'
    # the correct annotEdgeWidth for a <300 x <300 image is 0.5
    assert pluginInstance.annotEdgeWidth==expected_width,f'expected annotEdgeWidth to be {expected_width} but was {pluginInstance.annotEdgeWidth}'
    assert pluginInstance.started==True,f'expected started to be True but was {pluginInstance.started}'


def test_opening_image_randgen_size(tmp_path,startAnnotatorJwidget):
    sizes=[(100,100,3),
        (300,300,3),(400,400,3),(500,500,3),(700,700,3),
        (1000,1000,3),(1200,1200,3),(1500,1500,3),(1800,1800,3),
        (2000,2000,3),(2300,2300,3),(3000,3000,3),(3100,3100,3)]
    expected_width=[0.5,
        0.5,1.0,1.0,1.5,
        1.5,2.0,2.0,3.0,
        3.0,5.0,5.0,7.0]
    pluginInstance=startAnnotatorJwidget()
    for s,e in zip(sizes,expected_width):
        # fake a random rgb image
        dims=s
        expected_width=e
        dummy_image=im_layer(dims)
        imageLayer = pluginInstance.viewer.add_layer(dummy_image)
        s=imageLayer.data.shape
        assert s[0]==dims[0],f'expected image to have dim1 dims[0] but was {s[0]}'
        assert s[1]==dims[1],f'expected image to have dim1 dims[1] but was {s[1]}'
        assert s[2]==dims[2],f'expected image to have dim1 dims[2] but was {s[2]}'
        pluginInstance.imgSize=s
        assert pluginInstance.imgSize==imageLayer.data.shape,f'unexpected image layer size'
        pluginInstance.defDir=tmp_path
        pluginInstance.defFile='dummy_name.png'
        pluginInstance.curPredictionImageName=pluginInstance.defFile
        pluginInstance.curPredictionImage=None
        pluginInstance.curOrigImage=None

        pluginInstance.finishOpenNewInit()
        assert pluginInstance.findROIlayer is not None,f'expected to find at least 1 ROI layer'
        assert pluginInstance.annotEdgeWidth==expected_width,f'expected annotEdgeWidth to be {expected_width} but was {pluginInstance.annotEdgeWidth}'


def test_opening_image_randgen_semantic(tmp_path,startAnnotatorJwidget):
    pluginInstance=startAnnotatorJwidget()
    sleep(1)
    pluginInstance.selectedAnnotationType='semantic'
    # fake a 100x100x3 random rgb image
    dims=(100,100,3)
    expected_width=0.5
    dummy_image=im_layer(dims)
    prev_count_im=countLayers(pluginInstance.viewer.layers)
    prev_count_shapes=countLayers(pluginInstance.viewer.layers,layerType=Shapes)
    # mock openNew fcn
    imageLayer = pluginInstance.viewer.add_layer(dummy_image)
    after_count_im=countLayers(pluginInstance.viewer.layers)
    assert after_count_im==prev_count_im+1,f'expected image layer count after image addition to be {prev_count_im+1} but was {after_count_im}'
    assert prev_count_shapes==1,f'expected shapes layer count to be 1 but was {prev_count_shapes}'
    s=imageLayer.data.shape
    pluginInstance.imgSize=s
    pluginInstance.defDir=tmp_path
    pluginInstance.defFile='dummy_name.png'
    pluginInstance.curPredictionImageName=pluginInstance.defFile
    pluginInstance.curPredictionImage=None
    pluginInstance.curOrigImage=None

    pluginInstance.finishOpenNewInit()
    #assert pluginInstance.findROIlayer is None,f'expected to find no ROI layer'
    labelLayer=pluginInstance.findLabelsLayerName(layerName='semantic')
    assert labelLayer is not None,f'expected to find semantic annotation layer'
    labelSize=labelLayer.data.shape
    assert labelSize[0]==s[0] and labelSize[1]==s[1]
    assert labelLayer.mode=='paint'
    assert labelLayer.opacity==0.5
    assert pluginInstance.started==True,f'expected started to be True but was {pluginInstance.started}'


def test_opening_image_randgen_bbox(tmp_path,startAnnotatorJwidget):
    pluginInstance=startAnnotatorJwidget()
    sleep(1)
    pluginInstance.selectedAnnotationType='bbox'
    # fake a 100x100x3 random rgb image
    dims=(100,100,3)
    expected_width=0.5
    dummy_image=im_layer(dims)
    prev_count_im=countLayers(pluginInstance.viewer.layers)
    prev_count_shapes=countLayers(pluginInstance.viewer.layers,layerType=Shapes)
    # mock openNew fcn
    imageLayer = pluginInstance.viewer.add_layer(dummy_image)
    after_count_im=countLayers(pluginInstance.viewer.layers)
    assert after_count_im==prev_count_im+1,f'expected image layer count after image addition to be {prev_count_im+1} but was {after_count_im}'
    assert prev_count_shapes==1,f'expected shapes layer count to be 1 but was {prev_count_shapes}'
    s=imageLayer.data.shape
    pluginInstance.imgSize=s
    pluginInstance.defDir=tmp_path
    pluginInstance.defFile='dummy_name.png'
    pluginInstance.curPredictionImageName=pluginInstance.defFile
    pluginInstance.curPredictionImage=None
    pluginInstance.curOrigImage=None

    pluginInstance.finishOpenNewInit()
    roiLayer=pluginInstance.findROIlayer()
    assert roiLayer is not None,f'expected to find at least 1 ROI layer'
    assert roiLayer.mode=='add_rectangle'
    assert pluginInstance.freeHandROIvis not in roiLayer.mouse_drag_callbacks
    assert pluginInstance.started==True,f'expected started to be True but was {pluginInstance.started}'

    # reset annot type
    pluginInstance.selectedAnnotationType='instance'
    pluginInstance.writeParams2File()

'''
def test_opening_image_real(tmp_path,qtbot,startAnnotatorJwidget,write_test_file):
    pluginInstance=startAnnotatorJwidget()
    file,data=write_test_file("dummy_name.png")
    pluginInstance.defDir=tmp_path
    pluginInstance.defFile='dummy_name.png'
    prev_count_im=countLayers(pluginInstance.viewer.layers)
    #qtbot.mouseClick(pluginInstance.btnOpen,Qt.LeftButton,delay=0)

    # mock clicking ok
    def handle_dialog():
        #while not pluginInstance.openQfileDialog.isVisible():
        #    pass
        while pluginInstance.openQfileDialog is None:
            pass
        #qtbot.keyPress(pluginInstance.openQfileDialog,Qt.Key_Enter)
        print(pluginInstance.openQfileDialog)
        pluginInstance.openQfileDialog.accept()
        #qtbot.keyPress(pluginInstance.openQfileDialog,Qt.Key_Enter)
        print('*/*/*/*/*/*/*/ PRESSED ENTER ON DIALOG /*/*/*/*/*/*/*/*')
    #openDialog=QApplication.focusWidget()
    #print(openDialog.__dir__())
    #openDialog.accept()
    QTimer.singleShot(500,handle_dialog)
    qtbot.mouseClick(pluginInstance.btnOpen,Qt.LeftButton,delay=0)

    after_count_im=countLayers(pluginInstance.viewer.layers)
    assert after_count_im==prev_count_im+1,f'expected image layer count after image addition to be {prev_count_im+1} but was {after_count_im}'
'''
def test_opening_test_image(qtbot,startAnnotatorJwidget):
    pluginInstance=startAnnotatorJwidget()
    sleep(10)
    pluginInstance.testMode=True
    prev_count_im=countLayers(pluginInstance.viewer.layers)
    qtbot.mouseClick(pluginInstance.btnOpen,Qt.LeftButton,delay=0)
    after_count_im=countLayers(pluginInstance.viewer.layers)
    assert after_count_im==prev_count_im+1,f'expected image layer count after image addition to be {prev_count_im+1} but was {after_count_im}'
    assert pluginInstance.curFileIdx==0
    assert pluginInstance.buttonPrev.isEnabled()==False
    assert pluginInstance.buttonNext.isEnabled()==False
    assert pluginInstance.lblCurrentFile.text()==" (1/1): img.png"
    assert pluginInstance.selectedAnnotationType=="instance"
    roiLayer=pluginInstance.findROIlayer()
    assert roiLayer.mode=="add_polygon"


def test_opening_test_image_w_ROI(qtbot,startAnnotatorJwidget):
    pluginInstance=startAnnotatorJwidget()
    sleep(10)
    pluginInstance.setTestMode(mode=True)
    qtbot.mouseClick(pluginInstance.btnOpen,Qt.LeftButton,delay=0)
    qtbot.mouseClick(pluginInstance.btnLoad,Qt.LeftButton,delay=0)
    roiLayer=pluginInstance.findROIlayer()
    assert roiLayer.mode=="add_polygon"
    assert pluginInstance.loadedROI==True
    assert pluginInstance.roiCount==49


def test_roi_loading_from_mask(tmp_path,startAnnotatorJwidget,save_roi_2_mask):
    pluginInstance=startAnnotatorJwidget()
    sleep(1)
    testName='dummy_mask.tiff'
    pluginInstance.defFile='dummy_mask.tiff'
    # load dummy roi
    rois=ImagejRoi.fromfile(pluginInstance.test_rois)
    shapesLayer=pluginInstance.extractROIdata(rois)
    assert shapesLayer is not None
    n=len(shapesLayer.data)
    assert n>0
    # add 2nd shapes layer with these rois
    pluginInstance.viewer.add_layer(shapesLayer)
    prev_count_shapes=countLayers(pluginInstance.viewer.layers,layerType=Shapes)
    assert prev_count_shapes==2
    filename,labelLayer=save_roi_2_mask(testName,shapesLayer,256,256)
    assert countLayers(pluginInstance.viewer.layers,layerType=Shapes)==2
    pluginInstance.loadRoisFromMask(tmp_path,False)
    after_count_shapes=countLayers(pluginInstance.viewer.layers,layerType=Shapes)
    assert after_count_shapes==prev_count_shapes
    newShapesLayer=pluginInstance.viewer.layers[-1]
    assert len(newShapesLayer.data)==2*n
    pluginInstance.viewer.layers.remove('ROI')
    pluginInstance.loadRoisFromMask(tmp_path,False)
    after_count_shapes=countLayers(pluginInstance.viewer.layers,layerType=Shapes)
    assert after_count_shapes==prev_count_shapes
    newShapesLayer=pluginInstance.viewer.layers[-1]
    assert len(newShapesLayer.data)==n


def test_classified_roi_loading(qtbot,startAnnotatorJwidget):
    pluginInstance=startAnnotatorJwidget()
    # use the classified roi.zip file
    pluginInstance.test_rois=pluginInstance.test_rois[:-4]+'_classes.zip'
    pluginInstance.setTestMode(True)
    qtbot.mouseClick(pluginInstance.btnLoad,Qt.LeftButton,delay=0)
    roiLayer=pluginInstance.findROIlayer()
    assert roiLayer is not None
    assert len(roiLayer.data)>0
    assert pluginInstance.startedClassifying==True


def test_add2roimanager(tmp_path,startAnnotatorJwidget):
    pluginInstance=startAnnotatorJwidget()
    sleep(1)
    # load dummy roi
    rois=ImagejRoi.fromfile(pluginInstance.test_rois)
    pluginInstance.add2RoiManager(rois)
    assert pluginInstance.manager is not None
    assert pluginInstance.roiCount>0 and pluginInstance.roiCount==len(rois)

    # init the manager too
    pluginInstance.viewer.layers.remove('ROI')
    pluginInstance.manager=None
    pluginInstance.add2RoiManager(rois)
    assert pluginInstance.manager is not None
    assert pluginInstance.roiCount>0 and pluginInstance.roiCount==len(rois)


def test_roi_loading_from_coords(tmp_path,startAnnotatorJwidget,save_roi_2_text):
    pluginInstance=startAnnotatorJwidget()
    sleep(1)
    testName='dummy_mask.csv'
    testName2='dummy_mask.txt'
    pluginInstance.defFile='dummy_mask.tiff'
    # load dummy roi
    rois=ImagejRoi.fromfile(pluginInstance.test_rois)
    shapesLayer=pluginInstance.extractROIdata(rois)
    prev_count_shapes=countLayers(pluginInstance.viewer.layers,layerType=Shapes)
    assert shapesLayer is not None
    n=len(shapesLayer.data)
    assert n>0
    assert prev_count_shapes==1
    pluginInstance.testMode=False

    # start an exporter
    pluginInstance.ExportFrame=napari_annotatorj.ExportFrame(pluginInstance.viewer,annotatorjObj=pluginInstance)
    # default COCO coords format (absolute)
    assert pluginInstance.ExportFrame.bboxFormat==0
    tmp=save_roi_2_text(testName,pluginInstance.ExportFrame,shapesLayer,0)
    assert os.path.isfile(os.path.join(tmp_path,testName))

    # see that absolute coords loading doesn't fail when no image is opened
    pluginInstance.loadRoisFromCoords(tmp_path)
    after_count_shapes=countLayers(pluginInstance.viewer.layers,layerType=Shapes)
    assert after_count_shapes==prev_count_shapes+1
    newShapesLayer=pluginInstance.viewer.layers[-1]
    assert newShapesLayer.name=='ROI'
    assert len(newShapesLayer.data)==n
    assert 'ROI' in pluginInstance.viewer.layers and 'ROI_prev' in pluginInstance.viewer.layers
    pluginInstance.viewer.layers.remove('ROI')
    assert countLayers(pluginInstance.viewer.layers,layerType=Shapes)==prev_count_shapes

    # delete the tmp file to ensure the next load doesn't attempt to load it
    os.remove(os.path.join(tmp_path,testName))
    assert not os.path.isfile(os.path.join(tmp_path,testName))

    # see that coord loading fails when no image is opened to calculate the relative coords to xy positions when saved as YOLO coords
    pluginInstance.ExportFrame.bboxFormat=1
    assert pluginInstance.ExportFrame.bboxFormat==1
    tmp=save_roi_2_text(testName2,pluginInstance.ExportFrame,shapesLayer,1,256,256) # test image size
    assert os.path.isfile(os.path.join(tmp_path,testName2))
    pluginInstance.loadRoisFromCoords(tmp_path)
    after_count_shapes=countLayers(pluginInstance.viewer.layers,layerType=Shapes)
    assert after_count_shapes==prev_count_shapes # no new loaded roi layer

    # now add the image then load the relative coords file again to succeed
    pluginInstance.testMode=True
    pluginInstance.openNew()
    # check if the image is opened
    assert countLayers(pluginInstance.viewer.layers)==1
    pluginInstance.testMode=False
    pluginInstance.defFile='dummy_mask.tiff'
    pluginInstance.loadRoisFromCoords(tmp_path)
    after_count_shapes=countLayers(pluginInstance.viewer.layers,layerType=Shapes)
    assert after_count_shapes==prev_count_shapes+2
    newShapesLayer=pluginInstance.viewer.layers[-1]
    assert len(newShapesLayer.data)==n


def test_saveData(qtbot,tmp_path,startAnnotatorJwidget,write_test_file):
    pluginInstance=init_annotatorj_w_image(startAnnotatorJwidget)
    testName='dummy.png'
    pluginInstance.defDir=tmp_path
    pluginInstance.defFile=testName
    testFile,testData=write_test_file(testName)
    assert countLayers(pluginInstance.viewer.layers)==1
    # load dummy roi
    rois=ImagejRoi.fromfile(pluginInstance.test_rois)
    shapesLayer=pluginInstance.extractROIdata(rois)
    pluginInstance.viewer.add_layer(shapesLayer)
    assert countLayers(pluginInstance.viewer.layers,layerType=Shapes)==2
    roiLayer=pluginInstance.findROIlayer()
    assert len(roiLayer.data)>0
    assert pluginInstance.stepping==False
    assert pluginInstance.startedClassifying==False
    assert pluginInstance.started==False
    assert pluginInstance.findImageLayer() is not None and pluginInstance.findImageLayer().data is not None
    assert pluginInstance.findOpenedImage()==True
    pluginInstance.started=True
    assert pluginInstance.started==True
    # save
    def handle_dialog():
        # get a reference to the dialog and handle it here
        while pluginInstance.classSelectionDialog is None:
            pass
        pluginInstance.saveClassSelectionOk()
    QTimer.singleShot(500, handle_dialog)
    qtbot.mouseClick(pluginInstance.btnSave,Qt.LeftButton,delay=1)
    assert pluginInstance.selectedClass=='normal'
    assert os.path.isfile(os.path.join(tmp_path,'normal','dummy_ROIs.zip'))
    assert pluginInstance.finishedSaving==True



def test_init_annotatorj_w_image(startAnnotatorJwidget):
    pluginInstance=startAnnotatorJwidget()
    sleep(10)
    assert pluginInstance.contAssist==False
    assert pluginInstance.classMode==False
    assert pluginInstance.editMode==False
    assert pluginInstance.chkEdit.isEnabled()==True
    assert pluginInstance.chckbxClass.isEnabled()==True
    prev_count_shapes=countLayers(pluginInstance.viewer.layers,layerType=Shapes)

    # add a dummy image layer
    # fake a 100x100x3 random rgb image
    dims=(100,100,3)
    dummy_image=im_layer(dims)
    prev_count_im=countLayers(pluginInstance.viewer.layers)
    imageLayer = pluginInstance.viewer.add_layer(dummy_image)
    after_count_im=countLayers(pluginInstance.viewer.layers)
    assert after_count_im==prev_count_im+1,f'expected image layer count after image addition to be {prev_count_im+1} but was {after_count_im}'
    return pluginInstance


# ---------------

# test checkboxes
def test_checkbox_init_contAssist(startAnnotatorJwidget):
    pluginInstance=init_annotatorj_w_image(startAnnotatorJwidget)
    pluginInstance.chckbxContourAssist.setChecked(True)
    sleep(1)
    assert pluginInstance.chckbxContourAssist.isChecked()==True
    assert pluginInstance.contAssist==True
    pluginInstance.preInitChkBxs()
    assert pluginInstance.chckbxAddAutomatically.isEnabled()==False
    assert pluginInstance.editMode==False
    assert pluginInstance.classMode==False
    assert pluginInstance.chckbxClass.isEnabled()==False


def test_checkbox_init_editMode(startAnnotatorJwidget):
    pluginInstance=init_annotatorj_w_image(startAnnotatorJwidget)
    pluginInstance.chkEdit.setChecked(True)
    sleep(1)
    assert pluginInstance.chkEdit.isChecked()==True
    assert pluginInstance.editMode==True
    pluginInstance.preInitChkBxs()
    assert pluginInstance.chckbxAddAutomatically.isEnabled()==False
    assert pluginInstance.contAssist==False
    assert pluginInstance.classMode==False
    assert pluginInstance.chckbxClass.isEnabled()==False
    assert pluginInstance.chckbxContourAssist.isEnabled()==False


def test_checkbox_init_classMode(startAnnotatorJwidget):
    pluginInstance=init_annotatorj_w_image(startAnnotatorJwidget)
    pluginInstance.chckbxClass.setChecked(True)
    sleep(1)
    assert pluginInstance.chckbxClass.isChecked()==True
    assert pluginInstance.classMode==True
    assert pluginInstance.editMode==False
    pluginInstance.initChkBoxes()
    assert pluginInstance.chkEdit.isEnabled()==False
    assert pluginInstance.editMode==False
    assert pluginInstance.contAssist==False
    assert pluginInstance.classMode==True
    assert pluginInstance.chckbxContourAssist.isEnabled()==False


def test_checkbox_init_editMode2(startAnnotatorJwidget):
    pluginInstance=init_annotatorj_w_image(startAnnotatorJwidget)
    pluginInstance.chkEdit.setChecked(True)
    sleep(1)
    assert pluginInstance.chkEdit.isChecked()==True
    assert pluginInstance.classMode==False
    assert pluginInstance.editMode==True
    pluginInstance.initChkBoxes()
    assert pluginInstance.chckbxClass.isEnabled()==False
    assert pluginInstance.editMode==True
    assert pluginInstance.contAssist==False
    assert pluginInstance.classMode==False
    assert pluginInstance.chckbxContourAssist.isEnabled()==False


def test_checkbox_init_constAssist2(startAnnotatorJwidget):
    pluginInstance=init_annotatorj_w_image(startAnnotatorJwidget)
    pluginInstance.chckbxContourAssist.setChecked(True)
    sleep(1)
    assert pluginInstance.chckbxContourAssist.isChecked()==True
    assert pluginInstance.classMode==False
    assert pluginInstance.editMode==False
    assert pluginInstance.contAssist==True
    pluginInstance.initChkBoxes()
    assert pluginInstance.chkEdit.isEnabled()==False
    assert pluginInstance.editMode==False
    assert pluginInstance.contAssist==True
    assert pluginInstance.classMode==False
    assert pluginInstance.chckbxClass.isEnabled()==False


def test_checkbox_init_semantic(startAnnotatorJwidget):
    pluginInstance=init_annotatorj_w_image(startAnnotatorJwidget)
    pluginInstance.selectedAnnotationType='semantic'
    sleep(1)
    assert pluginInstance.classMode==False
    assert pluginInstance.editMode==False
    assert pluginInstance.contAssist==False
    pluginInstance.initChkBoxes()
    assert pluginInstance.chkEdit.isEnabled()==False
    assert pluginInstance.editMode==False
    assert pluginInstance.contAssist==False
    assert pluginInstance.classMode==False
    assert pluginInstance.chckbxContourAssist.isEnabled()==False
    assert pluginInstance.chckbxClass.isEnabled()==False


def test_checkbox_init_bbox(startAnnotatorJwidget):
    pluginInstance=init_annotatorj_w_image(startAnnotatorJwidget)
    pluginInstance.selectedAnnotationType='bbox'
    sleep(1)
    assert pluginInstance.classMode==False
    assert pluginInstance.editMode==False
    assert pluginInstance.contAssist==False
    pluginInstance.initChkBoxes()
    assert pluginInstance.chkEdit.isEnabled()==False
    assert pluginInstance.editMode==False
    assert pluginInstance.contAssist==False
    assert pluginInstance.classMode==False
    assert pluginInstance.chckbxContourAssist.isEnabled()==False
    assert pluginInstance.chckbxClass.isEnabled()==True


def test_checkbox_contAssist(startAnnotatorJwidget):
    pluginInstance=init_annotatorj_w_image(startAnnotatorJwidget)
    assert countLayers(pluginInstance.viewer.layers)==1
    prev_count_shapes=countLayers(pluginInstance.viewer.layers,layerType=Shapes)

    # start testing the checkbox
    pluginInstance.chckbxContourAssist.setChecked(True)
    sleep(1)
    assert pluginInstance.chckbxContourAssist.isChecked()==True
    assert pluginInstance.contAssist==True
    assert pluginInstance.classMode==False
    assert pluginInstance.editMode==False
    assert pluginInstance.chkEdit.isEnabled()==False
    assert pluginInstance.chckbxClass.isEnabled()==False
    after_count_shapes=countLayers(pluginInstance.viewer.layers,layerType=Shapes)
    assert after_count_shapes==prev_count_shapes+1,f'expected shapes layer count after contour assist init to be {prev_count_shapes+1} but was {after_count_shapes}'
    contAssistLayer=pluginInstance.viewer.layers[-1]
    assert contAssistLayer.name=='contourAssist'
    assert contAssistLayer.mode=='add_polygon'

    # check default unchecked states
    pluginInstance.chckbxContourAssist.setChecked(False)
    sleep(1)
    assert countLayers(pluginInstance.viewer.layers,layerType=Shapes)==prev_count_shapes
    assert pluginInstance.chckbxContourAssist.isChecked()==False
    assert pluginInstance.contAssist==False

    # remove image to trigger fail
    pluginInstance.viewer.layers.remove('Image')
    assert countLayers(pluginInstance.viewer.layers)==0
    pluginInstance.chckbxContourAssist.setChecked(True)
    sleep(1)
    assert pluginInstance.chckbxContourAssist.isChecked()==True
    assert pluginInstance.contAssist==False


def test_checkbox_classMode(startAnnotatorJwidget):
    pluginInstance=init_annotatorj_w_image(startAnnotatorJwidget)
    assert countLayers(pluginInstance.viewer.layers)==1
    prev_count_shapes=countLayers(pluginInstance.viewer.layers,layerType=Shapes)
    num_dw = len(pluginInstance.viewer.window._dock_widgets)

    # start testing the checkbox
    pluginInstance.chckbxClass.setChecked(True)
    sleep(1)
    assert pluginInstance.chckbxClass.isChecked()==True
    assert pluginInstance.contAssist==False
    assert pluginInstance.classMode==True
    assert pluginInstance.editMode==False
    assert pluginInstance.chkEdit.isEnabled()==True
    assert pluginInstance.chckbxContourAssist.isEnabled()==False
    after_count_shapes=countLayers(pluginInstance.viewer.layers,layerType=Shapes)
    assert after_count_shapes==prev_count_shapes,f'expected shapes layer count after contour assist init to be {prev_count_shapes} but was {after_count_shapes}'
    roiLayer=pluginInstance.findROIlayer()
    assert roiLayer.name=='ROI'
    assert roiLayer.mode=='select'

    assert len(pluginInstance.viewer.window._dock_widgets)==num_dw+1
    assert pluginInstance.classesFrame is not None,f'Failed to open Classes widget'
    assert len(pluginInstance.classFrameNames)==2

    # check default unchecked states
    pluginInstance.chckbxClass.setChecked(False)
    sleep(1)
    assert pluginInstance.chckbxClass.isChecked()==False
    assert pluginInstance.classMode==False
    assert pluginInstance.chckbxContourAssist.isEnabled()==True
    roiLayer=pluginInstance.findROIlayer()
    assert roiLayer.mode=='add_polygon'
    assert mouse_bindings.select not in roiLayer.mouse_drag_callbacks
    assert pluginInstance.customShapesLayerSelect not in roiLayer.mouse_drag_callbacks

    # check semantic disabled state
    pluginInstance.selectedAnnotationType='semantic'
    pluginInstance.chckbxClass.setChecked(True)
    sleep(1)
    assert pluginInstance.chckbxClass.isChecked()==True
    assert pluginInstance.classMode==False

    # reset
    pluginInstance.selectedAnnotationType='instance'
    pluginInstance.chckbxClass.setChecked(False)

    # remove image to trigger fail
    pluginInstance.viewer.layers.remove('Image')
    assert countLayers(pluginInstance.viewer.layers)==0
    pluginInstance.chckbxClass.setChecked(True)
    sleep(1)
    assert pluginInstance.chckbxClass.isChecked()==True
    assert pluginInstance.classMode==False



def test_the_fcns(startAnnotatorJwidget):
    #pluginInstance=startAnnotatorJwidget()
    #test_init_fcns(startAnnotatorJwidget)
    #test_opening_image_randgen(startAnnotatorJwidget,im_layer)
    pass