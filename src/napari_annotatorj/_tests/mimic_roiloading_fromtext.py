import napari
from _dock_widget import AnnotatorJ,ExportFrame
from roifile import ImagejRoi
from napari.layers import Shapes, Image, Labels, Layer
import os,sys

def countLayers(layers,layerType=Image):
    c=0
    for x in layers:
        if (x.__class__ is layerType):
            c+=1
    return c

viewer = napari.Viewer()
pluginInstance=AnnotatorJ(viewer)
viewer.window.add_dock_widget(pluginInstance,name='AnnotatorJ')

tmp_path="C:/work/szbk/annotatorj/testing_napari/dummying_nucleus/coording/"
pluginInstance.defFile='dummy_mask.tiff'
filename='dummy_mask.csv'
filename2='dummy_mask.txt'
# load dummy roi
rois=ImagejRoi.fromfile(pluginInstance.test_rois)
shapesLayer=pluginInstance.extractROIdata(rois)
n=len(shapesLayer.data)
print(f'numshapes: {n}')
prev_count_shapes=countLayers(pluginInstance.viewer.layers,layerType=Shapes)
print(f'before: {prev_count_shapes}')

if os.path.isfile(os.path.join(tmp_path,filename)):
    os.remove(os.path.join(tmp_path,filename))
if os.path.isfile(os.path.join(tmp_path,filename2)):
    os.remove(os.path.join(tmp_path,filename2))

pluginInstance.ExportFrame=ExportFrame(pluginInstance.viewer,annotatorjObj=pluginInstance)
bboxes=pluginInstance.ExportFrame.fillBboxList(shapesLayer,len(shapesLayer.data))
pluginInstance.ExportFrame.saveExportedCSV(bboxes,os.path.join(tmp_path,filename))
success=os.path.isfile(os.path.join(tmp_path,filename))
print(f'wrote file: {success}')

pluginInstance.loadRoisFromCoords(tmp_path)
print(f'loaded: {countLayers(pluginInstance.viewer.layers,layerType=Shapes)}')
#'''
after_count_shapes=countLayers(pluginInstance.viewer.layers,layerType=Shapes)
print(f'after: {after_count_shapes}')
newShapesLayer=pluginInstance.viewer.layers[-1]
print(f'new shapes num: {len(newShapesLayer.data)}')
#'''

# remove it
print(f'layers before REMOVE: {pluginInstance.viewer.layers}')
pluginInstance.viewer.layers.remove('ROI')
print(f'--------------- REMOVING ROI LAYER -----------------')
print(f'after remove layer count: {countLayers(pluginInstance.viewer.layers,layerType=Shapes)}')
print(f'layers after REMOVE: {pluginInstance.viewer.layers}')
os.remove(os.path.join(tmp_path,filename))

# write yolo file
bboxes=pluginInstance.ExportFrame.fillBboxListYOLO(shapesLayer,n,256,256)
pluginInstance.ExportFrame.saveExportedCSVyolo(bboxes,os.path.join(tmp_path,filename2))
success=os.path.isfile(os.path.join(tmp_path,filename2))
print(f'wrote second file: {success}')

pluginInstance.loadRoisFromCoords(tmp_path)
print(f'loaded 2nd: {countLayers(pluginInstance.viewer.layers,layerType=Shapes)}')
newShapesLayer=pluginInstance.viewer.layers[-1]
print(f'numshapes 2nd: {len(newShapesLayer.data)}')

pluginInstance.testMode=True
pluginInstance.openNew()
print(f'opened image')
pluginInstance.testMode=False
pluginInstance.defFile='dummy_mask.tiff'
pluginInstance.loadRoisFromCoords(tmp_path)
print(f'loaded 3rd: {countLayers(pluginInstance.viewer.layers,layerType=Shapes)}')
newShapesLayer=pluginInstance.viewer.layers[-1]
print(f'numshapes 3rd: {len(newShapesLayer.data)}')


napari.run()




