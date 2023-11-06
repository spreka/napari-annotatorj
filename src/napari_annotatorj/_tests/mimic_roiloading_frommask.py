import napari
from _dock_widget import AnnotatorJ
from roifile import ImagejRoi
from napari.layers import Shapes, Image, Labels, Layer

def countLayers(layers,layerType=Image):
    c=0
    for x in layers:
        if (x.__class__ is layerType):
            c+=1
    return c

viewer = napari.Viewer()
pluginInstance=AnnotatorJ(viewer)
viewer.window.add_dock_widget(pluginInstance,name='AnnotatorJ')

tmp_path="C:/work/szbk/annotatorj/testing_napari/dummying_nucleus/masks_demo/labelled_masks/"
pluginInstance.defFile='full adj.tiff'
# load dummy roi
rois=ImagejRoi.fromfile(pluginInstance.test_rois)
shapesLayer=pluginInstance.extractROIdata(rois)
n=len(shapesLayer.data)
print(f'numshapes: {n}')
# add 2nd shapes layer with these rois
pluginInstance.viewer.add_layer(shapesLayer)
prev_count_shapes=countLayers(pluginInstance.viewer.layers,layerType=Shapes)
print(f'before: {prev_count_shapes}')
#'''
pluginInstance.viewer.layers.remove('ROI')
print(f'is 2?: {countLayers(pluginInstance.viewer.layers,layerType=Shapes)}')
pluginInstance.loadRoisFromMask(tmp_path,False)
after_count_shapes=countLayers(pluginInstance.viewer.layers,layerType=Shapes)
print(f'after: {after_count_shapes}')
newShapesLayer=pluginInstance.viewer.layers[-1]
print(f'new shapes num: {len(newShapesLayer.data)}')
#'''

napari.run()