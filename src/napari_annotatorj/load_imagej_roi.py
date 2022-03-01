'''
Runs a test of the AnnotatorJ plugin's basic loading functionality. Reads a demo image and its corresponding ImageJ ROI.zip file containing annotations.
'''

import napari
from _dock_widget import AnnotatorJ

viewer = napari.Viewer()
pluginInstance=AnnotatorJ(viewer)
pluginInstance.setTestMode(True)
viewer.window.add_dock_widget(pluginInstance,name='AnnotatorJ')
pluginInstance.openNew()
pluginInstance.loadROIs()
napari.run()