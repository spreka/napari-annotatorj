'''
Runs the AnnotatorJ plugin.
'''

import napari
from _dock_widget import AnnotatorJ

viewer = napari.Viewer()
pluginInstance=AnnotatorJ(viewer)
viewer.window.add_dock_widget(pluginInstance,name='AnnotatorJ')
napari.run()