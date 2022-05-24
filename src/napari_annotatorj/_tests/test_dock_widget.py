import napari_annotatorj
import pytest
from time import sleep

# this is your plugin name declared in your napari.plugins entry point
MY_PLUGIN_NAME = "napari-annotatorj"
# the name of your widget(s)
MY_WIDGET_NAMES = ["AnnotatorJ"]

'''
@pytest.mark.parametrize("widget_name", MY_WIDGET_NAMES)
def test_something_with_viewer(widget_name, make_napari_viewer, napari_plugin_manager):
    napari_plugin_manager.register(napari_annotatorj, name=MY_PLUGIN_NAME)
    viewer = make_napari_viewer()
    num_dw = len(viewer.window._dock_widgets)
    viewer.window.add_plugin_dock_widget(
        plugin_name=MY_PLUGIN_NAME, widget_name=widget_name
    )
    assert len(viewer.window._dock_widgets) == num_dw + 1
'''

#'''
def test_widet_loading(qtbot,make_napari_viewer):
    viewer = make_napari_viewer()
    pluginInstance=napari_annotatorj.AnnotatorJ(viewer)
    qtbot.addWidget(pluginInstance)
    num_dw = len(viewer.window._dock_widgets)
    viewer.window.add_dock_widget(pluginInstance,name='AnnotatorJ')
    assert len(viewer.window._dock_widgets) == num_dw + 1

    # give some time for the plugin init function to complete successfully on a thread loading the unet model
    sleep(10)
    #assert pluginInstance.trainedUNetModel is not None, f'trainedUnetModel is None'
#'''

@pytest.fixture
def im_layer():
    return Image(np.random.random(5,100,100),name='Image')