import napari_annotatorj
import pytest
from time import sleep
from napari_annotatorj._dock_widget import OptionsFrame,ColourSelector,TrainWidget,HelpWidget,Q3DWidget,FileListWidget

# this is your plugin name declared in your napari.plugins entry point
MY_PLUGIN_NAME = "napari-annotatorj"
# the name of your widget(s)
MY_WIDGET_NAMES = ["AnnotatorJ","AnnotatorJExport"]

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
@pytest.fixture
def test_annotatorj_widget():
    def test_main_widget_loading(qtbot,make_napari_viewer):
        viewer = make_napari_viewer()
        pluginInstance=napari_annotatorj.AnnotatorJ(viewer)
        qtbot.addWidget(pluginInstance)
        num_dw = len(viewer.window._dock_widgets)
        viewer.window.add_dock_widget(pluginInstance,name='AnnotatorJ')
        assert len(viewer.window._dock_widgets) == num_dw + 1

        # give some time for the plugin init function to complete successfully on a thread loading the unet model
        sleep(10)
        #assert pluginInstance.trainedUNetModel is not None, f'trainedUnetModel is None'
        return pluginInstance

    return test_main_widget_loading
#'''

@pytest.mark.parametrize("widget_name", MY_WIDGET_NAMES)
def test_all_widget_loading(widget_name, make_napari_viewer, napari_plugin_manager):
    napari_plugin_manager.register(napari_annotatorj, name=MY_PLUGIN_NAME)
    viewer = make_napari_viewer()
    num_dw = len(viewer.window._dock_widgets)
    viewer.window.add_plugin_dock_widget(
        plugin_name=MY_PLUGIN_NAME, widget_name=widget_name
    )
    assert len(viewer.window._dock_widgets) == num_dw + 1

def test_widget_loading_extras(make_napari_viewer,qtbot):
    viewer=make_napari_viewer()
    pluginInstance=napari_annotatorj.AnnotatorJ(viewer)
    qtbot.addWidget(pluginInstance)
    num_dw = len(viewer.window._dock_widgets)
    viewer.window.add_dock_widget(pluginInstance,name='AnnotatorJ')
    assert len(viewer.window._dock_widgets) == num_dw + 1

    # add classes widget
    pluginInstance.classesFrame=napari_annotatorj.ClassesFrame(pluginInstance.viewer,pluginInstance)
    assert len(viewer.window._dock_widgets) == num_dw + 2

    # add options widget
    pluginInstance.openOptionsFrame()
    assert len(viewer.window._dock_widgets) == num_dw + 3

    # add colours widget
    pluginInstance.addColourWidget()
    assert len(viewer.window._dock_widgets) == num_dw + 4

    # add training widget
    pluginInstance.openTrainWidget()
    assert len(viewer.window._dock_widgets) == num_dw + 5

    # add help widget
    pluginInstance.openHelpWidgetDock()
    assert len(viewer.window._dock_widgets) == num_dw + 6

    # add 3d widget
    pluginInstance.open3DWidget()
    assert len(viewer.window._dock_widgets) == num_dw + 7

    # add filelist widget
    pluginInstance.openFileListWidget()
    assert len(viewer.window._dock_widgets) == num_dw + 8


@pytest.fixture
def im_layer():
    return Image(np.array(np.random.random((100,100,3))),name='Image')