import numpy as np
import skimage.io
from napari_annotatorj import napari_get_reader
import pytest

@pytest.fixture
def write_im_to_file(tmp_path):

    def write_func(filename):
        # write some fake data using your supported file format
        #my_test_file = str(tmp_path / "myfile.npy")
        my_test_file = str(tmp_path / "myfile.tiff")
        original_data = np.random.rand(20, 20)
        #np.save(my_test_file, original_data)
        skimage.io.imsave(my_test_file, original_data)

        return my_test_file,original_data

    return write_func

# tmp_path is a pytest fixture
def test_reader(tmp_path):
    """An example of how you might test your plugin."""

    my_test_file = str(tmp_path / "myfile.tiff")
    original_data = np.random.rand(20, 20)
    skimage.io.imsave(my_test_file, original_data)

    # try to read it back in
    reader = napari_get_reader(my_test_file)
    assert callable(reader)


def test_reader_return(write_im_to_file):
    my_test_file,original_data=write_im_to_file("myfile.tiff")
    reader = napari_get_reader(my_test_file)
    assert callable(reader)

    # make sure we're delivering the right format
    layer_data_list = reader(my_test_file)
    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) > 0

    # make sure it's the same as it started
    np.testing.assert_allclose(original_data, layer_data_tuple[0])


def test_get_reader_pass():
    reader = napari_get_reader("fake.file")
    assert reader is None


def test_get_reader_path_list(write_im_to_file):
    path1,_=write_im_to_file("myfile1.tiff")
    path2,_=write_im_to_file("myfile2.tiff")

    reader=napari_get_reader([path1,path2])
    assert callable(reader)
