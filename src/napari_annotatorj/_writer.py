"""
This module is an example of a barebones writer plugin for napari

It implements the ``napari_get_writer`` and ``napari_write_image`` hook specifications.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs
"""

import skimage.io
import numpy

supported_layers = ['labels']


def napari_get_writer(path, layer_types):
	# Check that only supported layers have been passed
	for x in set(layer_types):
		if x not in supported_layers:
			return None
	
	if isinstance(path, str) and path.endswith('.tiff'):
		return napari_write_labels
	else:
		return None
	return napari_write_labels


def napari_write_labels(path: str, data: numpy.ndarray, meta: dict):
	if data is None:
		return None
	out=data.astype('uint16')
	skimage.io.imsave(path,out,check_contrast=False)
	print('Saved exported image: {}'.format(path))
	print('---------------------')

	return path
