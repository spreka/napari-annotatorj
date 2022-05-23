# help the manual startup script import:
try:
	from .unet.model import *
except ImportError as e:
	try:
		from unet.model import *
	except Exception as e:
		print(e)
		exit()

import skimage.io as io
import cv2
import numpy as np

def importArgs():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--results', type=str, required=True)
	parser.add_argument('--train', type=str, required=False)
	parser.add_argument('--val', type=str, default=None)
	parser.add_argument('--test', type=str, default=None)
	parser.add_argument('--epochs', type=int, default=5)
	parser.add_argument('--steps', type=int, default=300)
	parser.add_argument('--model', type=str, default=None)
	parser.add_argument('--batch', type=int, default=2)
	parser.add_argument('--gpu', type=str)
	parser.add_argument('--write', type=bool, default=True)
	parser.add_argument('--size', type=int, default=256)
	args = parser.parse_args()

	return args


def loadUnetModel(modelName,importMode=0):
	model=None
	# load json and create model
	print('starting loadUnetModel...')
	try:
		if importMode==0:
			# import from '.json' + '_weights.h5' files
			json_file = open(modelName+'.json', 'r')
			loaded_model_json = json_file.read()
			json_file.close()
			model = model_from_json(loaded_model_json)
			# load weights into new model
			model.load_weights(modelName+'_weights.h5')
		elif importMode==1:
			# import from a single '.hdf5' file
			model=load_model(modelName+'.hdf5')
	except Exception as e:
		print(f'Could not load model {modelName}_weights.h5')
		print(e)
		raise(e)
	else:
		print(f"Loaded model from disk: {modelName}_weights.h5")
	finally:
		return model


def loadUnetModelSetGpu(modelName,importMode=0,gpuSetting='0'):
	setGpu(gpuSetting)
	model=None
	# load json and create model
	print('starting loadUnetModel...')
	try:
		if importMode==0:
			# import from '.json' + '_weights.h5' files
			json_file = open(modelName+'.json', 'r')
			loaded_model_json = json_file.read()
			json_file.close()
			model = model_from_json(loaded_model_json)
			# load weights into new model
			model.load_weights(modelName+'_weights.h5')
		elif importMode==1:
			# import from a single '.hdf5' file
			model=load_model(modelName+'.hdf5')
	except Exception as e:
		print(f'Could not load model {modelName}_weights.h5')
		print(e)
		raise(e)
	else:
		print(f"Loaded model from disk: {modelName}_weights.h5")
	finally:
		return model
		

def trainIfNoModel(args):
	# if model exists skip training and only predict test images
	if os.path.isfile(args.model+'.json') and os.path.isfile(args.model+'_weights.h5'):
		# load json and create model
		model=loadUnetModel(args.model)

	else:
		# do training first
		# help the manual startup script import:
		try:
			from .unet.data import trainGenerator
		except ImportError as e:
			try:
				from unet.data import trainGenerator
			except Exception as e:
				print(e)
				exit()
		
		data_gen_args = dict(rotation_range=0.2,
							width_shift_range=0.05,
							height_shift_range=0.05,
							shear_range=0.05,
							zoom_range=0.05,
							horizontal_flip=True,
							fill_mode='nearest')
		myGene = trainGenerator(args.batch,args.train,'images','unet_masks',data_gen_args,save_to_dir = None,target_size = (256,256),image_color_mode = "rgb")

		model = unet(input_size = (256,256,3))
		model_checkpoint = ModelCheckpoint(args.model+'.hdf5', monitor='loss',verbose=1, save_best_only=True)
		model.fit_generator(myGene,steps_per_epoch=args.steps,epochs=args.epochs,callbacks=[model_checkpoint])

		# save model as json
		model_json = model.to_json()
		with open(args.model+'.json', 'w') as f:
			f.write(model_json)

		# save weights too
		model.save_weights(args.model+'_weights.h5')

	return model


def predictUnet(model,imgPath,target_size=(256,256),flag_multi_class=False):
	if isinstance(imgPath,str):
		img = io.imread(imgPath,as_gray = False) #as_gray = True #as_gray = as_gray
	elif isinstance(imgPath,np.ndarray):
		# already an image matrix
		img=imgPath
		pass
	else:
		# this should never happen
		print('Input is neither a path nor an image')
		return None

	'''
	if len(img.shape)==3:
		# still RGB, convert
		img=skimage.color.rgb2gray(img)
	'''
	if len(img.shape)==3:
		img=img[:,:,0:3]
	elif len(img.shape)==2:
		tmp=np.zeros((img.shape[0],img.shape[1],3),img.dtype)
		for k in range(2):
			tmp[:,:,k]=img
		img=tmp

	print(img.shape)
	img = img / 255
	orig_size=img.shape
	img = trans.resize(img,target_size)
	img = np.reshape(img,img.shape+(1,)) if ((not flag_multi_class) and len(img.shape)==2) else img
	img = np.reshape(img,(1,)+img.shape)

	# predict
	pred_image=model.predict(img,batch_size=1,verbose=1) # batch_size=1
	pred_image2write=pred_image[0,:,:,0]
	pred_image2write=cv2.resize(pred_image2write,(orig_size[1],orig_size[0]),interpolation = cv2.INTER_CUBIC)

	return pred_image2write


def predictNSaveImage(model,test_path,out_path,target_size = (256,256),flag_multi_class = False,num_class = 2,write_images=True):
	imageFiles = [f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path,f))]
	imcount = len(imageFiles)
	print('Found ',imcount,' files in ',test_path)

	os.makedirs(name=out_path, exist_ok=True)

	if not write_images:
		predictions=[]

	for index, imageFile in enumerate(imageFiles):
		# read image
		print("Image:", str(index + 1), "/", str(imcount), "(", imageFile, ")")
		
		pred_image2write=predictUnet(model,os.path.join(test_path,imageFile),target_size)

		if write_images:
			# write output image
			io.imsave(os.path.join(out_path,imageFile),pred_image2write)
		else:
			predictions.append(pred_image2write)

	if not write_images:
		return predictions
	else:
		return None


def predictUnetCustomSize(model,imgPath,target_size=(256,256),flag_multi_class=False):
	if isinstance(imgPath,str):
		img = io.imread(imgPath,as_gray = False) #as_gray = True #as_gray = as_gray
	elif isinstance(imgPath,np.ndarray):
		# already an image matrix
		img=imgPath
		pass
	else:
		# this should never happen
		print('Input is neither a path nor an image')
		return None

	if len(img.shape)==3:
		img=img[:,:,0:3]
	elif len(img.shape)==2:
		tmp=np.zeros((img.shape[0],img.shape[1],3),img.dtype)
		for k in range(2):
			tmp[:,:,k]=img
		img=tmp

	print(img.shape)
	img = img / 255
	orig_size=img.shape

	s=orig_size[0:2]
	if s[0]>256 or s[1]>256:
		if s[0]>=512 or s[1]>=512:
			if s[0]>=1024 or s[1]>=1024:
				if s[0]<2048 or s[1]<2048:
					img=trans.resize(img,(int(s[0]/2),int(s[1]/2)))
					s=img.shape
					s=s[0:2]
				else:
					# just resize to 1024x1024
					maxdim=1024
					img=trans.resize(img,(maxdim,maxdim))
					s=img.shape
					s=s[0:2]
			# crop it
			maxdim = min(s)
			temp = maxdim / 2 ** 8
			if temp != int(temp):
				maxdim = (int(temp) + 1) * 2 ** 8
				temp=int(temp) + 1
			img=trans.resize(img,(maxdim,maxdim))
			pred_image2write=np.zeros((maxdim,maxdim),dtype='float32')
			crops=[]
			for i in range(int(temp)):
				for j in range(int(temp)):
					cropped=img[(i)*256:(i+1)*256,(j)*256:(j+1)*256,:]
					#crops.append(cropped)
					pred_cropped=calcPred(model,cropped,flag_multi_class)
					# stitch
					pred_image2write[(i)*256:(i+1)*256,(j)*256:(j+1)*256]=pred_cropped
			'''
			pred_cropped=calcPredCrops(model,crops,flag_multi_class)
			for i in range(int(temp)):
				for j in range(int(temp)):
					# stitch
					pred_image2write[(i)*256:(i+1)*256,(j)*256:(j+1)*256]=pred_cropped[(i)*temp+j]
			'''
		else:
			# just resize to 256x256
			maxdim=256
			img=trans.resize(img,(maxdim,maxdim))

			pred_image2write=calcPred(model,img,flag_multi_class)
	else:
		# small image, just resize (upscale) to 256x256
		img = trans.resize(img,(256,256))
		pred_image2write=calcPred(model,img,flag_multi_class)

	#print('pred done')
	pred_image2write=cv2.resize(pred_image2write,(orig_size[1],orig_size[0]),interpolation = cv2.INTER_CUBIC)
	#pred_image2write=predictUnet(model,img,target_size)

	return pred_image2write


def calcPred(model,img,flag_multi_class):
	img = np.reshape(img,img.shape+(1,)) if ((not flag_multi_class) and len(img.shape)==2) else img
	img = np.reshape(img,(1,)+img.shape)

	# predict
	pred_image=model.predict(img,batch_size=1,verbose=0) # batch_size=1,verbose=1
	pred_image2write=pred_image[0,:,:,0]
	return pred_image2write


def calcPredCrops(model,img,flag_multi_class):
	if isinstance(img,list) and len(img)>1:
		for idx,val in enumerate(img):
			tmp=img[idx]
			tmp = np.reshape(tmp,tmp.shape+(1,)) if ((not flag_multi_class) and len(tmp.shape)==2) else tmp
			tmp = np.reshape(tmp,(1,)+tmp.shape)
			img[idx]=tmp
		bsize=len(img)
		preds=[]
		for el in img:
			pred_image=model.predict(el,batch_size=1,verbose=1)
			preds.append(pred_image[0,:,:,0])
		return preds
	else:
		img = np.reshape(img,img.shape+(1,)) if ((not flag_multi_class) and len(img.shape)==2) else img
		img = np.reshape(img,(1,)+img.shape)
		bsize=1

	# predict
	pred_image=model.predict(img,batch_size=bsize,verbose=1) # batch_size=1
	pred_image2write=pred_image[0,:,:,0]
	return pred_image2write


def predictNSaveImageCrop(model,test_path,out_path,target_size = (256,256),flag_multi_class = False,num_class = 2,write_images=True):
	imageFiles = [f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path,f))]
	imcount = len(imageFiles)
	print('Found ',imcount,' files in ',test_path)

	os.makedirs(name=out_path, exist_ok=True)

	if not write_images:
		predictions=[]

	for index, imageFile in enumerate(imageFiles):
		# read image
		print("Image:", str(index + 1), "/", str(imcount), "(", imageFile, ")")
		
		pred_image2write=predictUnetCustomSize(model,os.path.join(test_path,imageFile),target_size=target_size,flag_multi_class=flag_multi_class)

		if write_images:
			# write output image
			io.imsave(os.path.join(out_path,imageFile),pred_image2write)
		else:
			predictions.append(pred_image2write)

	if not write_images:
		return predictions
	else:
		return None


def predict(model,args):
	# start prediction either after training or model loading
	preds=predictNSaveImage(model,args.test,args.results,write_images=args.write) #,target_size=(args.size,args.size))


def predictCustomSize(args):
	# start prediction either after training or model loading
	model=unet(pretrained_weights=None,input_size=(args.size,args.size,3))
	print(f'size: {args.size}')
	model.load_weights(args.model+'_weights.h5')
	preds=predictNSaveImage(model,args.test,args.results,write_images=args.write,target_size=(args.size,args.size))


def predictCustomSizeCrop(model,args):
	# start prediction either after training or model loading
	preds=predictNSaveImageCrop(model,args.test,args.results,write_images=args.write,target_size=(args.size,args.size))


def setGpu(gpuSetting='0'):
	if gpuSetting is not None:
		import tensorflow as tf
		if gpuSetting=='cpu':
			tf.config.set_visible_devices([], 'GPU')
			print('Using CPU for U-Net predictions')
		else:
			# check if there is a gpu at all
			g=tf.config.list_physical_devices('GPU')
			if len(g)>int(gpuSetting):
				tf.config.set_visible_devices(g[int(gpuSetting)], 'GPU')
				print(f'Using GPU {gpuSetting} for U-Net predictions')
			else:
				tf.config.set_visible_devices([], 'GPU')
				print(f'The selected GPU device ({gpuSetting}) is not available, using CPU for U-Net predictions')


def callPredictUnet(modelName,imageName,gpuSetting='0'):
	model=loadUnetModel(modelName)
	setGpu(gpuSetting)
	pred=predictUnet(model,imageName)
	return pred


def callPredictUnetLoaded(model,imageName,gpuSetting='0'):
	setGpu(gpuSetting)
	pred=predictUnet(model,imageName)
	return pred


def callPredictUnetLoadedNoset(model,imageName,target_size=(256,256)):
	pred=predictUnet(model,imageName)
	return pred


def callPredictUnetLoadedNosetCustomSize(model,imageName):
	pred=predictUnetCustomSize(model,imageName)
	return pred


def estimateImgSize(self,img):
	resSize=(256,256)
	if img is not None:
		s=img.shape
		maxdim = max(s)
		temp = maxdim / 2 ** 6
		if temp != int(temp):
			maxdim = (int(temp) + 1) * 2 ** 6
		resSize=(maxdim,maxdim)
	else:
		#debug:
		print(f'img is None')
	print(f'set size: ({resSize})')
	return resSize


def main():
	args=importArgs()
	
	setGpu(args.gpu)
	
	if args.size==256:
		model=trainIfNoModel(args)
		preds=predict(model,args)
	else:
		#preds=predictCustomSize(args)
		model=trainIfNoModel(args)
		preds=predictCustomSizeCrop(model,args)
	print('Finished predicting images')

	return preds


if __name__ == '__main__':
	main()