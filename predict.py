import os
import numpy as np

from skimage.io import imsave
from keras import backend as K
from preprocessing.data import load_train_data, load_test_data
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

from preprocessing.data import preprocess
from preprocessing.data import normalize


def predict(data_path,model_str,ckpt_path_=''):

    #add models here:
    if model_str=='unet':from designs import unet as design
    if model_str=='unet64filters':from designs import unet64filters as design
    if model_str=='flatunet':from designs import flatunet as design
    if model_str=='unet64batchnorm':from designs import unet64batchnorm as design

    imgs_train, imgs_mask_train = load_train_data(data_path)

    print('Creating and compiling model...')
    model = design.build()
   
    print('Loading and preprocessing test data...')
    imgs_test, imgs_id_test = load_test_data(data_path)
    imgs_test = preprocess(imgs_test)
    imgs_test = normalize(imgs_test)

    if ckpt_path_=='':ckpt_path = os.path.join(data_path, 'internal/checkpoints')
    else: ckpt_path=ckpt_path_  
    ckpt_file = os.path.join(ckpt_path, 'weights_'+model_str+'.h5')
    model.load_weights(ckpt_file)
    print('Loading saved weights :',ckpt_file)

    print('Predicting masks on test data...')
    imgs_mask_test = model.predict(imgs_test, verbose=1)

    out_path = os.path.join(data_path, 'output')
    out_file = os.path.join(data_path, 'internal/npy/imgs_mask_test.npy')
    if ckpt_path_!='': 
        if not os.path.exists(ckpt_path+'/predictions/'+model_str+'/'): os.mkdir(ckpt_path+'/predictions/'+model_str+'/')
        out_path = ckpt_path+'/predictions/'
        out_file = ckpt_path+'/predictions/'+model_str+'/imgs_mask_test.npy'

    np.save(out_file, imgs_mask_test)

    print('Saving predicted masks to files to:',out_path+'/'+model_str)
    if not os.path.exists(out_path+'/'+model_str):
        os.mkdir(out_path+'/'+model_str)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(out_path+'/'+model_str, '{0:0>5}_pred.png'.format(image_id)), image)


if __name__ == '__main__':
    data_path = '/media/data/180505_v1'
    predict(data_path)
