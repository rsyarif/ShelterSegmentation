import os
import numpy as np

from skimage.io import imsave
from skimage.transform import resize

from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint,CSVLogger,EarlyStopping
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

from preprocessing.data import load_train_data
from preprocessing.data import preprocess
from preprocessing.data import normalize
from preprocessing.data import normalize_mask

# MODEL
# every design module should have a least:
# build(): returns the actual model
# preprocess(): reshape input data


def train(data_path,model_str,
          number_of_epochs=2,
          batch_size=10,
          test_data_fraction=0.15,
          checkpoint_period=10,
          load_prev_weights=False,
          early_stop_patience=10):

    #add models here:
    if model_str=='unet':from designs import unet as design
    if model_str=='unet64filters':from designs import unet64filters as design
    if model_str=='flatunet':from designs import flatunet as design
    if model_str=='unet64batchnorm':from designs import unet64batchnorm as design

    # DATA LOADING AND PREPROCESSING
    print('Loading and preprocessing train data...')

    # load input images
    imgs_train, imgs_mask_train = load_train_data(data_path)

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = normalize(imgs_train)
    imgs_mask_train = normalize_mask(imgs_mask_train)

    # BUILD MODEL
    print('Creating and compiling model...')
    model = design.build()
    #print layout of model:
    #model.summary()

    # set up saving weights at checkpoints,
    if not os.path.exists(data_path+'/internal/checkpoints'): os.makedirs(data_path+'/internal/checkpoints')
    ckpt_dir = os.path.join(data_path, 'internal/checkpoints')
    ckpt_file = os.path.join(ckpt_dir, 'weights_'+model_str+'_epoch{epoch:02d}.h5')
    model_checkpoint = ModelCheckpoint(ckpt_file,
                                       monitor='val_loss',
                                       save_best_only=True,
                                       save_weights_only=True,
                                       period=checkpoint_period)

    # save epoch logs to txt
    CSV_LOG_FILENAME = os.path.join(ckpt_dir,'log_'+model_str+'.csv')
    csv_logger = CSVLogger(CSV_LOG_FILENAME)

    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=early_stop_patience, verbose=1)

    if(load_prev_weights): 
      try:
        model.load_weights(data_path+'/internal/prev_checkpoints_to_load/weights_'+model_str+'.h5')
        print('Loading prev weights:',data_path+'/internal/prev_checkpoints_to_load/weights_'+model_str+'.h5')
      except Exception as e:
        print('Problem loading the saved weight!')
        print(e)
        return


    # FIT MODEL
    print('Fitting model...')
    model_out = model.fit(imgs_train, imgs_mask_train,
              batch_size=batch_size,
              epochs=number_of_epochs,
              verbose=1,
              shuffle=True,
              validation_split=test_data_fraction,
              callbacks=[model_checkpoint,csv_logger,early_stop])

    model.save(ckpt_dir+'/weights_'+model_str+'.h5') #save model and final weights.

    return model_out


# What to do when this file is run:
if __name__ == '__main__':
    #data_path = '/media/data/180505_v1/'
    data_path = '/Users/rizki/Documents/Projects/ShelterSegmentation_take2/shelterdata/180505_v1/'

    train(data_path,'unet')


