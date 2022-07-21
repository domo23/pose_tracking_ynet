from read_data import get_data_list, get_object_index, get_kp3d, split_dataset, CLASSES
from data_augmentator import DataAugmentator
from y_net import YNet

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import argparse

# defaults
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
HM_CHANNELS = 9
BATCH_SIZE = 8
EPOCHS = 15
BUFFER_SIZE = 256
BOUNDARY_FACTOR = 0.3
OBJECT_NAME = '035_power_drill'
VALIDATION_SCENES = ['0050']
YCB_DS_PATH = 'C:/\\Users\\domo2\\Downloads\\YCB_Video_Dataset'
camera_matrix = np.array([
    [1.066778e+03, 0.000000e+00, 3.129869e+02],
    [0.000000e+00, 1.067487e+03, 2.413109e+02],
    [0.000000e+00, 0.000000e+00, 1.000000e+00]])

# argument parser definition
parser = argparse.ArgumentParser(
    description='Train YNet network on YCB-Video dataset.')
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                    help=f'batch size for training network, default {BATCH_SIZE}')
parser.add_argument('--epochs', type=int, default=EPOCHS,
                    help=f'amount of epochs for training network, default {EPOCHS}')
parser.add_argument('--object_name', default=OBJECT_NAME,
                    help=f'name of object for training, default {OBJECT_NAME}')
parser.add_argument('--ycb_ds_path', default=YCB_DS_PATH,
                    help=f'absolute path to YCB-Video dataset, default {YCB_DS_PATH}')
parser.add_argument('--validation_scenes', default=VALIDATION_SCENES, nargs='+',
                    help=f'validation scenes for training, default {VALIDATION_SCENES}')
args = parser.parse_args()


assert args.object_name in CLASSES, f'Object name must be one of {CLASSES}'

obj_index = get_object_index(args.object_name)
keypoints_3d = get_kp3d(obj_index)
assert keypoints_3d is not None, f'No 3d keypoints defined in keypoints.py for object {args.object_name}'

# Load data - this might take few minutes
dataset_array = get_data_list(obj_index, args.ycb_ds_path)
train_dataset, val_dataset = split_dataset(args.validation_scenes, dataset_array)

data_augmentator = DataAugmentator(
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH,
    img_channels=IMG_CHANNELS,
    heatmap_channels=HM_CHANNELS,
    batch_size=args.batch_size,
    buffer_size=BUFFER_SIZE,
    obj_index=obj_index
)

train_batches = data_augmentator.get_train_batches(train_dataset)
val_batches = data_augmentator.get_val_batches(val_dataset)

ynet = YNet(
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH,
    img_channels=IMG_CHANNELS,
    hm_channels=HM_CHANNELS,
    keypoints_3d=keypoints_3d,
    camera_matrix=camera_matrix
)

callbacks = ynet.get_callbacks()
callbacks.append(tf.keras.callbacks.TensorBoard(
    log_dir='tensorboard', histogram_freq=1))


# Train network
VAL_SUBSPLITS = 5
VALIDATION_STEPS = len(val_dataset)//args.batch_size//VAL_SUBSPLITS
TRAIN_LENGTH = len(train_dataset)
STEPS_PER_EPOCH = TRAIN_LENGTH // args.batch_size

model_history = ynet.model.fit(
    train_batches,
    epochs=args.epochs,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    validation_data=val_batches,
    callbacks=callbacks)


# Plot history
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
plt.figure()
plt.plot(model_history.epoch, loss, 'r', label='Training loss')
plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.legend()
plt.savefig('loss_plot.png')
