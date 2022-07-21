from read_data import get_data_list_eval, get_object_index, get_kp3d, get_model_3d, CLASSES
from data_augmentator import DataAugmentator
from y_net import YNet
from pose_tracker import PoseTracker
from utils import get_rmatrix_tvec_from_gt, get_pose_dict, add_error, project_to_2d
from canvas import Canvas

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import argparse
from tqdm import tqdm

# defaults
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
HM_CHANNELS = 9
BATCH_SIZE = 1
EPOCHS = 15
BUFFER_SIZE = 256
BOUNDARY_FACTOR = 0.3
OBJECT_NAME = '035_power_drill'
VALIDATION_SCENES = ['0050']
YCB_DS_PATH = 'C:/\\Users\\domo2\\Downloads\\YCB_Video_Dataset'
WEIGHTS_PATH = 'best_weights.hdf5'
camera_matrix = np.array([
    [1.066778e+03, 0.000000e+00, 3.129869e+02],
    [0.000000e+00, 1.067487e+03, 2.413109e+02],
    [0.000000e+00, 0.000000e+00, 1.000000e+00]])


# argument parser definition
parser = argparse.ArgumentParser(
    description='Evaluate YNet network on YCB-Video dataset.')
parser.add_argument('--object_name', default=OBJECT_NAME,
                    help=f'name of object for training, default {OBJECT_NAME}')
parser.add_argument('--ycb_ds_path', default=YCB_DS_PATH,
                    help=f'absolute path to YCB-Video dataset, default {YCB_DS_PATH}')
parser.add_argument('--validation_scenes', default=VALIDATION_SCENES, nargs='+',
                    help=f'validation scenes, default {VALIDATION_SCENES}')
parser.add_argument('--weights_path', default=WEIGHTS_PATH,
                    help=f'path with trained weights, default {WEIGHTS_PATH}')
args = parser.parse_args()


assert args.object_name in CLASSES, f'Object name must be one of {CLASSES}'

obj_index = get_object_index(args.object_name)
keypoints_3d = get_kp3d(obj_index)
model_3d = get_model_3d(args.ycb_ds_path, args.object_name)
assert keypoints_3d is not None, f'No 3d keypoints defined in keypoints.py for object {args.object_name}'

val_dataset = get_data_list_eval(
    obj_index, args.ycb_ds_path, args.validation_scenes)

data_augmentator = DataAugmentator(
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH,
    img_channels=IMG_CHANNELS,
    heatmap_channels=HM_CHANNELS,
    batch_size=BATCH_SIZE,
    buffer_size=BUFFER_SIZE,
    obj_index=obj_index
)

ynet = YNet(
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH,
    img_channels=IMG_CHANNELS,
    hm_channels=HM_CHANNELS,
    keypoints_3d=keypoints_3d,
    camera_matrix=camera_matrix
)

ynet.model.load_weights(args.weights_path)

tracker = PoseTracker(
    ynet=ynet,
    model_3d=model_3d,
    keypoints_3d=keypoints_3d,
    camera_matrix=camera_matrix
)

pose_prediction_eval = []
init_pose = True
print('Predicting pose for evaluation dataset...')
with tqdm(total=len(val_dataset), leave=False) as pbar:
    for img_org, quaternion, translation, kp_coords in data_augmentator.generate_pair_eval2(val_dataset):
        pose = np.asarray([*quaternion, *translation])
        if init_pose is True:
            tracker.init_pose(pose)
            init_pose = False
        r_matrix_gt, t_vec_gt = get_rmatrix_tvec_from_gt(pose)
        r_matrix, t_vec = tracker.predict(img_org)
        pose_prediction_eval.append((r_matrix_gt, t_vec_gt, r_matrix, t_vec))
        pbar.update(1)


add_errors = [
    add_error(get_pose_dict(r_matrix, t_vec),
              get_pose_dict(r_matrix_gt, t_vec_gt), model_3d)
    for (r_matrix_gt, t_vec_gt, r_matrix, t_vec)
    in pose_prediction_eval
]

px_err = [
    np.sqrt(np.sum(np.power(project_to_2d(r_matrix, t_vec, model_3d, camera_matrix) -
            project_to_2d(r_matrix_gt, t_vec_gt, model_3d, camera_matrix), 2), axis=0)).mean()
    for (r_matrix_gt, t_vec_gt, r_matrix, t_vec)
    in pose_prediction_eval
]


## 5px error
treshold = 5
plt.plot(px_err, label='Mean distance')
plt.plot([treshold for i in range(len(px_err))],
         label=f'{treshold} px treshold')
plt.ylim([0, 10])
plt.legend()
plt.title(
    f'PX mean squared distance for {args.object_name} - {(np.asarray(px_err) < treshold).sum() / len(px_err) * 100:.2f}%')
plt.savefig('5px_error.png')


## ADD error
treshold = 0.025
plt.plot(add_errors, label=f'mean={np.mean(add_errors): .3f}')
plt.plot([treshold for i in range(len(add_errors))],
         label=f'{treshold} treshold')
plt.ylim(0, 0.1)
plt.title(
    f'ADD metric evaluation for {args.object_name} - {(np.asarray(add_errors) < treshold).sum() / len(add_errors) * 100:.2f}%')
plt.legend()
plt.savefig('add_error.png')


## Sample prediction
img_org, quaternion, translation, keypoints = next(x for i,x in enumerate(data_augmentator.generate_pair_eval2(val_dataset)) if i == 4)
pose = np.asarray([*quaternion, *translation])
tracker.init_pose(pose)
r_matrix, t_vec = tracker.predict(img_org)
d = tracker.meta
canvas = Canvas(IMG_WIDTH, IMG_HEIGHT, HM_CHANNELS)
fig = canvas.get_fig(d['img'], d['pose'], d['pred_keypoints'])
plt.savefig('sample_prediction.png')