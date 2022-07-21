import os
import regex
import scipy.io
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from keypoints import kp3d
import random
from tqdm import tqdm

CLASSES = [
    '002_master_chef_can',
    '003_cracker_box',
    '004_sugar_box',
    '005_tomato_soup_can',
    '006_mustard_bottle',
    '007_tuna_fish_can',
    '008_pudding_box',
    '009_gelatin_box',
    '010_potted_meat_can',
    '011_banana',
    '019_pitcher_base',
    '021_bleach_cleanser',
    '024_bowl',
    '025_mug',
    '035_power_drill',
    '036_wood_block',
    '037_scissors',
    '040_large_marker',
    '051_large_clamp',
    '052_extra_large_clamp',
    '061_foam_brick'
]


def get_object_index(object_name):
    """
    Returns index of object element in classes list (from image_sets/classes.txt). Index is incremented by 1, since original script was written in matlab.
    """
    return CLASSES.index(object_name) + 1


def get_kp3d(obj_index):
    return kp3d.get(obj_index)


def get_obj_frame_ids(obj_index, ycb_ds_path, scenes=None):
    """
    Parse *-box.txt files in data directory, and search for given object.
    """

    bbox_r = regex.compile(r'(\d{6})-box\.txt')

    obj_frames = []
    obj_name = CLASSES[obj_index-1]

    print('Scanning YCB data directory for object frames...')

    directories = scenes if scenes is not None else os.listdir(
        os.path.join(ycb_ds_path, 'data'))

    for directory_idx, directory in enumerate(directories):
        files = os.listdir(os.path.join(ycb_ds_path, 'data', directory))
        bbox_files = [*filter(lambda x: bbox_r.match(x), files)]
        with tqdm(total=len(bbox_files), leave=False) as pbar:
            for bbox_file in bbox_files:
                pbar.set_description(
                    f'Scanning scene {directory}, {directory_idx}/{len(directories)}')
                pbar.update(1)
                full_name = os.path.join(
                    ycb_ds_path, 'data', directory, bbox_file)
                with open(full_name) as f:
                    if obj_name in f.read():
                        idx = bbox_r.match(bbox_file).groups()[0]
                        obj_frames.append(os.path.join(
                            ycb_ds_path, 'data', directory, idx))

    return obj_frames


def get_obj_frame_synth_ids(obj_index, ycb_ds_path):
    """
    Parse *-meta.mat files in data_syn directory, and search for given object.
    """

    mat_r = regex.compile(r'(\d{6})-meta\.mat')
    files = os.listdir(os.path.join(ycb_ds_path, 'data_syn'))
    mat_files = [*filter(lambda x: mat_r.match(x), files)]
    indexes = [mat_r.match(file).groups()[0] for file in mat_files]
    obj_frames = []

    print('Scanning YCB data_syn directory for object frames...')
    with tqdm(total=len(indexes), leave=False) as pbar:
        for idx in indexes:
            try:
                full_mat_name = os.path.join(
                    ycb_ds_path, 'data_syn', f'{idx}-meta.mat')
                mat = scipy.io.loadmat(full_mat_name)
                obj_data = np.where(mat['cls_indexes'].flatten() == obj_index)

                if not len(obj_data[0]):
                    continue

                obj_frames.append(os.path.join(ycb_ds_path, 'data_syn', idx))

            except:
                print(f'error on parsing id {idx}')
            finally:
                pbar.update(1)

    return obj_frames


def get_keypoints_2d(mat, obj_idx, kp3d):
    Rt = mat['poses'][:, :, obj_idx]
    camera = mat['intrinsic_matrix']
    P = camera @ Rt
    kp2d = P @ np.c_[kp3d, np.ones(len(kp3d))].T
    kp2d[0, :] = kp2d[0, :] / kp2d[2, :]
    kp2d[1, :] = kp2d[1, :] / kp2d[2, :]
    return np.array(kp2d[0:2, :], dtype=np.int16).T


def read_mat_file(obj_index, path):
    mat = scipy.io.loadmat(path)
    mat_obj_idx = np.where(mat['cls_indexes'].flatten() == obj_index)[0][0]
    kp2d = get_keypoints_2d(mat, mat_obj_idx, get_kp3d(obj_index))
    quaternion = Rotation.from_matrix(
        mat['poses'][:, :3, mat_obj_idx]).as_quat()
    translation = mat['poses'][:, 3, mat_obj_idx]
    return kp2d, quaternion, translation


def get_data_list(obj_index, ycb_ds_path):
    frames1 = get_obj_frame_ids(obj_index, ycb_ds_path)
    frames2 = get_obj_frame_synth_ids(obj_index, ycb_ds_path)
    frames = [*frames1, *frames2]
    sufixes = ['-color.png', '-label.png', '-meta.mat']

    data_list = list()
    print('Assembling gathered data for training and validation...')
    with tqdm(total=len(frames), leave=False) as pbar:
        for frame in frames:
            path = frame.split(os.sep)
            color, label, meta = [os.path.join(
                *path[:-1], path[-1] + sufix) for sufix in sufixes]
            kp2d, quaternion, translation = read_mat_file(obj_index, meta)
            data_list.append((color, label, kp2d, quaternion, translation))
            pbar.update(1)

    return data_list


def get_data_list_eval(obj_index, ycb_ds_path, scenes):
    frames = get_obj_frame_ids(obj_index, ycb_ds_path, scenes)
    sufixes = ['-color.png', '-label.png', '-meta.mat']

    data_list = list()
    print('Assembling gathered data for validation...')
    with tqdm(total=len(frames), leave=False) as pbar:
        for frame in frames:
            path = frame.split(os.sep)
            color, label, meta = [os.path.join(
                *path[:-1], path[-1] + sufix) for sufix in sufixes]
            kp2d, quaternion, translation = read_mat_file(obj_index, meta)
            data_list.append((color, label, kp2d, quaternion, translation))
            pbar.update(1)

    return data_list


def split_dataset(validation_scenes, dataset_array):
    def filter_func(x):
        return x[0].split(os.sep)[1] not in validation_scenes

    def filter_func2(x):
        return x[0].split(os.sep)[1] in validation_scenes

    train_dataset = list(filter(filter_func, dataset_array))
    random.shuffle(train_dataset)
    val_dataset = list(filter(filter_func2, dataset_array))

    return train_dataset, val_dataset


def get_model_3d(ycb_ds_path, object_name):
    path = os.path.join(*os.path.split(ycb_ds_path),
                        'models', object_name, 'points.xyz')
    return np.loadtxt(path, delimiter=" ")
