import cv2
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from utils import crop_img, get_resized_bbox
import tensorflow_addons as tfa

class DataAugmentator:
    def __init__(self, img_height, img_width, img_channels, heatmap_channels, batch_size, buffer_size, obj_index):
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        self.heatmap_channels = heatmap_channels
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.obj_index = obj_index
 
    
    def getBbox(self, imageName):
        mask = np.array(ImageOps.grayscale(Image.open(imageName)))
        mask[mask > 10] = 255
        mask[mask <= 10] = 0

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            xmin = x
            xmax = x + w
            ymin = y
            ymax = y + h
            return (xmin, xmax, ymin, ymax)

    
    def getBbox2(self, mask):
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            xmin = x
            xmax = x + w
            ymin = y
            ymax = y + h
            return (xmin, xmax, ymin, ymax)
        
        return (0, mask.shape[0], 0, mask.shape[1])
    
    
    def check_if_point_in_mask(self, point, mask, distance=1):
        for y in range(point[0] - distance, point[0] + distance):
            for x in range(point[1] - distance, point[1] + distance):
                if x >= mask.shape[0] or y >= mask.shape[1]:
                    continue
                
                if mask[x, y] > 0:
                    return True
        return False
    
    
    def move_keypoints_coords(self, keypoints, bbox, mask, hide_occluded=True):
        new_keypoints = []
        
        for x, y in keypoints:
            new_x = (0 if x < bbox[0] or x > bbox[1] else x - bbox[0])/(abs(bbox[0] - bbox[1]))
            new_y = (0 if y < bbox[2] or y > bbox[3] else y - bbox[2])/(abs(bbox[2] - bbox[3]))
            tmp_keypoints = [new_x, new_y] if new_x != 0 and new_y != 0 else [0, 0]
            scaled_keypoints = [int(tmp_keypoints[0] * mask.shape[0]), int(tmp_keypoints[1] * mask.shape[1])]
            tmp_keypoints = tmp_keypoints if (not hide_occluded or self.check_if_point_in_mask(scaled_keypoints, mask)) else [0, 0]
            new_keypoints.append(tmp_keypoints)
        
        return new_keypoints
    
    
    def getBboxProb(self, label_path, img_shape, prob=0.9):
        return (self.getBbox(label_path) 
            if np.random.random() < prob 
            else (0, img_shape[0], 0, img_shape[1]))


    def getBboxProb2(self, mask, img_shape, prob=0.9):
        return (self.getBbox2(mask) 
            if np.random.random() < prob 
            else (0, img_shape[0], 0, img_shape[1]))
        
    
    def getMaskedImgProb(self, img, mask, prob=0.05):
        if np.random.random() < prob:
            img = img.copy()
            img[mask == 0] = (0, 0, 0)
            background = np.random.random(img.shape)
            background[mask != 0] = (.0, .0, .0)
            img += background
        return img
    
    
    def gaussian_k(self, x0, y0, sigma, width, height):
        """ Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
        """
        x = np.arange(0, width, 1, float) ## (width,)
        y = np.arange(0, height, 1, float)[:, np.newaxis] ## (height,1)
        return (np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2)) 
                if x0 != 0 and y0 != 0 
                else np.zeros((width, height), dtype=np.float32))


    def generate_hm(self, height, width, landmarks, s=3):
        """ Generate a full Heap Map for every landmarks in an array
        Args:
            height    : The height of Heat Map (the height of target output)
            width     : The width  of Heat Map (the width of target output)
            joints    : [(x1,y1),(x2,y2)...] containing landmarks
            maxlenght : Lenght of the Bounding Box
        """
        Nlandmarks = len(landmarks)
        hm = np.zeros((height, width, Nlandmarks), dtype = np.float32)
        for i in range(Nlandmarks):
            if not np.array_equal(landmarks[i], [-1,-1]):
                hm[:,:,i] = self.gaussian_k(
                    landmarks[i][0] * width,
                    landmarks[i][1] * height,
                    s,
                    width, 
                    height)
            else:
                hm[:,:,i] = np.zeros((height,width))
        return hm
    
    
    def generate_pair(self, dataset):
        for img_path, label_path, keypoints, pose in dataset:
            img = np.array(Image.open(img_path))[:, :, :3] / 255.0
            mask = np.array(ImageOps.grayscale(Image.open(label_path)))
            # img = getMaskedImgProb(img, mask)

            bbox = self.getBboxProb(label_path, img.shape)
            bbox = get_resized_bbox(bbox, img.shape)

            img = crop_img(img, bbox, self.img_height, self.img_width)
            mask = crop_img(mask, bbox, self.img_height, self.img_width)

            keypoints = self.move_keypoints_coords(keypoints, bbox, mask)
            heat_maps = self.generate_hm(self.img_width, self.img_height, keypoints)
                
            yield (img, pose[:4]), heat_maps

    
    def generate_pair2(self, dataset, boundary_factor=None):
        for img_path, mask, keypoints, quaternion, translation in dataset:
            img = np.array(Image.open(img_path))[:, :, :3] / 255.0
            mask = self.read_mask(mask)
            # img = self.getMaskedImgProb(img, mask)

            bbox = self.getBboxProb2(mask, img.shape)
            bbox = get_resized_bbox(bbox, img.shape, boundary_factor)

            img = crop_img(img, bbox, self.img_height, self.img_width)
            mask = crop_img(mask, bbox, self.img_height, self.img_width)

            keypoints = self.move_keypoints_coords(keypoints, bbox, mask)
            heat_maps = self.generate_hm(self.img_width, self.img_height, keypoints)
                
            yield (img, quaternion), heat_maps
            
            
    def generate_pair_eval(self, dataset):
        for img_path, label_path, keypoints, pose in dataset:
            img_org = np.array(Image.open(img_path)) / 255.0
            yield img_org, pose, keypoints,         
            
    
    def generate_pair_eval2(self, dataset):
        for img_path, mask, keypoints, quaternion, translation in dataset:
            img_org = np.array(Image.open(img_path)) / 255.0
            yield img_org, quaternion, translation, keypoints


    def get_tf_dataset(self, dataset, boundary_factor=None):
        return tf.data.Dataset.from_generator(
            generator=lambda: self.generate_pair2(dataset, boundary_factor),
            output_signature=(
                (
                    tf.TensorSpec(shape=(self.img_height, self.img_width, self.img_channels), dtype=tf.float32, name='image'),
                    tf.TensorSpec(shape=(4, ), dtype=tf.float32, name='pose')
                ),
                tf.TensorSpec(shape=(self.img_height, self.img_width, self.heatmap_channels), dtype=tf.float32, name='heat_maps')
            )
        )
    
    
    def get_train_batches(self, train_dataset, boundary_factor=None):
        return (self.get_tf_dataset(train_dataset, boundary_factor)
            .shuffle(self.buffer_size)
            .batch(self.batch_size)
            .repeat()
            .map(Augment())
            .prefetch(buffer_size=tf.data.AUTOTUNE))
    
    
    def get_val_batches(self, val_dataset, boundary_factor=None):
        return self.get_tf_dataset(val_dataset, boundary_factor).batch(self.batch_size)

    
    def read_mask(self, path):
        img = Image.open(path)
        img = np.asarray(img) * 255
        img[img != self.obj_index] = 0
        img[img == self.obj_index] = 1
        return np.array(img, dtype=np.uint8)
        
        
class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
    
    def call(self, inputs, labels):
        seed = np.random.random(2)
        poses = self.randomize_poses(inputs[1])
        images = tf.image.stateless_random_brightness(inputs[0], 0.5, seed)
        images = tf.image.stateless_random_contrast(images, lower=0.1, upper=0.9, seed=seed)

        if np.random.random() > 0.8:
            images = tfa.image.gaussian_filter2d(images, filter_shape = 3 + int(np.random.random() * 3), sigma = 1 + int(np.random.random() * 3))

        if np.random.random() > 0.8:
            images = tf.image.rgb_to_grayscale(images)
            images = tf.image.grayscale_to_rgb(images)

        return (inputs[0], poses), labels
    
    def randomize_poses(self, poses):
        return poses + tf.random.uniform(shape=tf.shape(poses)) * 0.1