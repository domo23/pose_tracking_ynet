import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf


class Canvas:
    def __init__(self, img_width, img_height, hm_channels):
        self.img_width = img_width
        self.img_height = img_height
        self.hm_channels = hm_channels


    def get_index(self, row: int, col: int, pos: int) -> int:
        return (row - 1) * col + pos


    def show_keypoints(self, tensor, row):
        reshaped = np.reshape(tensor, (self.img_width, self.img_height, self.hm_channels))
        images = [reshaped[:,:,i,tf.newaxis] for i in range(self.hm_channels)]
        for i, image in enumerate(images, 1):
            ax = self.fig.add_subplot(self.rows, self.cols, self.get_index(row, self.cols, i))
            ax.imshow(tf.keras.utils.array_to_img(image))
            ax.set_axis_off()


    def get_fig(self, image, pose, labels, predictions=None):
        self.fig = plt.figure(figsize=(15, 5))
        self.cols = self.hm_channels
        self.rows = 2 if predictions is None else 3
        
        # show image
        ax = self.fig.add_subplot(self.rows, self.cols, 1)
        ax.imshow(tf.keras.utils.array_to_img(image))
        ax.set_axis_off()
        
        # show labels
        self.show_keypoints(labels, 2)
        
        # show predictions
        if predictions is not None:
            self.show_keypoints(predictions, 3)
        
        return self.fig

    
    def display(self, image, pose, labels, predictions=None):
        self.get_fig(image, pose, labels, predictions=None)
        plt.show()