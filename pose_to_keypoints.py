import tensorflow_graphics.geometry.transformation as tfg_transformation
import tensorflow as tf

class PoseToKeypoints(tf.keras.layers.Layer):
    def __init__(self, keypoints_3d, camera_matrix):
        super(PoseToKeypoints, self).__init__()
        self.keypoints_3d = tf.Variable(
            initial_value = keypoints_3d.T,
            trainable=False,
            name='keypoints_3d',
            dtype=tf.float32)

        self.camera_matrix = tf.Variable(
            initial_value = camera_matrix,
            trainable=False,
            name='camera_matrix',
            dtype=tf.float32)
            
        self.t = tf.Variable(
            initial_value=tf.constant([0, 0, 0.7])[:, tf.newaxis],
            trainable=False,
            dtype=tf.float32
        )


    def get_config(self):
        return super().get_config()


    def call(self, inputs):
        arr1 = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for pose in inputs:
            R = tfg_transformation.rotation_matrix_3d.from_quaternion(pose)
            camera_coords = R @ self.keypoints_3d + self.t
            img_coords = self.camera_matrix @ camera_coords
            normalized_coords = tf.transpose(img_coords[:2] / img_coords[2]) * tf.constant([1.0/640, 1.0/480])
            arr1 = arr1.write(arr1.size(), tf.transpose(normalized_coords))
        return arr1.stack()