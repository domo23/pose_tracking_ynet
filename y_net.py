from gc import callbacks
import tensorflow as tf
from keras_pyramid_pooling_module import PyramidPoolingModule
from gaussian import Gaussian
from pose_to_keypoints import PoseToKeypoints


class YNet():
    def __init__(self, img_height, img_width, img_channels, hm_channels, keypoints_3d, camera_matrix):
        self.img_width = img_width
        self.img_height = img_height
        self.img_channels = img_channels
        self.hm_channels = hm_channels
        self.keypoints_3d = keypoints_3d
        self.camera_matrix = camera_matrix
        self.model_weight_path = 'best_weights.hdf5'
        self.model = self.create_model()
        
    
    def get_pose_branch(self, pose_input, filters, data_format):
        x = PoseToKeypoints(self.keypoints_3d, self.camera_matrix)(pose_input)
        x.trainable = False
        x = Gaussian([32, 32])(x)
        x.trainable = False
        x = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name='pose_conv', data_format=data_format)(x)
        return x
    
    def image_feature_module(self, input, size):
        ## Branch 1
        x = tf.keras.layers.SeparableConv2D(size, 3, padding='same')(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.SeparableConv2D(size * 2, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.SeparableConv2D(size, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        ## Branch 2
        y = tf.keras.layers.Conv2D(size, 3, padding='same')(input)
        y = tf.keras.layers.BatchNormalization()(y)
        
        return tf.keras.layers.Add()([x, y])

    
    def upsampling_branch(self, input, idx, processing_size, up_size, data_format):
        x = tf.keras.layers.SeparableConv2D(
            filters=processing_size, 
            kernel_size=(int(self.img_height/4), int(self.img_width/4)),
            activation='relu', 
            padding='same', 
            name=f'bottleneck_1_branch_{idx}', 
            data_format=data_format)(input)

        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.SeparableConv2D(
            filters=processing_size,
            kernel_size=(1, 1), 
            activation='relu', 
            padding='same',
            name=f'bottleneck_2_branch_{idx}', 
            data_format=data_format)(x)

        x = tf.keras.layers.BatchNormalization()(x)

        up1 = tf.keras.layers.Conv2DTranspose(
            filters=up_size, 
            kernel_size=(4, 4), 
            strides=(2, 2), 
            use_bias=False, 
            name=f'upsample_1_branch_{idx}', 
            data_format=data_format, 
            padding='same')(x)
        
        up2 = tf.keras.layers.Conv2DTranspose(
            filters=1, 
            kernel_size=(4, 4), 
            strides=(2, 2), 
            use_bias=False, 
            name=f'upsample_2_branch_{idx}', 
            data_format=data_format, 
            padding='same')(up1)
        
        return up2
    
    def create_model(self):
        IMAGE_ORDERING = "channels_last"
        nfmp_block1 = 128
        nfmp_block2 = 256
        nfmp_block3 = 512
        img_input = tf.keras.layers.Input(shape=(self.img_height, self.img_width, self.img_channels), name='img_input')
        pose_input = tf.keras.layers.Input(shape=(4, ), name='pose_input')

        x = PyramidPoolingModule(20, (3, 3), padding='same')(img_input)

        pose_output = self.get_pose_branch(pose_input, nfmp_block2, IMAGE_ORDERING)

        # Encoder Block 1
        x = tf.keras.layers.Conv2D(nfmp_block1, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.Conv2D(nfmp_block1, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)
        x = self.image_feature_module(x, nfmp_block1)
        block1 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
            
        # Encoder Block 2
        x = tf.keras.layers.Conv2D(nfmp_block2, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(block1)
        x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.Conv2D(nfmp_block2, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)
        x = self.image_feature_module(x, nfmp_block2)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)

        x = tf.keras.layers.Concatenate()([x, pose_output])

        # outputs = [self.upsampling_branch(x, i+1, nfmp_block2, nfmp_block1, IMAGE_ORDERING) for i in range(self.hm_channels)]
        # out = tf.keras.layers.Concatenate()(outputs)

        x = tf.keras.layers.SeparableConv2D(
            filters=nfmp_block3, 
            kernel_size=(int(self.img_height/4), int(self.img_width/4)),
            activation='relu', 
            padding='same', 
            name="bottleneck_1", 
            data_format=IMAGE_ORDERING)(x)

        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.SeparableConv2D(
            filters=nfmp_block3,
            kernel_size=(1, 1), 
            activation='relu', 
            padding='same',
            name="bottleneck_2", 
            data_format=IMAGE_ORDERING)(x)

        x = tf.keras.layers.BatchNormalization()(x)

        up1 = tf.keras.layers.Conv2DTranspose(
            filters=nfmp_block1, 
            kernel_size=(4, 4), 
            strides=(2, 2), 
            use_bias=False, 
            name='upsample_1', 
            data_format=IMAGE_ORDERING, 
            padding='same')(x)
        
        up2 = tf.keras.layers.Conv2DTranspose(
            filters=self.hm_channels, 
            kernel_size=(4, 4), 
            strides=(2, 2), 
            use_bias=False, 
            name='upsample_2', 
            data_format=IMAGE_ORDERING, 
            padding='same')(up1)
        
        model = tf.keras.Model([img_input, pose_input], up2)
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), sample_weight_mode="temporal")
        return model
        
        
    def get_callbacks(self):
        return [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                verbose=1,
                min_delta=1e-5,
                mode='auto'),
            tf.keras.callbacks.ModelCheckpoint(
                monitor='val_loss',
                filepath=self.model_weight_path,
                save_best_only=True,
                save_weights_only=True,
                mode='auto')]