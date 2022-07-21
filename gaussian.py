import tensorflow as tf

class Gaussian(tf.keras.layers.Layer):
    def __init__(self, output_shape):
        super(Gaussian, self).__init__()
        self.output_shape2 = output_shape

    def build(self, input_shape):
        self.X = self.get_variable(
            range = self.output_shape2[0],
            repeat = input_shape[2],
            name='x_array'
        )

        self.Y = self.get_variable(
            range = self.output_shape2[1],
            repeat = input_shape[2],
            name='y_array',
            expand=True
        )

        self.input_shape2 = input_shape

    def get_variable(self, range, repeat, name, expand=False):
        value = tf.transpose(
            tf.repeat(
                tf.range(0, range, 1, dtype=tf.float32)[tf.newaxis, :],
                repeat,
                axis=0))
        
        value = value[:, tf.newaxis] if expand else value

        return tf.Variable(
            initial_value=value,
            trainable=False,
            name=name
        )


    def get_config(self):
        return super().get_config()


    def call(self, inputs):
        arr1 = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for batch in inputs:
            X0 = batch[0] * self.output_shape2[0] 
            Y0 = batch[1] * self.output_shape2[1]
            S = 1
            heatmaps = tf.math.exp(-((self.X-X0)**2 + (self.Y-Y0)**2) / (2*S**2))
            arr1 = arr1.write(arr1.size(), heatmaps)
        return arr1.stack()