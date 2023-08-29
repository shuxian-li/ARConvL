import tensorflow as tf
from funcs_utility import distance


class PRUnit(tf.keras.layers.Layer):
    def __init__(self, class_num, center_num, **kwargs):
        self.class_num = class_num
        self.center_num = center_num
        super(PRUnit, self).__init__()

    def build(self, input_shape):
        self.weight = self.add_weight(shape=(self.class_num * self.center_num, input_shape[-1]),
                                      initializer="he_uniform", name='Weight',
                                      trainable=True)
        self.axis_esti = self.add_weight(shape=(1, self.class_num * self.center_num),
                                         initializer=tf.constant_initializer(0.1), name='Axis',
                                         trainable=True)

        super(PRUnit, self).build(input_shape)

    def call(self, x):
        dist = distance(x, self.weight)
        region_esti = self.axis_esti ** 2

        return self.weight, dist, region_esti

    def get_config(self):
        config = {"class_num": self.class_num, "center_num": self.center_num}
        base_config = super(PRUnit, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))