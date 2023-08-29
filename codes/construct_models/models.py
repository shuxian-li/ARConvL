import tensorflow as tf
from construct_models.prototype_unit import PRUnit
from construct_models.CNN_nets import mnist_CNN, Resnet


def get_features(data_name, feature_dim):
    if data_name == 'mnist' or data_name == 'fashion_mnist':
        x_in = tf.keras.layers.Input(shape=(28, 28, 1))
        features = mnist_CNN(x_in, feature_dim)
    else:
        resnet = Resnet()
        model = resnet.get_model()
        x_in = model.input
        features = model.output

    return x_in, features


def rCPL(data_name, feature_dim, class_num, center_num=1):
    x_in, features = get_features(data_name, feature_dim)
    [prototypes, dist, region_esti] = PRUnit(class_num, center_num)(features)

    model = tf.keras.Model(inputs=x_in, outputs=[features, prototypes, dist, region_esti])

    return model
