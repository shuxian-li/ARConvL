import tensorflow as tf
import numpy as np
import os
from construct_models.models import rCPL
from data_preprocess.get_image_data import get_data_name_from_id, get_data, oe_data, train_val_split, get_data_info, \
    handle_data, get_random_eraser
from define_train import train
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_classifier(model_id, data_name, feature_dim, class_num):
    if model_id == 0:
        model_name = "ARConvL"
        clf = rCPL(data_name, feature_dim, class_num)
    else:
        raise Exception("Undefined model_id")

    return model_name, clf


def get_save_path(data_name, oe_id, ratio, model_name, feature_dim):
    save_path_base = './results_save'
    if not os.path.exists(save_path_base):
        os.mkdir(save_path_base)
    save_path_base = './results_save/models/'
    if not os.path.exists(save_path_base):
        os.mkdir(save_path_base)
    save_path_data_name = save_path_base + data_name + '/'
    if not os.path.exists(save_path_data_name):
        os.mkdir(save_path_data_name)
    if oe_id is not None:
        save_path_oe = save_path_data_name + '/' + str(oe_id) + '_' + str(ratio) + '/'
        if not os.path.exists(save_path_oe):
            os.mkdir(save_path_oe)
    else:
        save_path_oe = save_path_data_name
    save_path = save_path_oe + model_name + '_' + str(feature_dim) + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    return save_path


def main_step(model_id, data_id, oe_id, imb_ratio, batch_size, repeat):
    for re in range(repeat):
        # set random seed
        # re = re + 3
        random.seed(re)
        np.random.seed(re)
        tf.random.set_seed(re)

        # get data and construct imbalanced data
        print('loading data...')
        data_name = get_data_name_from_id(data_id)
        X, y, X_test, y_test = get_data(data_name)

        if data_name == 'mnist':
            feature_dim = feature_dim_
            epoch_num = 50
            step_decay_value = 0
            opt = 1
        elif data_name == 'fashion_mnist':
            feature_dim = 64
            epoch_num = 50
            step_decay_value = 0
            opt = 1
        elif data_name == 'SVHN' or data_name == 'SSLAD-2D':
            feature_dim = 64  # Resnet
            epoch_num = 50
            step_decay_value = 1
            opt = 1
        else:
            feature_dim = 64  # Resnet
            epoch_num = 100
            step_decay_value = 2
            opt = 1
        print('step decay: ', step_decay_value, 'optimizer: ', opt)

        if len(np.shape(y)) > 1:
            # [b, 1] => [b]
            y = np.squeeze(y, axis=1)
            y_test = np.squeeze(y_test, axis=1)
        if oe_id is not None:
            # construct imbalanced data for balanced data sets
            X, y = oe_data(X, y, oe_id, imb_ratio)

        # split data into the training and validation sets
        X_train, y_train, X_val, y_val = train_val_split(X, y, ratio=0.1)
        class_num, per_class_num, imbalance_weight = get_data_info(y_train)  # get data info
        print("class num: ", class_num)
        print("per class num: ", per_class_num)
        print("imbalance weight: ", imbalance_weight)

        """
        Main experiments
        """
        # prepare model and data
        model_name, _ = get_classifier(model_id, data_name, feature_dim, class_num)
        save_path = get_save_path(data_name, oe_id, imb_ratio, model_name, feature_dim)
        train_num = X_train.shape[0]

        if data_name != 'mnist' and data_name != 'fashion_mnist':
            print('Use ImageDataGenerator')
            datagen = ImageDataGenerator(width_shift_range=4,
                                         height_shift_range=4,
                                         horizontal_flip=True,
                                         preprocessing_function=get_random_eraser(p=1, pixel_level=True))
            datagen.fit(X_train)
            train_loader = datagen.flow(X_train, y_train, batch_size=batch_size)
        else:
            train_loader = handle_data(X_train, y_train, batch_size=batch_size)
        val_loader = handle_data(X_val, y_val, batch_size)
        test_loader = handle_data(X_test, y_test, batch_size)
        print('done.')

        # our and focal loss
        alpha = imbalance_weight
        alpha = np.expand_dims(alpha, 0)

        _, clf = get_classifier(model_id, data_name, feature_dim, class_num)
        clf.summary()
        train(model_name, clf, train_loader, val_loader, test_loader, epoch_num, alpha, feature_dim,
              save_path, re, step_decay_value, train_num, batch_size, opt)


def main(model_id_set, data_id_set, oe_id_set, imb_ratio_set, batch_size, repeat):
    for ratio in imb_ratio_set:
        for data_id in data_id_set:
            for model_id in model_id_set:
                # imbalance?
                if data_id < 10:
                    for oe_id in oe_id_set:
                        main_step(model_id, data_id, oe_id, ratio, batch_size, repeat)
                else:
                    oe_id = None
                    main_step(model_id, data_id, oe_id, ratio, batch_size, repeat)


if __name__ == "__main__":
    # parameter setting
    # training setting
    batch_size_ = 128
    repeat_ = 1

    imb_ratio_set_ = [10]
    model_id_set_ = [0]

    """Due to the upload limitation, data set 'SVHN' cannot be uploaded to the 'Supplementary Material', 
    thus it is not available in this code version."""
    # data_id: (0, mnist), (1, fashion_mnist), (2, svhn), (3, cifar10)
    # Could run data_id: 0, 1, 3 in this code version for 'Supplementary Material'
    data_id_set_ = [0]
    feature_dim_ = 2  # for mnist
    oe_id_set_ = [0]

    main(model_id_set_, data_id_set_, oe_id_set_, imb_ratio_set_, batch_size_, repeat_)
