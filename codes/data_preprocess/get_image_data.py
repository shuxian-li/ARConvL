from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import scipy.io as scio


def get_data_name_from_id(data_id):
    if data_id == 0:
        data_name_str = "mnist"
    elif data_id == 1:
        data_name_str = "fashion_mnist"
    elif data_id == 2:
        data_name_str = "SVHN"
    elif data_id == 3:
        data_name_str = "cifar10"
    else:
        raise Exception("Undefined data_id.")

    return data_name_str

def get_data(data_name_str):
    # read data
    if data_name_str == 'mnist':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif data_name_str == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    elif data_name_str == 'cifar10':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif data_name_str == 'SVHN':
        data_path = "./data_used"
        train_file = data_path + "/SVHN/train_32x32.mat"
        test_file = data_path + "/SVHN/test_32x32.mat"
        train = scio.loadmat(train_file)
        X_train = train.get('X')
        X_train = np.transpose(X_train, (3, 0, 1, 2))
        y_train = train.get('y')-1
        test = scio.loadmat(test_file)
        X_test = test.get('X')
        X_test = np.transpose(X_test, (3, 0, 1, 2))
        y_test = test.get('y')-1
    else:
        raise Exception("Undefined data_name.")

    return X_train, y_train, X_test, y_test

def oe_data(X, y, oe_id, ratio=10):
    y_unique = np.unique(y)
    class_num = len(y_unique)
    X_oe = X
    y_oe = y
    for i in range(class_num):
        if i % 2 == oe_id:
            idx_min = np.where(y_oe == i)
            idx_else = np.where(y_oe != i)
            X_min = X_oe[idx_min]
            y_min = y_oe[idx_min]
            n_min = int(len(y_min) / ratio)
            idx_random = np.random.choice(np.arange(len(y_min)), size=n_min, replace=False)
            X_oe = np.r_[X_min[idx_random], X_oe[idx_else]]
            y_oe = np.r_[y_min[idx_random], y_oe[idx_else]]

    return X_oe, y_oe

def train_val_split(X_all, y_all, ratio=0.1):
    # Split data into the training and validation sets
    if ratio > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=ratio, stratify=y_all)
    else:
        X_train = X_all
        y_train = y_all
        X_val = X_all
        y_val = y_all

    return X_train, y_train, X_val, y_val

def get_data_info(y):
    """
    Return
        class_num, per_class_num, imbalance_ratio
    """
    y_unique = np.unique(y)
    class_num = len(y_unique)
    per_class_num = []
    for i in range(class_num):
        per_class_num = np.r_[per_class_num, sum(y == i)]
    imbalance_weight = 1/per_class_num/min(1/per_class_num)

    return class_num, per_class_num, imbalance_weight


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1 / 0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        if input_img.ndim == 3:
            img_h, img_w, img_c = input_img.shape
        elif input_img.ndim == 2:
            img_h, img_w = input_img.shape

        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            if input_img.ndim == 3:
                c = np.random.uniform(v_l, v_h, (h, w, img_c))
            if input_img.ndim == 2:
                c = np.random.uniform(v_l, v_h, (h, w))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w] = c

        return input_img

    return eraser


def handle_data(X_tf, y_tf, batch_size=128):
    loader = tf.data.Dataset.from_tensor_slices((X_tf, y_tf))

    loader = loader.shuffle(len(y_tf))

    loader = loader.batch(batch_size)

    return loader

