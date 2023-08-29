import numpy as np
import tensorflow as tf
from loss_funcs import my_final_loss
from funcs_utility import plot_feature
from funcs_prediction import predict_from_loader
from data_preprocess.get_image_data import train_val_split


def step_decay(epoch, value, opt=0):
    if opt == 0:
        learning_rate = 1e-3
    else:
        learning_rate = 1e-1

    if value == 0:
        new_learning_rate = learning_rate
        if epoch <= 25:
            pass
        elif 25 < epoch <= 40:
            new_learning_rate = learning_rate * 0.1
        else:
            new_learning_rate = learning_rate * 0.01

    elif value == 1:  # SVHN
        new_learning_rate = learning_rate
        if epoch <= 25:
            pass
        elif 25 < epoch <= 40:
            new_learning_rate = learning_rate * 0.1
        else:
            new_learning_rate = learning_rate * 0.01

    else:
        new_learning_rate = learning_rate
        if epoch <= 50:
            pass
        elif 50 < epoch <= 80:
            new_learning_rate = learning_rate * 0.1
        else:
            new_learning_rate = learning_rate * 0.01

    print('Learning rate:', new_learning_rate)

    return new_learning_rate


def train_step(method, model, x_batch, y_batch, alpha, metric, lr, lr_region, opt):
    tem_scale = 1
    with tf.GradientTape() as model_tape:
        if method == 'ARConvL':
            [features, prototypes, dist, region] = model(x_batch, training=True)
            loss, beta, margin, probs_yc = my_final_loss(features, prototypes, region, y_batch, alpha,
                                                         tem_scale=tem_scale)
            logits = -dist
        else:
            raise Exception("Undefined model.")

        # [b, 10]
        y_batch = tf.one_hot(y_batch, depth=10)
        metric.update_state(y_batch, logits)

    # compute gradient
    grads = model_tape.gradient(loss, model.trainable_variables)

    if opt == 0:
        if method == 'ARConvL':
            tf.keras.optimizers.Adam(learning_rate=lr).apply_gradients(
                zip(grads[0:-1], model.trainable_variables[0:-1]))
            tf.keras.optimizers.Adam(learning_rate=lr_region).apply_gradients(
                zip([grads[-1]], [model.trainable_variables[-1]]))
        else:
            tf.keras.optimizers.Adam(learning_rate=lr).apply_gradients(zip(grads, model.trainable_variables))
    else:
        if method == 'ARConvL':
            tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9).apply_gradients(
                zip(grads[0:-1], model.trainable_variables[0:-1]))
            tf.keras.optimizers.SGD(learning_rate=lr_region, momentum=0.9).apply_gradients(
                zip([grads[-1]], [model.trainable_variables[-1]]))
        else:
            tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9).apply_gradients(
                zip(grads, model.trainable_variables))

    return loss, metric, region, beta, margin, probs_yc


def train(model_name, clf, train_loader, val_loader, test_loader, epoch_num, alpha, feature_dim,
              save_path, re, step_decay_value, train_num, batch_size, opt=1):
    Gmean_max = 0
    metric = tf.keras.metrics.CategoricalAccuracy()
    for epoch in range(epoch_num):
        lr = step_decay(epoch, step_decay_value, opt)
        if epoch <= 2:
            lr_region = 1e-3
        else:
            lr_region = 1e-2
        print('region Learning rate:', lr_region)
        for step, (x_batch, y_batch) in enumerate(train_loader):
            loss, metric, region, beta, margin, probs_yc = train_step(model_name, clf, x_batch, y_batch, alpha, metric,
                                                                      lr, lr_region, opt)
            if step % 40 == 0:
                print(epoch, step, 'loss:', float(loss), 'acc:', metric.result().numpy())
                metric.reset_states()
            if (step+1) > train_num / batch_size:
                break

        r_esti = region
        print('r_esti: ', r_esti)
        print('beta: ', beta-1)
        print('margin: ', margin)
        print('probs_yc', probs_yc)

        if epoch % 1 == 0:
            balanced_accuracy, recall, Gmean = predict_from_loader(val_loader, model_name, clf,
                                                                   header_pf="Val PF with " + model_name)
            predict_from_loader(test_loader, model_name, clf, header_pf="Test PF with " + model_name)

        # evaluation
        if Gmean >= Gmean_max:
            Gmean_max = Gmean
            print('save: ', Gmean_max)
            tf.keras.models.save_model(clf, save_path + 'Model_' + str(re) + '.h5')

        # plot
        if feature_dim == 2:
            if model_name == 'ARConvL':
                for step, (x_step, y_step) in enumerate(train_loader):
                    if step == 0:
                        X_train = x_step
                        y_train = y_step
                    else:
                        X_train = np.r_[X_train, x_step]
                        y_train = np.r_[y_train, y_step]
                X_train, y_train, _, _ = train_val_split(X_train, y_train, ratio=0.9)
                feature, prototype, dist, region = clf(X_train)
                region_adjusted = region
                plot_feature(feature, y_train, prototype, region_adjusted, 10, 1, name='train')
