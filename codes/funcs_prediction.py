import tensorflow as tf
import numpy as np
import sklearn.metrics as sk_metrics


def predict(model_name, clf, X_pre):
    # test
    if model_name == 'ARConvL':
        features, prototypes, dist, region = clf(X_pre)
        probs_cx = tf.nn.softmax(-dist)
        probs_pre = probs_cx
    else:
        raise Exception("Undefined model.")

    y_pre = tf.argmax(probs_pre, -1)

    return y_pre


def Gmean_compute(recall):
    Gmean = 1
    for r in recall:
        Gmean = Gmean * r
    Gmean = pow(Gmean, 1/len(recall))

    return Gmean


def classification_report(y_true, y_pre, header_pf="None"):
    """
    report classification PF evaluation in a formal way
    label values: 0 - clean, 1 - defective
    """
    confusion_mat = sk_metrics.confusion_matrix(y_true, y_pre)
    balanced_accuracy = sk_metrics.balanced_accuracy_score(y_true, y_pre)
    recall = sk_metrics.recall_score(y_true, y_pre, average=None)
    Gmean = Gmean_compute(recall)

    tpr = np.diag(confusion_mat) / np.sum(confusion_mat, 1)
    fpr = np.zeros([len(recall) - 1, len(recall)])
    for r in range(len(recall)):
        fpr_tmp = confusion_mat[:, r] / np.sum(confusion_mat, 1)
        fpr_tmp = np.delete(fpr_tmp, r, axis=0)
        fpr[:, r] = fpr_tmp

    Auc = np.mean(np.mean((1 + tpr - fpr) / 2))

    # report
    print("#### " + header_pf)
    print("recall: ", recall)
    print("balanced_accuracy: ", balanced_accuracy)
    print("Auc: ", Auc)
    print("Gmean: ", Gmean)

    return balanced_accuracy, recall, Gmean, Auc


def predict_from_loader(loader, model_name, clf, header_pf="None", data_num=None):
    y_acc = []
    y_pre_acc = []
    for step, (x_step, y_step) in enumerate(loader):
        y_pre_step = predict(model_name, clf, x_step)
        if model_name == 'affinity_CNN':
            y_step = np.argmax(y_step[:, 0:-1], 1)
        y_acc = np.r_[y_acc, y_step]
        y_pre_acc = np.r_[y_pre_acc, y_pre_step]
        if data_num is not None:
            if step == 0:
                batch_size = np.shape(x_step)[0]
            if (step + 1) > data_num / batch_size:
                break
    balanced_accuracy, recall, Gmean, Auc = classification_report(y_acc, y_pre_acc, header_pf)

    return balanced_accuracy, recall, Gmean