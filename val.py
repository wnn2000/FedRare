import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from tqdm import tqdm


def compute_bacc(model, dataloader, get_confusion_matrix, args):
    assert dataloader.dataset.tta_num == args.tta
    num_classes = 7
    all_preds = np.zeros([dataloader.dataset.get_num_images(), num_classes])
    all_labels = np.zeros(dataloader.dataset.get_num_images())
    model.eval()
    ii = 0
    with torch.no_grad():
        for (x, label) in dataloader:
            x = x.cuda()
            if model.project == False:
                now_pred = model(x)
            else:
                _, now_pred = model(x)
            new_pred, new_label = get_val_result(
                now_pred.cpu().numpy(), label.cpu().numpy(), num_classes, args)

            all_preds[ii:ii + new_pred.shape[0], :] = new_pred
            all_labels[ii:ii + new_pred.shape[0]] = new_label
            ii += new_pred.shape[0]

    all_result = np.argmax(all_preds, 1)

    assert all_result.shape[0] == all_labels.shape[0]
    assert all_result.shape[0] == dataloader.dataset.get_num_images()

    acc = balanced_accuracy_score(all_labels, all_result)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(all_labels, all_result)

    if get_confusion_matrix:
        return acc, conf_matrix
    else:
        return acc


def get_val_result(pred, label,  num_classes, args):
    val_sample_repeat_num = args.tta

    if val_sample_repeat_num > 0:
        pred = pred.reshape(-1, val_sample_repeat_num, num_classes)
        pred = np.transpose(pred, (0, 2, 1))
        label = label.reshape(-1, val_sample_repeat_num)
        label = label[:, 0]
        predictions = np.mean(pred, 2)
    else:
        predictions = pred

    return predictions, label
