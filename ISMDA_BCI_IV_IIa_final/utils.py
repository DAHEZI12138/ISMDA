import torch
import random
import numpy as np
import os
import sys
import logging
from shutil import copy
import matplotlib
import matplotlib.pyplot as plt
from openTSNE import TSNE
from sklearn.metrics import confusion_matrix
from dataloader.BCIIV import data_generator
from trainer.training_evaluation import cross_domain_test_fea

# plot feature map of subject n
def plot_feature_map(configs, home_path, n, device, save_dir_features, save_dir_models):
    src_train_dl, trg_train_dl = data_generator(configs, n)
    cross_domain_test_fea(src_train_dl, trg_train_dl, device, home_path, save_dir_features, n, save_dir_models)
    dataset = torch.load(os.path.join(home_path + save_dir_features, 'final_train_' + str(n) + '_to_' + str(n) + '_round_' + str(n) + '.pt')) #Read the features of subject n
    train_features = dataset['train_features'].numpy()
    train_labels = dataset['train_labels'].numpy()
    test_features = dataset['test_features'].numpy()
    test_labels = dataset['test_labels'].numpy()

    tsen = TSNE(
        perplexity=configs.perplexity,
        metric="euclidean",
        n_jobs=8,
        random_state=42,
        verbose=True
    )
    embedding_train = tsen.fit(train_features)
    embedding_test = tsen.fit(test_features)
    plot(embedding_train, train_labels, embedding_test, test_labels, colors = MOTOR_COLORS, title='Feature Map')
    name = os.path.join('features_BCI_IIa' + str(n))
    plt.savefig(name, format='png')
    plt.show()


def plot_all_confusion_matrix(home_path, classes, args):
    file_names = os.listdir(os.path.join(home_path + args.save_dir_pre_true_label))
    y_pred = []
    y_test = []

    for file_name in file_names:
        test_dataset = torch.load(os.path.join(home_path + args.save_dir_pre_true_label, file_name))
        y_pred = np.concatenate((y_pred, test_dataset['pred_labels'].numpy()), axis=0)
        y_test = np.concatenate((y_test, test_dataset['true_labels'].numpy()), axis=0)

    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    plot_confusion_matrix(cm, classes)

def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    - cm : The value of the computed confusion matrix
    - classes : Columns corresponding to each row and each column in the confusion matrix
    - normalize : TTrue:Show percentage, False:Show number of items
    """
    np.set_printoptions(precision=2)
    savename = 'bci_iv'
    FP = sum(cm.sum(axis=0)) - sum(np.diag(cm))
    FN = sum(cm.sum(axis=1)) - sum(np.diag(cm))
    TP = sum(np.diag(cm))
    TN = sum(cm.sum().flatten()) - (FP + FN + TP)
    SUM = TP + FP
    PRECISION = TP / (TP + FP)  # accuracy
    RECALL = TP / (TP + FN)  # recall


    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("显示百分比：")
        np.set_printoptions(precision=2)
        print(cm)
    else:
        print('显示具体数字：')
        print(cm)

    ind_array = np.arange(len(classes) + 1)
    x, y = np.meshgrid(ind_array, ind_array)  # Generate Coordinate Matrix
    diags = np.diag(cm)  # Diagonal True Positive values
    TP_FNs, TP_FPs = [], []
    for x_val, y_val in zip(x.flatten(), y.flatten()):  # Parallel traversal
        max_index = len(classes)
        if x_val != max_index and y_val != max_index:  # Plot the numerical values of each cell in the confusion matrix
            c = cm[y_val][x_val] * 100
            c = round(c,2)
            plt.text(x_val, y_val, str(c) + '%', color='black', fontsize=12, va='center', ha='center')
        elif x_val == max_index and y_val != max_index:  # Plot the recall rates for each data category, corresponding to the rightmost column.
            TP = diags[y_val]
            TP_FN = cm.sum(axis=1)[y_val]
            recall = TP / (TP_FN)
            if recall != 0.0 and recall > 0.01:
                recall = str('%.2f' % (recall * 100,))
            elif recall == 0.0:
                recall = '0'
            TP_FNs.append(TP_FN)
            plt.text(x_val, y_val, str(recall) + '%', color='black', va='center', ha='center', fontsize=12)
        elif x_val != max_index and y_val == max_index:  # Plot the precision rates for each data category, which corresponds to the bottom row.
            TP = diags[x_val]
            TP_FP = cm.sum(axis=0)[x_val]
            precision = TP / (TP_FP)
            if precision != 0.0 and precision > 0.01:
                precision = str('%.2f' % (precision * 100,)) + '%'
            elif precision == 0.0:
                precision = '0'
            TP_FPs.append(TP_FP)
            plt.text(x_val, y_val, str(precision), color='black', va='center', ha='center', fontsize=12)

    plt.text(max_index, max_index, str('%.2f' % (PRECISION * 100,)) + '%', color='black', va='center',ha='center', fontsize=12)
    A = np.zeros([5, 5])
    plt.imshow(A, alpha=0.1, cmap='gray')
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.YlGnBu, vmax=1, vmin=0)

    plt.title(title)
    plt.colorbar()
    classes1 = []
    classes0 = []
    for l in classes:
        classes1.append(l)
        classes0.append(l)
    classes1.append('Precision')
    classes0.append('Recall')
    xlocations = np.array(range(len(classes)+1))
    plt.xticks(xlocations, classes0)
    plt.yticks(xlocations, classes1)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # offset the tick
    tick_marks = np.array(range(len(classes)+1)) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.tight_layout()
    plt.savefig(savename, format='png')
    plt.show()



def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark  = False


def _logger(logger_name, level=logging.DEBUG):
    # Method to return a custom logger with the given name and level

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def copy_Files(destination, da_method):
    destination_dir = os.path.join(destination, "MODEL_BACKUP_FILES")
    os.makedirs(destination_dir, exist_ok=True)
    copy("train_CD.py", os.path.join(destination_dir, "train_CD.py"))
    copy(f"trainer/{da_method}.py", os.path.join(destination_dir, f"{da_method}.py"))
    copy(f"trainer/training_evaluation.py", os.path.join(destination_dir, f"training_evaluation.py"))
    copy(f"config_files/configs.py", os.path.join(destination_dir, f"configs.py"))
    copy("dataloader/BCIIV.py", os.path.join(destination_dir, "BCIIV.py"))
    copy(f"models/models.py", os.path.join(destination_dir, f"models.py"))
            
def plot(x_train, y_train, x_test, y_test, colors, ax=None, title=None, draw_legend=True, draw_centers=True, draw_cluster_labels=True,
         legend_kwargs=None, label_order=None, **kwargs):

    if ax is None:
        _, ax = matplotlib.pyplot.subplots(figsize=(8, 8))

    if title is not None:
        ax.set_title(title, fontsize=25)

    plot_params = {"alpha": kwargs.get("alpha", 1), "s": kwargs.get("s", 20)}

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y_train), label_order))
        classes = [l for l in label_order if l in np.unique(y_train)]
    else:
        classes = np.unique(y_train)

    classes0 = []
    for yi in classes:
        if yi == 0:
            yi = "L H"
        elif yi == 1:
            yi = "R H"
        elif yi == 2:
            yi = "Feet"
        elif yi == 3:
            yi = "Tongue"
        classes0.append(yi)


    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes0, default_colors())}

    y_train0 = []
    y_test0 = []
    for y_0 in y_train:
        if y_0 == 0:
            y_0 = "L H"
        elif y_0 == 1:
            y_0 = "R H"
        elif y_0 == 2:
            y_0 = "Feet"
        elif y_0 == 3:
            y_0 = "Tongue"
        y_train0.append(y_0)
    for y_1 in y_test:
        if y_1 == 0:
            y_1 = "L H"
        elif y_1 == 1:
            y_1 = "R H"
        elif y_1 == 2:
            y_1 = "Feet"
        elif y_1 == 3:
            y_1 = "Tongue"
        y_test0.append(y_1)

    point_colors = list(map(colors.get, y_test0))
    ax.scatter(x_test[:, 0], x_test[:, 1], c=point_colors, rasterized=False, **plot_params, marker="x")

    # Plot mediods
    if draw_centers:
        centers0 = []
        centers1 = []
        for yi in classes:
            mask0 = yi == y_train
            mask1 = yi == y_test
            centers0.append(np.median(x_train[mask0, :2], axis=0))
            centers1.append(np.median(x_test[mask1, :2], axis=0))
        centers0 = np.array(centers0)
        centers1 = np.array(centers1)

        center_colors = list(map(colors.get, classes0))
        ax.scatter(centers1[:, 0], centers1[:, 1], c=center_colors, s=40, alpha=1, edgecolor="k", marker='v')

        # Draw mediod labels
        if draw_cluster_labels:
            for idx, label in enumerate(classes0):
                ax.text(
                    centers1[idx, 0],
                    centers1[idx, 1] + 2.2,
                    label,
                    fontsize=kwargs.get("fontsize", 20),
                    horizontalalignment="center",
                )

    if draw_legend:
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=10,
                alpha=1,
                linewidth=0,
                label=yi,
                markeredgecolor="k",
            )
            for yi in classes0
        ]
        legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(0.8, 0.9), frameon=False,)
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)



MACOSKO_COLORS = {
    "Amacrine cells": "#A5C93D",
    "Astrocytes": "#8B006B",
    "Bipolar cells": "#2000D7",
    "Cones": "#538CBA",
    "Fibroblasts": "#8B006B",
    "Horizontal cells": "#B33B19",
    "Microglia": "#8B006B",
    "Muller glia": "#8B006B",
    "Pericytes": "#8B006B",
    "Retinal ganglion cells": "#C38A1F",
    "Rods": "#538CBA",
    "Vascular endothelium": "#8B006B",
}
MOTOR_COLORS = {
    "L H": "#A5C93D",
    "R H": "#8B006B",
    "Feet": "#2000D7",
    "Tongue": "#B33B19"
}
def enable_dropout(model):
    if type(model) == tuple:
        model[0].train()
        model[1].train()
    else:
        for m in model.modules():
            m.train()

def parameter_count(module):
    trainable, non_trainable = 0, 0
    for p in module.parameters():
        if p.requires_grad:
            trainable += p.numel()
        else:
            non_trainable += p.numel()
    return trainable, non_trainable

def ensure_directories_exist(*directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"document '{directory}' not exist, has been build.")
        else:
            print(f"document '{directory}' exist.")