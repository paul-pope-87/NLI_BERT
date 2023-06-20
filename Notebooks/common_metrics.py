# metrics helper functions

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, plot_roc_curve
from torch.nn.functional import softmax
from sklearn.metrics import confusion_matrix

# the following functions are directly from:
# https://github.com/vinyluis/Articles/blob/main/ROC%20Curve%20and%20ROC%20AUC/ROC%20Curve%20-%20Multiclass.ipynb

def plot_roc_curve(tpr, fpr, scatter = True, ax = None):
    '''
    Plots the ROC Curve by using the list of coordinates (tpr and fpr).
    
    Args:
        tpr: The list of TPRs representing each coordinate.
        fpr: The list of FPRs representing each coordinate.
        scatter: When True, the points used on the calculation will be plotted with the line (default = True).
    '''
    if ax == None:
        plt.figure(figsize = (5, 5))
        ax = plt.axes()
    
    if scatter:
        sns.scatterplot(x = fpr, y = tpr, ax = ax)
    sns.lineplot(x = fpr, y = tpr, ax = ax)
    sns.lineplot(x = [0, 1], y = [0, 1], color = 'green', ax = ax)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

def calculate_tpr_fpr(y_real, y_pred):
    '''
    Calculates the True Positive Rate (tpr) and the True Negative Rate (fpr) based on real and predicted observations
    
    Args:
        y_real: The list or series with the real classes
        y_pred: The list or series with the predicted classes
        
    Returns:
        tpr: The True Positive Rate of the classifier
        fpr: The False Positive Rate of the classifier
    '''
    
    # Calculates the confusion matrix and recover each element
    cm = confusion_matrix(y_real, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    
    # Calculates tpr and fpr
    tpr =  TP/(TP + FN) # sensitivity - true positive rate
    fpr = 1 - TN/(TN+FP) # 1-specificity - false positive rate
    
    return tpr, fpr

def get_all_roc_coordinates(y_real, y_proba):
    '''
    Calculates all the ROC Curve coordinates (tpr and fpr) by considering each point as a threshold for the predicion of the class.
    
    Args:
        y_real: The list or series with the real classes.
        y_proba: The array with the probabilities for each class, obtained by using the `.predict_proba()` method.
        
    Returns:
        tpr_list: The list of TPRs representing each threshold.
        fpr_list: The list of FPRs representing each threshold.
    '''
    tpr_list = [0]
    fpr_list = [0]
    for i in range(len(y_proba)):
        threshold = y_proba[i]
        y_pred = y_proba >= threshold
        tpr, fpr = calculate_tpr_fpr(y_real, y_pred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list, fpr_list


# this is adapted from the source above
def plot_one_vs_one_roc(y_true, logits, idx_target):
    '''
    Plots the Probability Distributions and the ROC Curves One vs One
    y_true is a tensor object n_samples long.
    logits are n_samples * n_classes, not yet softmaxed
    
    '''

    target_idx = {idx_target[a]:a for a in idx_target}
    # get all unique (ordered) pairs of classes
    c = [[[idx_target[a], idx_target[b]] for a in idx_target \
                         if idx_target[a] != idx_target[b]] for b in idx_target]
    
    classes_combinations = []
    for a in c:
        for b in a:
            classes_combinations.append(b)

    no_combo = len(classes_combinations)
    
    plt.figure(figsize = (10, len(classes_combinations)*3.0))
    bins = [i/20 for i in range(20)] + [1]
    roc_auc_ovo = {}
    
    for i in range(len(classes_combinations)):
        # Gets the class
        comb = classes_combinations[i]
        c1 = comb[0]
        c2 = comb[1]
        c1_index = target_idx[c1]
        c2_index = target_idx[c2]
        title = c1 + " vs " +c2

        # Prepares an auxiliar dataframe to help with the plots
        df_aux = pd.DataFrame(None)
        df_aux['class'] = y_true
        probs = softmax(logits, dim = -1)
        probs = pd.DataFrame(data = probs)
        df_aux = pd.concat([df_aux, probs], axis = 1)

        # Slices only the subset with both classes
        df_aux = df_aux[(df_aux['class'] == c1_index) | (df_aux['class'] == c2_index)]
        df_aux['class'] = [1 if y == c1_index else 0 for y in df_aux['class']]
        df_aux['prob'] = df_aux[c1_index]
        df_aux = df_aux.reset_index(drop = True)

        # Plots the probability distribution for the class and the rest
        locs = np.arange(1, 2*len(classes_combinations), 2)
        
        ax = plt.subplot(30, 2, locs[i])
        sns.histplot(x = "prob", data = df_aux, hue = 'class', color = 'b', ax = ax, bins = bins)
        ax.set_title(title)
        ax.legend([f"Class 1: {c1}", f"Class 0: {c2}"])
        ax.set_xlabel(f"P(x = {c1})")

        # Calculates the ROC Coordinates and plots the ROC Curves
        ax_bottom = plt.subplot(30, 2, locs[i]+1)
        tpr, fpr = get_all_roc_coordinates(df_aux['class'], df_aux['prob'])
        plot_roc_curve(tpr, fpr, scatter = False, ax = ax_bottom)
        ax_bottom.set_title("ROC Curve One-vs-One")

        # Calculates the ROC AUC OvO
        roc_auc_ovo[title] = roc_auc_score(df_aux['class'], df_aux['prob'])
    plt.tight_layout()


