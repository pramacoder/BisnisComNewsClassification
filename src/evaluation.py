import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc, 
    precision_recall_curve, 
    average_precision_score,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

def plot_confusion_matrix(y_true, y_pred, classes, normalize=True, save_path=None, show=False):
    """
    Plots the confusion matrix. Can be normalized or absolute.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm) # handle division by zero
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
        data_to_plot = cm_norm
    else:
        fmt = 'd'
        title = 'Absolute Confusion Matrix'
        data_to_plot = cm
        
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data_to_plot, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    plt.close()

def plot_roc_curve(y_true, y_score, classes, save_path=None, show=False):
    """
    Plots One-vs-Rest ROC Curve for multiclass classification.
    y_score: shape (n_samples, n_classes) - probabilities from model.predict_proba
    """
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
             color='deeppink', linestyle=':', linewidth=4)
             
    cmap = plt.get_cmap('tab10')
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=cmap(i % 10), lw=2,
                 label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) to Multi-Class')
    plt.legend(loc="lower right", fontsize='small', ncol=2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    plt.close()

def plot_pr_curve(y_true, y_score, classes, save_path=None, show=False):
    """
    Plots Precision-Recall curve for multiclass.
    """
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_true_bin[:, i], y_score[:, i])

    # Micro-average PR curve
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_bin.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(y_true_bin, y_score, average="micro")

    plt.figure(figsize=(10, 8))
    plt.plot(recall["micro"], precision["micro"], color='gold', lw=2, linestyle=':',
             label=f'micro-average PR curve (area = {average_precision["micro"]:0.2f})')
             
    cmap = plt.get_cmap('tab10')
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], color=cmap(i % 10), lw=2,
                 label=f'PR curve of class {classes[i]} (area = {average_precision[i]:0.2f})')
                 
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left", fontsize='small', ncol=2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    plt.close()
    
def plot_classification_report(y_true, y_pred, classes, save_path=None, show=False):
    """
    Plots a heatmap of the classification report (precision, recall, f1).
    """
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    df_report = pd.DataFrame(report).iloc[:-1, :].T # exclude support row
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_report, annot=True, cmap='viridis', fmt=".2f")
    plt.title('Classification Report Heatmap')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    plt.close()
