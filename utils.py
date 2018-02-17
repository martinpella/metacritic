import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import itertools

def plot_embeddings(red_term_doc, y_train, qty=1000):
    svd_pos = red_term_doc[y_train == 1]
    svd_neg = red_term_doc[y_train == 0]
    
    plt.figure(figsize=(10, 10))
    plt.scatter(svd_pos[:qty, 0], svd_pos[:qty, 1], alpha=0.5, marker='.', label='Positive')
    plt.scatter(svd_neg[:qty, 0], svd_neg[:qty, 1], alpha=0.5, marker='.', label='Negative');
    plt.legend();
    
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')