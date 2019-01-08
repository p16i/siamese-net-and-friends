import matplotlib.pyplot as plt
import numpy as np


def plot_embedding(emd, labels, title='Embedding', filename=None):
    uniq_labels = sorted(np.unique(labels))
    plt.figure(figsize=(5, 5))

    for label in uniq_labels:
        indices = np.argwhere(labels==label).reshape(-1)
        class_embedding = emd[indices, :]

        plt.scatter(class_embedding[:, 0], class_embedding[:, 1], marker='o', label='Class %d' % label, alpha=0.6)
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.title(title)
    plt.legend()

    if filename:
        plt.savefig(filename)


def plot_stats(stats, prefix="", metrics=['loss'], filename=None):
    plt.figure(figsize=(10*len(metrics), 3))
    epoch_labels = list(range(1, len(stats['train_loss'])+1))
    for i, metric in enumerate(metrics):
        plt.subplot(1, 2, i+1)
        plt.plot(epoch_labels, stats['train_%s' % metric], label='train')
        plt.plot(epoch_labels, stats['val_%s' % metric], label='val')

        if i == 0:
            plt.legend()
        plt.xticks(epoch_labels)
        plt.title("%s: %s" % (prefix, metric))

    if filename:
        plt.savefig(filename)