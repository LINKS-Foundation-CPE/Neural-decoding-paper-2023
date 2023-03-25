import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import matplotlib
import seaborn as sns
def clean_logs(log_path, keep=5):
    logs = os.listdir(log_path)
    while len(logs) > keep:
        os.remove(log_path + '/' + logs[0])
        logs.pop(0)


def f_unpack_dict(dct):
    """
    Unpacks all sub-dictionaries in given dictionary recursively. There should be no duplicated keys
    across all nested subdictionaries, or some instances will be lost without warning

    Parameters:
    ----------------
    dct : dictionary to unpack

    Returns:
    ----------------
    : unpacked dictionary
    """

    res = {}
    for (k, v) in dct.items():
        if isinstance(v, dict):
            res = {**res, **f_unpack_dict(v)}
        else:
            res[k] = v

    return res


def progressbar(it, prefix="", size=60, file=None):
    import sys
    if file == None:
        file = sys.stdout
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()


def my_confusion_matrix(predictions, true_values, labels=None, return_labels=False):#, label_encoder, grouped_category=True, title=None, new_order=None):
    from collections import Iterable
    """
        Evaluate a confusion matrix given a single or multiple predictions and their respective true values

        Parameters
        ----------
        predictions: iterable collection of predicted classes, can be a single array 
            or multiple ones stacked: in that case an average matrix will be evaluated

        true_values: iterable collection of true classes, can be a single array 
            or multiple ones stacked: in that case an average matrix will be evaluated
        
        labels: list, optional, default=None, collection of all possible classes
        
        return_labels: bool, optional, default=False, whether to return the labels

        """
    
    if not isinstance(predictions, Iterable) or isinstance(predictions, (str, bytes)):
        raise ValueError('A non-string iterable is required')
    
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    
    if not predictions.shape == true_values.shape and len(true_values.shape) != 1:
        raise ValueError(f'Incompatible predictions and true_values dimension while they are {predictions.shape} and {true_values.shape}')

    if len(predictions.shape) == 1:
        predictions = np.expand_dims(predictions, axis=0)
    
    unique_true_values = True if len(true_values.shape) == 1 else False
    
    
    if type(labels) not in [list, np.ndarray] :
        labels = list(set(np.concatenate((predictions.flatten(), true_values.flatten()))))
        labels.sort()
        
    if type(labels) is np.ndarray:
        labels = list(labels)
    
    matrices = []
    for pred_id in range(predictions.shape[0]):
        pred = predictions[pred_id]
        true_y = true_values if unique_true_values else true_values[pred_id]
        conf_matrix = np.zeros(shape=(len(labels), len(labels)))
        for h in range(len(pred)):
            row = labels.index(true_y[h])
            col = labels.index(pred[h])
            conf_matrix[row][col] += 1
        matrices.append(conf_matrix)

    avg_matrix = np.array(matrices).mean(axis=0)
    
    if return_labels:
        return avg_matrix, labels
    
    else:
        return avg_matrix
    
    
def distance_from_diagonal(confusion_matrix):
    """
        Given a confusion matrix, returns a dictionary with the distribution of distance 
            from diagonal

        Parameters
        ----------
        confusion_matrix: np.array with 2 dimensions

    """    
    errors = {}
    (rows, columns) = confusion_matrix.shape
    for i in range(rows):
        for j in range(columns):
            distance = j - i
            if distance not in errors.keys():
                errors[distance] = 0
            errors[distance] += confusion_matrix[i][j]
    return errors


def find_cap_to_balance(repetitions, margin=10):
    n = len(repetitions)
    s = repetitions.copy()
    s.sort()

    rep = 0
    while True or rep < len(repetitions):
        temp = [s[rep+1]]*(n-rep-1)
        for elem in s[:rep+1]:
            temp.append(elem)
        temp = np.array(temp)
        balance = temp.std()/temp.mean()*100
        # print(temp, round(balance, 4), '%')
        if balance >= margin:
            cap = s[rep+1]
            break
        rep += 1


    for j in range(cap):
        temp = [s[rep+1]-j]*(n-rep-1)
        for elem in s[:rep+1]:
            temp.append(elem)
        temp = np.array(temp)
        balance = temp.std()/temp.mean()*100
        # if j%15 == 0:
            # print(temp, round(balance, 4), '%')
        if balance <= margin:
            # print(temp, round(balance, 4), '%')
            cap = s[rep+1]-j
            break

    return cap

def display_testing_phase_results(conf_matrix, y_test, my_labels, errors_distribution, n_outputs, network_name='network',save=True):
    fig = plt.figure(figsize=(11, 6))
    gs = matplotlib.gridspec.GridSpec(nrows=3, ncols=3)

    ax1 = fig.add_subplot(gs[:, :-1])
#     ax1.set_title(network_name, fontsize=22)
    conf_matrix_norm = normalize(conf_matrix, axis=1, norm='l1')
    im = ax1.imshow(conf_matrix_norm, cmap=sns.cubehelix_palette(as_cmap=True, light=.95, gamma=0.75), vmin=0, vmax=1)

    ax1.vlines([i + 0.5 for i in range(n_outputs)], -0.5, n_outputs - 0.5, linewidth=2, colors='White')
    ax1.hlines([i + 0.5 for i in range(n_outputs)], -0.5, n_outputs - 0.5, linewidth=2, colors='White')
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Fraction of class samples',size=18)
    rows, columns = conf_matrix.shape
    for i in range(rows):
        for j in range(columns):
            c = conf_matrix[i][j]
            if c > 0:
                color = 'White' if conf_matrix_norm[i][j]>=.65 else 'Black'
#                 ax1.text(j, i, str(round(c,0)), va='center', ha='center', c=color, fontsize=12)
#     ax1.set_xticks(list(range(n_outputs)))
#     ax1.set_yticks(list(range(n_outputs)))
    ax1.set_ylabel('true value', fontsize=18)
    ax1.set_xlabel('prediction', fontsize=18)

    ax2 = fig.add_subplot(gs[1, -1])
    ax2.set_title('Distance from\n diagonal', fontsize=16)
    x = list(errors_distribution.keys())
    x.sort()
    y = [errors_distribution[distance] for distance in x]
    ax2.step(x, y, where='mid', c='midnightblue')
    ax2.axvspan(-1.5, 1.5, alpha=0.4, color='palevioletred')
    x_max = min(max(list(errors_distribution.keys())), 10)
    lim = x_max if x_max%2 == 0 else x_max-1
    lim = min(lim, 10)
    ax2.set_xticks(list(range(-lim, lim+2, 2)))
    ax2.set_xlim(-x_max, x_max)
    ax2.set_yscale('log')
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()
    ax2.set_ylabel('samples', fontsize=16)
    ax2.set_xlabel('distance', fontsize=16, y=-10)
    
    e0 = errors_distribution[0]
    e1 = errors_distribution[1]+errors_distribution[-1]
    
    ax3 = fig.add_subplot(gs[2, -1])
    ax3.axis('off')
#     ax3.text(0,0.9,f'On diagonal: {round(e0, 3)} [{round(e0/len(y_test)*100, 1)}%]', fontsize=11.5)
#     ax3.text(0,0.8,f'Diagonal ±1: {round(e0+e1, 3)} [{round((e0+e1)/len(y_test)*100, 1)}%]', fontsize=11.5)
    ax3.text(0,0.9,f'Accuracy: {round(e0/len(y_test)*100, 1)}%', fontsize=20)
    ax3.text(0,0.5,f'Diagonal ±1: {round((e0+e1)/len(y_test)*100, 1)}%', fontsize=20)

#     plt.tight_layout(rect=[0, 0, 1, 0.96], pad=3.0)
    plt.tight_layout()
    if save:
        fig.savefig('testing_results_'+network_name+'.pdf')
        fig.savefig('testing_results_'+network_name+'.png')
    plt.show()