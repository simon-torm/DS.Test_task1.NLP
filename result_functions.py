import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Function for print main metrics for prediction
def print_scores(y_val, y_pred, threshold=0.5, plot_fscore_threshold = False):
    """
    Takes y_val, y_pred and calculates accuracy, precision, recall, F1-Score with threshold.
    """


    # Rounding the prediction with the select threshold
    th_y_pred = np.where(np.array(y_pred) > threshold, 1, 0)

    # Print scores
    acc_score = accuracy_score(y_val, th_y_pred)
    pr_score, re_score, f_score, _ = precision_recall_fscore_support(y_val,
                                                                     th_y_pred,
                                                                     average='binary')
    print(f"Accuracy score: {round(acc_score, 3)}")
    print(f"Precision score: {round(pr_score, 3)}")
    print(f"Recall score: {round(re_score, 3)}")
    print(f"F-score: {round(f_score, 3)}")

    if plot_fscore_threshold:
        # Calculate F1-Score for the thresholds
        f_scores = list()
        thresholds = np.linspace(0, 1, 100)

        for step in thresholds:
            th_y_pred = np.where(np.array(y_pred) > step, 1, 0)
            _, _, f_score, _ = precision_recall_fscore_support(y_val,
                                                               th_y_pred,
                                                               average='binary')
            f_scores.append(f_score)

        # Drawing F1-Score for the thresholds
        ax = sns.lineplot(x=thresholds, y=f_scores)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('F1-Score')

        # Print the best F1-Score and the appropriate threshold
        print(f"\nMax F1-Score: {round(np.max(f_scores), 3)} (threshold: {round(thresholds[np.argmax(f_scores)], 3)})\n")



# Function for drawing a history graphs from NNs
def plot_history_nn(history, combined_keys=None, calculate_f1_score=False):
    """
    Takes a history dict from NN and plots all metrics.

    Parameters
    ----------
    history : history.history object what NN.fit returns
    combined_keys : array_like
        If neen to combine several metrics in one window.
        Metrics contained in the same list like:
        combined_keys=[[metric1, metric2, metric3], [metric4]]
        will be drawn in one window (metric1-3)
    calculate_f1_score : Boolean
        Will calculate the F1-Score for each epoch.
        For "combined_keys" use the "val_f1_score" key
    """


    # Calculate F1-Score for all epochs if calculate_f1_score == True
    if calculate_f1_score:
        precision = np.array(history['val_precision'])
        recall = np.array(history['val_recall'])
        history['val_f1_score'] = 2 * ((precision * recall) / (precision + recall))
        # Replace np.nan to zero
        history['val_f1_score'] = np.nan_to_num(history['val_f1_score'])

    # Check if uses without combined_keys
    if not combined_keys:
        keys = list(history.keys())
        # Raise error if number of metrics not a pair
        if len(keys) % 2 != 0:
            raise ValueError("History object doesn't have a pair number of keys")
        x = list(range(len(history[keys[0]])))
    else:
        keys = list(combined_keys)
        # Check for each element of combined_keys that it is a list
        for key in keys:
            if type(key) != list:
                raise ValueError(f"Element:'{key}' is not a list")
        x = list(range(len(history[keys[0][0]])))

    # Prepare windows
    rows = int(np.ceil(len(keys) / 2))
    _, axs = plt.subplots(rows, 2, figsize=(15, 5 * rows))

    # Draw metrics
    for i, key in enumerate(keys):
        indx = np.unravel_index(i, shape=axs.shape)

        if combined_keys:
            for inter_key in key:
                sns.lineplot(x=x, y=history[inter_key], ax=axs[indx], label=inter_key)
        else:
            sns.lineplot(x=x, y=history[key], ax=axs[indx], label=key)
        axs[indx].set_xlabel("Epoch")

    # Print the best results for each metric
    for key in history.keys():
        # Save min/max results
        min_res = str(round(np.min(history[key]), 3))
        max_res = str(round(np.max(history[key]), 3))
        # Find the appropriate epoch for min/max results
        epoch_min_res = np.argmin(history[key]) + 1
        epoch_max_res = np.argmax(history[key]) + 1

        indent = ' ' * (15 - len(key))

        print(f"{key}: {indent} min-{min_res}({epoch_min_res} epoch), \t max-{max_res}({epoch_max_res} epoch)")

