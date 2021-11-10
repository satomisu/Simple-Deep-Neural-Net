import numpy as np


def prediction_accuracy(truth, prediction):
    # Assumes that inputs are 1d arrays
    # Get dimensions and lengths of truth and prediction
    truth_dim_1d = 1 in truth.shape
    prediction_dim_1d = 1 in prediction.shape
    len_truth = len(truth)
    len_prediction = len(prediction)
    # If both of them are 1d and have the same number of elements
    if truth_dim_1d and prediction_dim_1d:
        if len_truth == len_prediction:
            # Compute prediction accuracy
            accuracy_list = []
            for i in range(len_truth):
                accuracy_list.append(compute_accuracy(truth[i], prediction[i]))
            # Then get average accuracy and std
            mean = np.mean(np.array(accuracy_list))
            std = np.std(np.array(accuracy_list))
            return mean, std
        else:
            print(f'number of elements in truth: {len_truth}'
                  f'number of elements in prediction" {len_prediction}'
                  f'They neen to match!')
            exit(0)
    else:
        print('analyze.prediction_accuracy function assumes that'
              ' the inputs are 1d arrays. Something else have been given.')
        exit(0)


def compute_accuracy(truth, prediction):
    pred_to_truth_ratio = prediction/truth
    # Absolute error between the truth and prediction
    error = abs(1-pred_to_truth_ratio)
    # The accuracy percentage
    accuracy = 100*(1-error)
    return accuracy
