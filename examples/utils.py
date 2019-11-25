from aif360.metrics import ClassificationMetric
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
def test(dataset, model, thresh_arr, unprivileged_groups, privileged_groups,favorable_label=1):
    try:
        # sklearn classifier or auto sklearn clf
        y_val_pred_prob = model.predict_proba(dataset.features)
        pos_ind = favorable_label
    except AttributeError:
        # aif360 inprocessing algorithm
        y_val_pred_prob = model.predict(dataset).scores
        pos_ind = 0

    metric_arrs = defaultdict(list)
    for thresh in thresh_arr:
        y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
            dataset, dataset_pred,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)

        metric_arrs['bal_acc'].append((metric.true_positive_rate()
                                       + metric.true_negative_rate()) / 2)
        metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
        metric_arrs['disp_imp'].append(metric.disparate_impact())
        metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
        metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
        metric_arrs['theil_ind'].append(metric.theil_index())

    return metric_arrs

def plot(x, x_name, y_left, y_left_name, y_right, y_right_name):
    fig, ax1 = plt.subplots(figsize=(10,7))
    ax1.plot(x, y_left)
    ax1.set_xlabel(x_name, fontsize=16, fontweight='bold')
    ax1.set_ylabel(y_left_name, color='b', fontsize=16, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)
    ax1.set_ylim(0.5, 0.8)

    ax2 = ax1.twinx()
    ax2.plot(x, y_right, color='r')
    ax2.set_ylabel(y_right_name, color='r', fontsize=16, fontweight='bold')
    if 'disparate impact' in y_right_name:
        ax2.set_ylim(0., 0.7)
    else:
        ax2.set_ylim(-0.25, 0.1)

    best_ind = np.argmax(y_left)
    ax2.axvline(np.array(x)[best_ind], color='k', linestyle=':')
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.grid(True)
    return fig


def odds_difference(solution, prediction, mask=None, test_mask=None):
    if test_mask.shape[0] == solution.shape[0]:
        mask = test_mask
    # compute average odds difference
    # assuem binary classification task
    # by default priviledge group has id = 1
    soln_pri_group, pred_pri_group = solution[mask == 1].reshape(-1, 1), prediction[mask == 1].reshape(-1, 1)
    soln_unpri_group, pred_unpri_group = solution[mask == 0].reshape(-1, 1), prediction[mask == 0].reshape(-1, 1)
    # print('Pri', soln_pri_group.shape, pred_pri_group.shape)
    # print('Unpri', soln_unpri_group.shape, pred_unpri_group.shape)
    pri_sum = (mask == 1).sum()
    unpri_sum = (mask == 0).sum()
    tn, fp, fn, tp = confusion_matrix(solution, prediction).ravel()
    #print('TPR', tp / (tp + fn))
    #print('FPR', fp / (fp + tn))
    pri_tn, pri_fp, pri_fn, pri_tp = (confusion_matrix(soln_pri_group, pred_pri_group) \
                                      / pri_sum).ravel()
    unpri_tn, unpri_fp, unpri_fn, unpri_tp = (confusion_matrix(soln_unpri_group, pred_unpri_group) \
                                              / unpri_sum).ravel()
    # refer to https://github.com/IBM/AIF360/blob/bb8f0b254cde5f13ab6c9b0cc92c2d7bc977089f/aif360/metrics/dataset_metric.py#L73
    pri_tpr = (pri_tp) / (pri_tp + pri_fn)
    unpri_tpr = (unpri_tp) / (unpri_tp + unpri_fn)
    tpr_diff = unpri_tpr - pri_tpr

    pri_fpr = (pri_fp) / (pri_fp + pri_tn)
    unpri_fpr = (unpri_fp) / (unpri_fp + unpri_tn)
    fpr_diff = unpri_fpr - pri_fpr
    #print(tpr_diff, fpr_diff)
    return 0.5 * (tpr_diff + fpr_diff)