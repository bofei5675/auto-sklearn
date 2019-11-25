"""
====================================================================
Extending Auto-Sklearn with Classification Component
====================================================================

The following example demonstrates how to create a new classification
component for using in auto-sklearn.
"""

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter, UniformFloatHyperparameter

import sklearn.metrics
import autosklearn.classification
import autosklearn.pipeline.components.classification
from autosklearn.pipeline.components.base \
    import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, SIGNED_DATA, UNSIGNED_DATA, \
    PREDICTIONS

from sklearn.metrics import confusion_matrix

# Create MLP classifier component for auto-sklearn.
class MLPClassifier(AutoSklearnClassificationAlgorithm):
    def __init__(self,
                 hidden_layer_depth,
                 num_nodes_per_layer,
                 activation,
                 alpha,
                 solver,
                 random_state=None,
                 ):
        self.hidden_layer_depth = hidden_layer_depth
        self.num_nodes_per_layer = num_nodes_per_layer
        self.activation = activation
        self.alpha = alpha
        self.solver = solver
        self.random_state = random_state

    def fit(self, X, y):
        print(X.shape, y.shape)
        self.num_nodes_per_layer = int(self.num_nodes_per_layer)
        self.hidden_layer_depth = int(self.hidden_layer_depth)
        self.alpha = float(self.alpha)

        from sklearn.neural_network import MLPClassifier
        hidden_layer_sizes = tuple(self.num_nodes_per_layer \
                                   for i in range(self.hidden_layer_depth))

        self.estimator = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                       activation=self.activation,
                                       alpha=self.alpha,
                                       solver=self.solver,
                                       random_state=self.random_state,
                                       )
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname':'MLP Classifier',
                'name': 'MLP CLassifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': False,
                # Both input and output must be tuple(iterable)
                'input': [DENSE, SIGNED_DATA, UNSIGNED_DATA],
                'output': [PREDICTIONS]
                }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        hidden_layer_depth = UniformIntegerHyperparameter(
            name="hidden_layer_depth", lower=1, upper=3, default_value=1
        )
        num_nodes_per_layer = UniformIntegerHyperparameter(
            name="num_nodes_per_layer", lower=16, upper=216, default_value=32
        )
        activation = CategoricalHyperparameter(
            name="activation", choices=['identity', 'logistic', 'tanh', 'relu'],
            default_value='relu'
        )
        alpha = UniformFloatHyperparameter(
            name="alpha", lower=0.0001, upper=1.0, default_value=0.0001
        )
        solver = CategoricalHyperparameter(
            name="solver", choices=['lbfgs', 'sgd', 'adam'], default_value='adam'
        )
        cs.add_hyperparameters([hidden_layer_depth,
                                num_nodes_per_layer,
                                activation,
                                alpha,
                                solver,
                                ])
        return cs


def odds_difference(solution, prediction, mask=None, test_mask=None):
    if mask is None:
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

if __name__ == '__main__':
    # fairness part
    # Add MLP classifier component to auto-sklearn.
    autosklearn.pipeline.components.classification.add_classifier(MLPClassifier)
    cs = MLPClassifier.get_hyperparameter_search_space()
    print(cs)

    # Generate data.
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Fit MLP classifier to the data.
    clf = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=30,
        per_run_time_limit=10,
        include_estimators=['MLPClassifier'],
        include_preprocessors=["no_preprocessing", ]
    )
    print('#' * 12 + 'Origin shape', X_train.shape, y_train.shape)
    clf.fit(X_train, y_train)

    # Print test accuracy and statistics.
    y_pred = clf.predict(X_test)
    print("accuracy: ", sklearn.metrics.accuracy_score(y_pred, y_test))
    print(clf.sprint_statistics())
    print(clf.show_models())
