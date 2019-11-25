"""
====================================================================
Extending Auto-Sklearn with Classification Component
====================================================================

The following example demonstrates how to create a new classification
component for using in auto-sklearn.
"""
import numpy as np
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
from aif360.datasets import MEPSDataset19
# self-defined functions
from utils import plot, test, odds_difference

# Fairness metrics
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.explainers import MetricTextExplainer

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




if __name__ == '__main__':
    # fairness part
    np.random.seed(1)
    (dataset_orig_panel19_train,
     dataset_orig_panel19_val,
     dataset_orig_panel19_test) = MEPSDataset19().split([0.5, 0.8], shuffle=True)

    sens_ind = 0
    sens_attr = dataset_orig_panel19_train.protected_attribute_names[sens_ind]

    unprivileged_groups = [{sens_attr: v} for v in
                           dataset_orig_panel19_train.unprivileged_protected_attributes[sens_ind]]
    privileged_groups = [{sens_attr: v} for v in
                         dataset_orig_panel19_train.privileged_protected_attributes[sens_ind]]
    metric_orig_panel19_train = BinaryLabelDatasetMetric(
        dataset_orig_panel19_train,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)
    explainer_orig_panel19_train = MetricTextExplainer(metric_orig_panel19_train)

    print(explainer_orig_panel19_train.disparate_impact())

    X_train, y_train, X_test, y_test = dataset_orig_panel19_train.features, dataset_orig_panel19_train.labels, \
                                       dataset_orig_panel19_test.features, dataset_orig_panel19_test.labels
    # Add MLP classifier component to auto-sklearn.
    autosklearn.pipeline.components.classification.add_classifier(MLPClassifier)
    cs = MLPClassifier.get_hyperparameter_search_space()
    print(cs)
    # Build Scorer
    protected_feature_id = 1  # RACE INDEX
    mask = X_train[:, protected_feature_id]
    '''
    If used default resampling strategies 'holdout', and set arguments shuffle=False
    Then, auto-sklearn will pick the first 0.67 samples to do training and other 0.33 
    for evaluation. Therefore, here we pass mask in this way.
    '''
    fairness_scorer = autosklearn.metrics.make_scorer(
        name="fairness",
        score_func=odds_difference,
        optimum=0,
        greater_is_better=False,
        needs_proba=False,
        needs_threshold=False,
        mask=mask[:5303],
        test_mask=mask[5303:]
    )

    # Generate data.
    # Fit MLP classifier to the data.
    # default evaluation strategies holdout
    # be sure to set shuffle to false
    clf = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=40,
        per_run_time_limit=10,
        include_estimators=['random_forest'], #['MLPClassifier'],
        include_preprocessors=["no_preprocessing", ],
        resampling_strategy_arguments={'shuffle': False}
    )

    print('#' * 12 + 'Origin shape', X_train.shape, y_train.shape)
    clf.fit(X_train, y_train, metric=fairness_scorer)

    # Print test accuracy and statistics.
    y_pred = clf.predict(X_test)
    print("accuracy: ", sklearn.metrics.accuracy_score(y_pred, y_test))
    print(clf.sprint_statistics())
    print(clf.show_models())

    thresh_arr = np.linspace(0.01, 0.5, 50)
    print('Generate metrics .............')
    metrics_arrs = test(dataset_orig_panel19_test,
                        clf, thresh_arr, unprivileged_groups,
                        privileged_groups)

    fig = plot(thresh_arr, 'Classification Thresholds',
               metrics_arrs['bal_acc'], 'Balanced Accuracy',
               metrics_arrs['avg_odds_diff'], 'avg. odds diff.')
    fig.show()
    print('Done ....')
