import argparse
import wandb
import pandas as pd
import script.config as cfg
from script.preprocessing import final_union
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset",
                default="dataset/chicago_taxi.csv",
                help="Training dataset path")
args = vars(ap.parse_args())


hyperparameter_defaults = dict(
    # data transformation
    fare_per_mile=True,
    fare_per_sec=True,
    mile_per_sec=True,
    long_lat_bin=10,
    # model
    num_leaves=31,
    max_depth=-1,
    learning_rate=0.1,
    n_estimators=200,
    min_split_gain=0,
    min_child_weight=0.001,
    min_child_samples=20,
    subsample=1,
    subsample_freq=0,
    colsample_bytree=1,
    reg_alpha=0,
    reg_lambda=0
)

wandb.init(project="sample-taxi-fare", config=hyperparameter_defaults)

config = wandb.config


# Prepare dataset
try:
    df = pd.read_csv(args["dataset"])
except:
    print("Cannot read dataset.")

X = df.copy()
y = X.pop(cfg.TARGET)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Hyperparameter tuning for dataset transformation
final_union.set_params(
    # At least one true
    feateng__fare_per_mile__use__use=config.fare_per_mile,
    feateng__fare_per_sec__use__use=config.fare_per_sec,
    feateng__mile_per_sec__use__use=config.mile_per_sec,
    cleaned__bined__bin__n_bins=config.long_lat_bin
)

# Train model
clf = LGBMClassifier(
    random_state=42,
    # Improve
    num_leaves=config.num_leaves,
    max_depth=config.max_depth,
    learning_rate=config.learning_rate,
    n_estimators=config.n_estimators,
    # Deal with overfit
    min_split_gain=config.min_split_gain,
    min_child_samples=config.min_child_samples,
    min_child_weight=config.min_child_weight,
    subsample=config.subsample,
    subsample_freq=config.subsample_freq,
    colsample_bytree=config.colsample_bytree,
    # Regularization
    reg_alpha=config.reg_alpha,
    reg_lambda=config.reg_lambda
)

# Create a model pipeline
model_pipeline = Pipeline([
    ('data', final_union),
    ('clf', clf)
])

# Calculate training metrics using cross validation strategy
score = cross_validate(model_pipeline,
                       X=X_train,
                       y=y_train,
                       scoring=['accuracy', 'f1'],
                       cv=5,
                       n_jobs=-1,
                       fit_params=None)

# Log cross validation metrics to wandb
cv_metrics = {
    "cv_accuracy": score['test_accuracy'].mean(),  # TODO: Improve this
    "cv_f1": score['test_f1'].mean()
}
wandb.log(cv_metrics)


def log_cv_plot(metric, cv=5):
    """Helper to log and plot metrics per fold to wandb

    Args:
        metric (str): 
            metric name.
        cv (int):
            number of folds.
    """
    title = f"{metric} per fold"
    plot_id = f"fold_{metric}"
    labels = [f"{n+1}" for n in range(cv)]
    data = [[label, val]
            for (label, val) in zip(labels, score['test_'+metric])]

    table = wandb.Table(data=data, columns=['fold', metric])
    wandb.log({plot_id: wandb.plot.bar(table, 'fold', metric,
                                       title=title)})


# Log metrics for each fold
metric_names = ['accuracy', 'f1']
for metric_name in metric_names:
    log_cv_plot(metric_name)


# Fit model to all training data
model_pipeline.fit(X_train, y_train)
# Evaluate on hold-out / test data
y_pred = model_pipeline.predict(X_test)
y_probas = model_pipeline.predict_proba(X_test)

# Log test metrics to wandb
test_metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred)
}
wandb.log(test_metrics)

# Get necessary data for plotting
X_train_transformed = model_pipeline['data'].transform(X_train)
X_test_transformed = model_pipeline['data'].transform(X_test)
feature_names = model_pipeline['clf'].feature_name_
labels = [0, 1]
# Visualize all classifier plots
# Some plots cannot be logged because LightGBM does not encode categorical data
wandb.sklearn.plot_classifier(model_pipeline['clf'],
                              X_train_transformed, X_test_transformed,
                              y_train, y_test, y_pred, y_probas,
                              labels=labels,
                              model_name='LightGBM',
                              feature_names=feature_names)
# TODO Visualize SHAP values


# For debuging
print(classification_report(y_test, y_pred))
