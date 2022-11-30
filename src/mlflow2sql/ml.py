import mlflow
from mlflow import log_metric, log_params, log_param, log_figure
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import datasets
from sklearn_evaluation import plot
from sklearn.model_selection import ParameterGrid


def run_default():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    rf_grid = ParameterGrid(
        dict(
            n_estimators=[5, 10, 15, 25, 50, 100],
            max_depth=[5, 10, None],
            criterion=["gini", "entropy", "log_loss"],
        )
    )

    gb_grid = ParameterGrid(
        dict(
            loss=["log_loss", "exponential"],
            learning_rate=[0.1, 0.5, 1.0],
            n_estimators=[5, 10, 15, 25, 50],
        )
    )

    svc_grid = ParameterGrid(
        dict(
            C=[1.0, 2.0, 4.0],
            kernel=["linear", "poly", "rbf", "sigmoid"],
            probability=[True],
        )
    )
    
    print("Training GradientBoostingClassifier...")
    run_grid(GradientBoostingClassifier, gb_grid)

    print("Training SVC...")
    run_grid(SVC, svc_grid)

    print("Training RandomForestClassifier...")
    run_grid(RandomForestClassifier, rf_grid)


def run_grid(class_, grid):
    for idx, model_kwargs in enumerate(grid):

        if (idx + 1) % 10 == 0:
            print(f"Executed {idx + 1}/{len(grid)} so far...")

        with mlflow.start_run():
            run_experiment(class_, model_kwargs)


def run_experiment(class_, model_kwargs):
    log_params(model_kwargs)
    log_param("model_name", class_.__name__)

    X, y = datasets.make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=5,
        n_classes=2,
        flip_y=0.1,
        class_sep=0.5,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    clf = class_(**model_kwargs)

    _ = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)

    cr = classification_report(y_test, y_pred, output_dict=True)

    accuracy = accuracy_score(y_test, y_pred)
    accuracy

    log_metric("precision", cr["1"]["precision"])
    log_metric("recall", cr["1"]["recall"])
    log_metric("f1", cr["1"]["f1-score"])

    fig, ax = plt.subplots()
    plot.confusion_matrix(y_test, y_pred, ax=ax)
    log_figure(fig, "confusion_matrix.png")
    plt.close()

    fig, ax = plt.subplots()
    plot.classification_report(y_test, y_pred, ax=ax)
    log_figure(fig, "classification_report.png")
    plt.close()

    fig, ax = plt.subplots()
    plot.precision_recall(y_test, y_score, ax=ax)
    log_figure(fig, "precision_recall.png")
    plt.close()
