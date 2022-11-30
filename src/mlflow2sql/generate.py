from pathlib import Path
import time
from subprocess import Popen
from uuid import uuid4
from contextlib import contextmanager

import psutil
import matplotlib.pyplot as plt
import numpy as np
import mlflow
from mlflow import log_metric, log_param, log_figure


def runs(n=10, experiment=None):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    if experiment:
        mlflow.set_experiment(experiment)
    else:
        mlflow.set_experiment(str(uuid4())[:8])

    for _ in range(n):
        with mlflow.start_run():
            log_param("param_number", 42)

            log_metric("metric_number", 2)

            log_metric("metric_multiple_increment", 1)
            log_metric("metric_multiple_increment", 2.1)

            fig, ax = plt.subplots()
            ax.scatter(np.random.rand(100), np.random.rand(100))

            log_figure(fig, "scatter.png")


def kill_process_and_children(pid: int, sig: int = 15):
    # https://stackoverflow.com/a/64173588/709975
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess as e:
        # Maybe log something here
        return

    for child_process in proc.children(recursive=True):
        child_process.send_signal(sig)

    proc.send_signal(sig)


@contextmanager
def start_mlflow(backend_store_uri=None, default_artifact_root=None):
    backend_store_uri = str(Path(backend_store_uri or ".").resolve())
    default_artifact_root = str(Path(default_artifact_root or ".").resolve())

    cmd = [
        "mlflow",
        "server",
        "--backend-store-uri",
        backend_store_uri,
        "--default-artifact-root",
        default_artifact_root,
    ]

    process = Popen(cmd)

    # wait for initialization
    time.sleep(3)

    try:
        yield
    finally:
        kill_process_and_children(process.pid)
