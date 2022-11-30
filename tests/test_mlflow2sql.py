from pathlib import Path
from unittest.mock import ANY

import numpy as np
import mlflow

from mlflow2sql import generate, export


def test_basic(tmp_mlflow_empty):
    generate.runs()

    experiments = export.find_experiments(
        backend_store_uri=".", default_artifact_root="."
    )

    exp = experiments[0]
    runs = exp.find_runs()
    run = runs[0]
    run_ = run.to_dict()

    # mlflow creates a "models" experiment
    assert len(experiments) == 2
    assert len(runs) == 10

    assert run_["params"] == {"param_number": 42}
    assert run_["metrics"] == {
        "metric_multiple_increment": (
            (ANY, ANY),
            ("1.0", "2.1"),
        ),
        "metric_number": ((ANY,), ("2.0",)),
    }
    assert set(run_["artifacts"]) == {"scatter_png"}


def test_nested_artifact(tmp_mlflow_empty):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("some-experiment")

    with mlflow.start_run():
        mlflow.log_text("some text", "text/text.txt")
        mlflow.log_text("more text", "some/nested/tex/t.txt")

    runs = export.find_runs(backend_store_uri=".", default_artifact_root=".")
    run = runs[0]
    run_ = run.to_dict()

    assert not run_["params"]
    assert not run_["metrics"]
    assert run_["run_id"]
    assert run_["experiment_id"]
    assert run_["experiment_name"] == "some-experiment"
    assert run_["artifacts"] == {"text_txt": "some text", "t_txt": "more text"}


def test_other_log_methods(tmp_mlflow_empty):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("some-experiment")

    Path("file.txt").write_text("content")
    image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    with mlflow.start_run():
        mlflow.log_artifact("file.txt")
        mlflow.log_dict(dict(a=1, b=2), "dictionary.json")
        mlflow.log_image(image, "image.png")
        mlflow.log_text("some text", "text.txt")

    runs = export.find_runs(backend_store_uri=".", default_artifact_root=".")
    run = runs[0]
    run_ = run.to_dict()

    assert not run_["params"]
    assert not run_["metrics"]
    assert run_["run_id"]
    assert run_["experiment_id"]
    assert run_["experiment_name"] == "some-experiment"
    assert run_["artifacts"]["file_txt"] == "content"
    assert run_["artifacts"]["dictionary_json"] == dict(a=1, b=2)
    assert run_["artifacts"]["text_txt"] == "some text"
    assert '<img src="data:image/png;base64' in run_["artifacts"]["image_png"]
