import base64
import ast
from glob import iglob
from pathlib import Path
import json
import warnings

import yaml


# %%
def get_ids(pattern):
    return {Path(path).name for path in iglob(pattern) if Path(path).is_dir()}


def _process_path(path):
    path = Path(path)
    return (path.name, Path(path).read_text())


def _process_metric_line(line):
    timestamp, value, _ = line.split(" ")
    return timestamp, value


def _process_metric(text):
    lines = text.splitlines()
    timestamps, values = zip(*(_process_metric_line(line) for line in lines))
    return timestamps, values


def _process_artifact(path):
    path = Path(path)

    if path.suffix == ".png":
        return path.name, _process_png(path)
    elif path.suffix == ".txt":
        return path.name, path.read_text()
    elif path.suffix == ".json":
        return path.name, json.loads(path.read_text())
    else:
        warnings.warn(f"Unsupported artifact: {path!s}. It'll be ignored...")
        return path.name, None


def _process_png(path):
    bytes = path.read_bytes()
    image_base64 = base64.encodebytes(bytes)
    html = (
        '<img src="data:image/png;base64,' + image_base64.decode("utf-8") + '"></img>'
    )
    return html


def _safe_literal_eval(source):
    try:
        return ast.literal_eval(source)
    # if it's a string...
    except (SyntaxError, ValueError):
        return source.strip()


# %%
class MlFlowRun:
    """A run captures a set of parameters, metrics and artifacts"""

    def __init__(
        self,
        backend_store_uri,
        default_artifact_root,
        run_id,
        experiment_id,
        experiment_name,
    ) -> None:
        self._backend_store_uri = backend_store_uri
        self._default_artifact_root = default_artifact_root
        self._run_id = run_id
        self._experiment_id = experiment_id
        self._experiment_name = experiment_name

    def get_metrics(self):
        return {
            k: _process_metric(v)
            for k, v in self._get_from_path(self._backend_store_uri, "metrics").items()
        }

    def get_params(self):
        return {
            k: _safe_literal_eval(v)
            for k, v in self._get_from_path(self._backend_store_uri, "params").items()
        }

    def get_artifacts(self):
        return self._get_from_path(
            self._default_artifact_root,
            "artifacts",
            processor=_process_artifact,
        )

    def _get_from_path(self, parent, path, processor=None):
        processor = processor or _process_path
        pairs = [
            processor(path)
            for path in iglob(
                f"{parent}/{self._experiment_id}/{self._run_id}/{path}/**/*",
                recursive=True,
            )
        ]
        return {
            pair[0].replace(".", "_"): pair[1] for pair in pairs if pair[1] is not None
        }

    def to_dict(self):
        return {
            "params": self.get_params(),
            "metrics": self.get_metrics(),
            "artifacts": self.get_artifacts(),
            "run_id": self._run_id,
            "experiment_id": self._experiment_id,
            "experiment_name": self._experiment_name,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._experiment_id!r}, {self._run_id!r})"


class MLFlowExperiment:
    """An experiment groups runs"""

    def __init__(self, backend_store_uri, default_artifact_root, id) -> None:
        self._backend_store_uri = backend_store_uri
        self._default_artifact_root = default_artifact_root
        self._id = id
        self._name = self._get_name()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(backend_store_uri="
            f"{self._backend_store_uri!r}, default_artifact_root="
            f"{self._default_artifact_root}, id={self._id!r})"
        )

    def find_runs(self):
        """Return a list with all MLFlow runs found"""
        runs = get_ids(f"{self._backend_store_uri}/{self._id}/**")
        artifacts = get_ids(f"{self._default_artifact_root}/{self._id}/**")
        ids = runs | artifacts

        return [
            MlFlowRun(
                backend_store_uri=self._backend_store_uri,
                default_artifact_root=self._default_artifact_root,
                run_id=id,
                experiment_id=self._id,
                experiment_name=self._name,
            )
            for id in ids
        ]

    def _get_name(self):
        path = Path(self._backend_store_uri, self._id, "meta.yaml")
        if path.exists():
            return yaml.safe_load(path.read_text()).get("name")
        else:
            return None


def find_experiments(backend_store_uri, default_artifact_root):
    """Return a list with all MLFlow experiments found"""
    runs = get_ids(f"{backend_store_uri}/*")
    artifacts = get_ids(f"{default_artifact_root}/*")
    ids = runs | artifacts
    return [
        MLFlowExperiment(
            backend_store_uri=backend_store_uri,
            default_artifact_root=default_artifact_root,
            id=id,
        )
        for id in ids
    ]


def find_runs(backend_store_uri, default_artifact_root):
    experiments = find_experiments(
        backend_store_uri=backend_store_uri, default_artifact_root=default_artifact_root
    )
    runs = [experiment.find_runs() for experiment in experiments]
    runs = [i for sub in runs for i in sub]
    return runs
