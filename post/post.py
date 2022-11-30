# %% [raw]
# title: "[]: Query your MLFlow experiments with SQL"

# %% [markdown]
# In this blog post, I'll show you how you can export your MLFlow data to a SQLite database so you can easily filter, search and aggregate all your Machine Learning experiments with SQL.
#
# TODO: add link to notebook version of this post
# TODO: add link to download sample data
#
# ## The problem with MLFlow
#
# MLFlow is one of the most widely used tools for experiment tracking, yet, its features for comparing experiments are severely limited. One basic operation I do with my Machine Learning experiments is to put plots side by side (e.g., confusion matrices or ROC curves); this simple operation isn't possible in MLFlow ([the issue](https://github.com/mlflow/mlflow/issues/2696) has been open since April 2020).
#
# Another critical limitation are its filtering and search capabilities. MLFlow defines its own filter DSL which is a limited version of a [SQL WHERE clause](https://www.mlflow.org/docs/latest/search-runs.html). Even though MLFlow supports several SQL databases as backend, you have to use this limited DSL which only allows basic filtering but does not support aggregation or any other operation that you can trivially do with some lines of SQL. For me, this is a major drawback since often I want to perform more than a basic filter.
#
# So let's see how we can get our data out of MLFlow and into a SQLite database so we can easily explore our Machine Learning experiments! Under the hood, this will use the [SQLiteTracker](https://sklearn-evaluation.readthedocs.io/en/latest/user_guide/SQLiteTracker.html) from the [sklearn-evaluation](https://github.com/ploomber/sklearn-evaluation) package.
#
# ## Setting up
#
# The code is in a Python package that you can install with `pip`:
#
# ```sh
# pip install mlflow2sql
# ```

# %% [markdown]
# ## Optional: Generate sample data
#
# If you don't have some sample MLFlow data, I wrote some utility functions to generate a few experiments and log them. However, you need to install a few extra dependencies:
#
# ```sh
# pip install "mlflow2sql[demo]"
# ```

# %% [markdown]
# If you're running this from a notebook, you can run the following cel to install the dependencies:

# %%
# %pip install "mlflow2sql[demo]" --quiet

# %% [markdown]
# To generate some sample MLFlow data, run the following in a Python session (this will create the data in the `./mlflow-data` directory), this will take a few minutes:

# %%
import shutil
from pathlib import Path

from mlflow2sql import generate, export, ml

# clean up the directory, if any
path = Path("mlflow-data")
if path.exists():
    shutil.rmtree(path)    

# generate sample data
with generate.start_mlflow(backend_store_uri="mlflow-data",
                           default_artifact_root="mlflow-data"):
    ml.run_default()

# %% [markdown]
# ## Getting all MLFlow runs
#
# Once you have some MLFlow data, start a Python session, define the `BACKEND_STORE_URI` and `DEFAULT_ARTIFACT_ROOT` variables in the next code snippet.
#
#
# Note that the two variables correspond to the two arguments with the same name that you pass when starting the MLFlow server:
#
# ```sh
# mlflow server --backend-store-uri BACKEND_STORE_URI --default-artifact-root DEFAULT_ARTIFACT_ROOT
# ```
#
# *Note:* If you didn't pass those arguments when running `mlflow server`, the default value is `./mlruns`.

# %%
from mlflow2sql import export, ml

# these two variables are ./mlruns by default
BACKEND_STORE_URI = "mlflow-data"
DEFAULT_ARTIFACT_ROOT = "mlflow-data"

# note: this was tested with MLFlow versions 1.30.0 and 2.0.1
runs = export.find_runs(backend_store_uri=BACKEND_STORE_URI,
                        default_artifact_root=DEFAULT_ARTIFACT_ROOT)

# %% [markdown]
# We can get the number of experiments we extracted:

# %%
len(runs)

# %% [markdown]
# ## Importing into a SQLite database
#
# Let's now initialize our experiment tracker ([documentation available here](https://sklearn-evaluation.readthedocs.io/en/latest/user_guide/SQLiteTracker.html)), and insert our MLFlow experiments:

# %%
from sklearn_evaluation import SQLiteTracker

# clean up the database, if any
db = Path("experiments.db")

if db.exists():
    db.unlink()

tracker = SQLiteTracker('experiments.db')
tracker.insert_many((run.to_dict() for run in runs))

# %% [markdown]
# Let's check the number of extracted experiments:

# %%
len(tracker)

# %% [markdown]
# Great! We got all of them in the database. Let's start exploring them with SQL!
#
# ## Querying experiments with SQL
#
# `SQLiteTracker` creates a table with all our experiments and it stores the logged data in a JSON object with three keys: `metrics`, `params`, and `artifacts`:

# %%
tracker.query("""
SELECT
    uuid,
    json_extract(parameters, '$.metrics') as metrics,
    json_extract(parameters, '$.params') as params,
    json_extract(parameters, '$.artifacts') as artifacts
    FROM experiments
LIMIT 3
""")

# %% [markdown]
# We can extract the metrics easily. Note that MLFlow stores metrics in the following format:
#
# ```txt
# (timestamp 1) (value 1)
# (timestamp 2) (value 2)
# ```
#
# Hence, when parsing it, we create two lists. One with the timestamps and another one with the values:
#
# ```python
# [[ts1, ts2], [val1, val2]]
# ```
#
# Let's extract the metrics values and sort by F1 score, note that `.query()` returns a `pandas.DataFrame` by default:

# %%
tracker.query("""
SELECT
    uuid,
    json_extract(parameters, '$.metrics.f1[1][0]') as f1,
    json_extract(parameters, '$.metrics.precision[1][0]') as precision,
    json_extract(parameters, '$.metrics.recall[1][0]') as recall,
    json_extract(parameters, '$.params.model_name') as model_name
FROM experiments
ORDER BY f1 DESC
LIMIT 10
""")

# %% [markdown]
# We can also extract the artifacts and display them inline by passing `as_frame=False` and `render_plots=True`:

# %%
results = tracker.query("""
SELECT
    uuid,
    json_extract(parameters, '$.metrics.f1[1][0]') as f1,
    json_extract(parameters, '$.params.model_name') as model_name,
    json_extract(parameters, '$.metrics.precision[1][0]') as precision,
    json_extract(parameters, '$.metrics.recall[1][0]') as recall,
    json_extract(parameters, '$.artifacts.confusion_matrix_png') as confusion_matrix,
    json_extract(parameters, '$.artifacts.precision_recall_png') as precision_recall
FROM experiments
ORDER BY f1 DESC
LIMIT 2
""", as_frame=False, render_plots=True)

results

# %% [markdown]
# ## Displaying image artifacts
#
# Looking at the plots is a bit difficult in the table view, but we can zoom in and extract them. Let's get a tab view from our top two experiments. Here are the confusion matrices:

# %%
results.get("confusion_matrix")

# %% [markdown]
# And here's the precision-recall curve:

# %%
results.get("precision_recall")

# %% [markdown]
# ## Filtering and sorting
#
# In our generated experiments, we trained some Support Vector Machines, Gradient Boosting and Random Forest. Let's filter by Random Forest, sort by F1 score and extract their parameters. We can easily do this with SQL!

# %%
tracker.query("""
SELECT
    uuid,
    json_extract(parameters, '$.metrics.f1[1][0]') as f1,
    json_extract(parameters, '$.params.model_name') as model_name,
    json_extract(parameters, '$.metrics.precision[1][0]') as precision,
    json_extract(parameters, '$.metrics.recall[1][0]') as recall,
    json_extract(parameters, '$.params.max_depth') as max_depth,
    json_extract(parameters, '$.params.n_estimators') as n_estimators,
    json_extract(parameters, '$.params.criterion') as criterion
FROM experiments
WHERE model_name = 'RandomForestClassifier'
ORDER BY f1 DESC
LIMIT 3
""", as_frame=False, render_plots=True)


# %% [markdown]
# ## Aggregating and plotting
#
# Finally, let's do something a bit more sophisticated. Let's get all the Random Forests, group by number of trees (`n_estimators`), and take the mean of our metrics. This will allow us to see what's the effect of increasing the number of trees in the model's perfomance:

# %%
df = tracker.query(
    """
SELECT
    json_extract(parameters, '$.params.n_estimators') as n_estimators,
    AVG(json_extract(parameters, '$.metrics.precision[1][0]')) as precision,
    AVG(json_extract(parameters, '$.metrics.recall[1][0]')) as recall,
    AVG(json_extract(parameters, '$.metrics.f1[1][0]')) as f1
FROM experiments
WHERE json_extract(parameters, '$.params.model_name') = 'RandomForestClassifier'
GROUP BY n_estimators
""",
    as_frame=True,
).set_index("n_estimators")

df

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

for metric in ["f1", "recall", "precision"]:
    df[metric].plot(ax=ax, marker="o", style="--")

ax.legend()
ax.grid()
ax.set_title("Effect of increasing n_estimators")
ax.set_ylabel("Metric")
_ = ax.set_xticks(df.index)

# %% [markdown]
# We can see that the increase in performance diminishes rapidly, after 50 estimators, there isn't much performance gain. This analysis allows us to find these type of insights and focus on other parameters for improving performance.
#
# Finally, let's group by model type and see how our models are doing on average:

# %%
df = tracker.query(
    """
SELECT
    json_extract(parameters, '$.params.model_name') as model_name,
    AVG(json_extract(parameters, '$.metrics.precision[1][0]')) as precision,
    AVG(json_extract(parameters, '$.metrics.recall[1][0]')) as recall,
    AVG(json_extract(parameters, '$.metrics.f1[1][0]')) as f1
FROM experiments
GROUP BY model_name
""",
    as_frame=True,
).set_index("model_name")

ax = plt.gca()
df.plot(kind="barh", ax=ax)
ax.grid()

# %% [markdown]
# We see that Random Forest and Gradient Boosting have comparable performance, on average (although take into account that we ran more Random Forest experiments), and SVC has lower performance.
#
# ## Closing remarks
#
# In this blog post, we showed how to export MLFlow data to a SQLite database, which allows us to use SQL to explore, aggregate and analyze our Machine Learning experiments, a lot better than MLFlow's limited querying capabilities!
#
# There are a few limitations to this first implementation, it'll only work if you're using the local filesystem for storing your MLFlow experiments and artifacts. And only `.txt`, `.json` and `.png` artifacts are supported. If you have any suggestions, feel free to open an issue on [GitHub](https://github.com/ploomber/mlflow2sql), or send us a message on [Slack!](https://ploomber.io/community/)
