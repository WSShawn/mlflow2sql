import re
import ast
from glob import glob
from os.path import basename, splitext

from setuptools import find_packages
from setuptools import setup

_version_re = re.compile(r"__version__\s+=\s+(.*)")

with open("src/mlflow2sql/__init__.py", "rb") as f:
    VERSION = str(
        ast.literal_eval(_version_re.search(f.read().decode("utf-8")).group(1))
    )

REQUIRES = ["pyyaml"]

DEV = [
    "pytest",
    "flake8",
    "invoke",
    "twine",
    # for testing (since we need to import generate.py)
    "mlflow",
    "numpy",
    "psutil",
    "matplotlib",
]

DEMO = [
    "scikit-learn",
    "matplotlib",
    "sklearn-evaluation",
    "mlflow",
]

setup(
    name="mlflow2sql",
    version=VERSION,
    description=None,
    license=None,
    author=None,
    author_email=None,
    url=None,
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    classifiers=[],
    keywords=[],
    install_requires=REQUIRES,
    extras_require={
        "dev": DEV,
        "demo": DEMO,
    },
    entry_points={
        # 'console_scripts': ['mlflow2sql=mlflow2sql.cli:cli'],
    },
)
