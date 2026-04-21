from setuptools import setup, find_packages

setup(
    name="veritas_trainer_package",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib",
        "gcsfs",
        "fsspec",
        "evaluate",
        "transformers==4.38.1",
        "datasets==2.18.0",
        "accelerate==0.28.0",
        "peft==0.9.0",
        "torch",
    ],
)