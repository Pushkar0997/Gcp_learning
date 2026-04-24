from setuptools import setup, find_packages

setup(
    name="veritas_trainer_package",
    version="0.1",
    packages=find_packages(),
    package_data={
        "trainer": ["accelerate_config.yaml"],
    },
    install_requires=[
        "pandas>=1.5.0,<2.3.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.0.0",
        "gcsfs>=2023.1.0",
        "fsspec>=2023.1.0",
        "evaluate>=0.4.0",
        "transformers==4.38.1",
        "datasets==2.18.0",
        "accelerate==0.28.0",
        "torch>=2.0.0",
        "python-json-logger>=2.0.0",
    ],
)