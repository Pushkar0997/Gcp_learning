from setuptools import setup, find_packages

setup(
    name="veritas_trainer_package",
    version="0.1",
    packages=find_packages(),
    package_data={
        "trainer": ["accelerate_config.yaml"],
    },
    install_requires=[
        # ── Data / numerics ───────────────────────────────────────────────
        "pandas>=1.5.0,<2.3.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.0.0",

        # ── GCS access ────────────────────────────────────────────────────
        "gcsfs>=2023.1.0",
        "fsspec>=2023.1.0",

        # ── HuggingFace stack  (versions are tightly coupled — do not bump) ──
        # huggingface_hub: transformers 4.38.1 requires <1.0; without this pin
        # pip may install hub 1.x which breaks the transformers import entirely.
        "huggingface_hub>=0.19.3,<1.0",
        # tokenizers: Longformer tokenizer requires >=0.13; cap at <0.20 to
        # avoid surprise API changes that broke training in earlier runs.
        "tokenizers>=0.13.0,<0.20",
        "transformers==4.38.1",
        "datasets==2.18.0",
        "accelerate==0.28.0",
        "evaluate>=0.4.0",

        # ── PyTorch (pre-installed in Vertex GPU container, listed for safety) ─
        "torch>=2.0.0",

        # ── Misc ──────────────────────────────────────────────────────────
        "tqdm>=4.64.0",
        "python-json-logger",
    ],
)