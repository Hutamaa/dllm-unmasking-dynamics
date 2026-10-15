"""Package declaration for editable research checkouts."""
from setuptools import setup

setup(
    name='dllm-unmasking-dynamics',
    version='0.1.0',
    description='Matched decoding experiments in masked diffusion language models',
    python_requires='>=3.9',
    packages=['dllm_unmasking', 'experiments'] + [f'experiments.phase{i}' for i in range(1, 7)],
    package_dir={'dllm_unmasking': 'src/dllm_unmasking', 'experiments': 'experiments'},
    package_data={'dllm_unmasking': ['source_layout.json']},
    # CPU/CUDA dependencies are intentionally installed from separate requirements.
)
