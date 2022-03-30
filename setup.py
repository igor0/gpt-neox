import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

PKG_DIR = os.path.dirname(os.path.abspath(__file__))

setuptools.setup(
    name="gpt-neox-igor",
    version="0.0.2",
    author="Igor Ostrovsky",
    author_email="igoros@gmail.com",
    description="GPT-NeoX (Igor's Version)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/igor0/gpt-neox",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/gpt-neox/issues",
    },
    packages=setuptools.find_packages(),
    install_requires=[
        "pybind11==2.6.2",
        "six",
        "regex",
        "numpy",
        "mpi4py==3.0.3",
        "wandb==0.10.28",
        "einops==0.3.0",
        "transformers==4.5.0",
        "tokenizers==0.10.2",
        "lm_dataformat==0.0.19",
        "ftfy==6.0.1",
        "deepspeed @ git+https://github.com/EleutherAI/DeeperSpeed.git@eb7f5cff36678625d23db8a8fe78b4a93e5d2c75",
        "lm-eval @ git+https://github.com/EleutherAI/lm-evaluation-harness.git@dc937d4b70af819c5695e09d94e59e4cdb1e40ad",
        "fused_kernels @ file://localhost{}/megatron/fused_kernels".format(PKG_DIR),
    ],
    python_requires=">=3.6",
)
