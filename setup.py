import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gpt-neox-igor",
    version="0.0.1",
    author="Igor Ostrovsky",
    author_email="igoros@gmail.com",
    description="GPT-NeoX (Igor's Version)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/igor0/gpt-neox",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/gpt-neox/issues",
    },
    package_dir={"": "megatron"},
    packages=setuptools.find_packages(where="megatron"),
    python_requires=">=3.6",
)
