from setuptools import setup, find_packages

setup(
    name="ddlp",
    version="0.1.0",
    author="Samuel Nordmann",
    author_email="snordmann@nvidia.com",
    description="Distributed Deep Learning Primitives",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["torch"],
    extras_require={
        "test": ["pytest>=6.0"],
        "fuser": ["nvfuser_direct"],
        "te": ["transformer_engine"],
    },
)
