from setuptools import setup, find_packages

setup(
    name="attention_saver",
    version="0.1.0",
    description="Extraction of ultra-long-context attention matrices and statistics for any HuggingFace LLMs",
    author="Mathis Pink",
    author_email="mpink@mpi-sws.org",
    url="https://github.com/MathisPink/attention_saver",  # update with your repo
    packages=find_packages(),
    install_requires=[
        "torch",
        "h5py",
        "numpy",
        "transformers"
    ],
    python_requires=">=3.8",
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
)