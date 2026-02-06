from setuptools import setup, find_packages

with open("STARRFISH_API_Documentation.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="STARRFISH",
    version="1.0.0",
    author="Guojie Zhong",
    author_email="guojiezhong@example.com",
    description="Single-cell Transcriptomic And Regulatory Region Readout For Identifying Specificity of cis-regulatory elements",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/starr-fish",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scanpy>=1.8.0",
        "anndata>=0.8.0",
        "scikit-learn>=0.24.0",
        "statsmodels>=0.13.0",
        "pybedtools>=0.9.0",
        "pysam>=0.19.0",
        "zarr>=2.10.0",
        "scvi-tools>=0.16.0",
    ],
    extras_require={
        "stan": [
            "pystan>=3.0.0",
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
)
