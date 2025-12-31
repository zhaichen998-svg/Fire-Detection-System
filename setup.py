from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fire-detection-system",
    version="1.0.0",
    author="Fire Detection Team",
    description="PyTorch-based fire detection system with baseline and improvements",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhaichen998-svg/Fire-Detection-System",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
