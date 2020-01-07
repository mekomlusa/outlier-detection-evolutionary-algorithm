import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="evood",
    version="0.0.1",
    author="Rose Lin",
    author_email="rl5vh@virginia.edu",
    description="A simple tool for outlier detection in high dimension",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mekomlusa/outlier-detection-evolutionary-algorithm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)