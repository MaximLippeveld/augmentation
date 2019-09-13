import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="augmentation",
    version="0.4",
    author="Joris Roels",
    author_email="jorisb.roels@ugent.be",
    description="A library, based on PyTorch, that performs data augmentation on the GPU",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JorisRoels/augmentation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)