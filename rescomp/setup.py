import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rescomp-djpassey", # Replace with your own username
    version="0.0.1",
    author="DJ Passey",
    author_email="djpasseyjr@gmail.com",
    description="A reservoir computers with graph specialization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/djpasseyjr/ReservoirSpecialization",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
