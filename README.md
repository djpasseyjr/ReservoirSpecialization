# Reservoir Specialization Package
Reservoir computer class with functionality to specialize and copy the top preforming subnetworks.

## Installation Instructions

To install the `rescomp` package included in this repository, clone the repository and add the repository directory to your `PYTHONPATH` so that Python can locate the package. After doing this, you can include the package classes and functions with `import rescomp as rc` from any directory.

**Clone the repo**
```
git clone https://github.com/djpasseyjr/ReservoirSpecialization.git
```
**Add repo directory to PYTHONPATH**

Open your `.bashrc` or `.bash_profile` file. (It should be located in your home directory for Linux or Mac. I'm not sure about Windows) Add the following line:
```
export PYTHONPATH="${PYTHONPATH}:/path/to/repo/ReservoirSpecialization"
```
Restart your bash session. After this, the following code should work:
```
$ python
Python 3.7.4 (default, Aug 13 2019, 15:17:50) 
[Clang 4.0.1 (tags/RELEASE_401/final)] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import rescomp as rc
>>> 
```
