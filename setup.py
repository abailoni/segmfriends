import setuptools
from distutils.extension import Extension
# from Cython.Build import cythonize
import numpy


exec(open('segmfriends/__version__.py').read())




setuptools.setup(
    name='segmfriends',
    version=__version__,
    description='Some segmentation tools',
    author='Alberto Bailoni',
    url='https://github.com/abailoni/segmfriends',
    long_description='',
    # ext_modules = cythonize(extensions),
    # include_dirs=[numpy.get_include()]
    packages=setuptools.find_packages(),
    # packages=['segmfriends', ],
)
