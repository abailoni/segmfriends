from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


__version__ = '0.1.1'


extensions = [
    Extension("segmfriends.features.mappings_CY", ["./segmfriends/features/mappings_CY.pyx"],
        include_dirs=[numpy.get_include()],
        # libraries=[...],
        # library_dirs=[...],
              ),
    Extension("segmfriends.transform.combine_segms_CY", ["./segmfriends/transform/combine_segms_CY.pyx"],
        include_dirs=[numpy.get_include()]),
]


setup(
    name='Segmentation friends',
    version=__version__,
    packages=['segmfriends', ],
    description='Some segmentation tools',
    author='Alberto Bailoni',
    url='https://github.com/abailoni/segmfriends',
    long_description='',
    author_email='alberto.bailoni@iwr.uni-heidelberg.de',
    ext_modules = cythonize(extensions),
    # include_dirs=[numpy.get_include()]
)


# setup(name='Neuro-skunkworks',
#       packages=['skunkworks',],
#      )
