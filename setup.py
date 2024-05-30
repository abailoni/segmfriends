from setuptools import setup
from distutils.extension import Extension
# from Cython.Build import cythonize
import numpy


exec(open('segmfriends/__version__.py').read())


# extensions = [
#     Extension("segmfriends.transform.combine_segms_CY", ["./segmfriends/transform/combine_segms_CY.pyx"],
#         include_dirs=[numpy.get_include()]),
# ]


setup(
    name='Segmentation friends',
    version=__version__,
    packages=['segmfriends', ],
    description='Some segmentation tools',
    author='Alberto Bailoni',
    url='https://github.com/abailoni/segmfriends',
    long_description='',
    # ext_modules = cythonize(extensions),
    # include_dirs=[numpy.get_include()]
)
