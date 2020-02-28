#!/usr/bin/env python
from distutils.core import setup

setup(
   name='image_processing_course',
   version='1.0',
   description='Code for Image processing and vision systems course @PUT',
   author='Michał Fularz, PUTvision',
   author_email='michal.fularz@put.poznan.pl',
   packages=['image_processing_course'],
   install_requires=[
       'opencv-python',
       'numpy',
       'scikit-image',
       'matplotlib'
   ],
)
