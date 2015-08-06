from setuptools import setup
import numpy as np

import os

if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

setup(name="navmet",
      version="0.1.0",
      install_requires=["numpy", "scipy", "matplotlib"],
      packages=['navmet', 'navmet.tests'],
      include_package_data=True,
      description="Robot Navigation Metrics",
      author="Billy Okal",
      author_email="sudo@makokal.com",
      url="https://github.com/srl-freiburg/navmet",
      license="MIT",
      use_2to3=True,
      classifiers=['Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX',
                   'Operating System :: Unix',
                   'Operating System :: MacOS',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.6',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.3',
                   ],
      include_dirs=[np.get_include()]
      )
