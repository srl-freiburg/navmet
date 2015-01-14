from setuptools import setup

import os

if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

setup(name="sadf",
      version="0.2",
      install_requires=["numpy"],
      extras_require={
          'plotting': ["matplotlib"]
      },
      packages=['navmet'],
      include_package_data=True,
      description="Robot navigation metrics for social compliance",
      author="Billy Okal",
      author_email="okal@cs.uni-freiburg.de",
      url="http://srl.informatik.uni-freiburg.de",
      license="BSD 2-clause",
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
      )
