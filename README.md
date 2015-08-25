Robot Navigation Metrics
============================

A set of objective and subjective metrics for evaluating robot navigation behavior especially in crowded environments where socially compliant behavior is required.

Features
-----------
- Objective metrics based on trajectory geometry
  - Path length
  - Cumulative heading changes
  - Path similarity (based on a refence trajectory)
- Subjective metrics inspired from computational social sciences
  - Intrusion counts into various spaces as defined by Proxemics, i.e intimate, personal, social or any custom defined regions
  - Two ways of counting intrusions:
      - Uniform circles
      - Anisotropic regions
  - Relation disturbance (robot crossing relation links between people, e.g. a group)

Requirements
---------------
1. Numpy
2. Matplotlib, optional (only used in examples)

Installation
--------------
```sh
git clone https://github.com/makokal/navmet.git
cd navmet
python setup.py build
python setup.py develop  # for local devel install
[sudo] python setup.py install  # for global install
```

Usage
--------
See `examples` folder


Roadmap
---------
- [ ] Adding more objective and subjectibve metrics
- [ ] 'Energy type' metrics
- [ ] Elliptical regions as in Proxemics

