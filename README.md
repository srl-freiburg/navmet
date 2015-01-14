`navmet` Robot Navigation Metrics
======================================

A set of objective and subjective metrics for evaluating robot navigation behavior especially in crowded environments where socially compliant behavior is required.

Features
-----------
- Objective metrics based on trajectory geometry
  - Path length
  - Cumulative heading changes
- Subjective metrics inspired from computational social sciences
  - Intrusion counts into various spaces as defined by Proxemics, i.e Intimate, Personal, Social and any custom defined ranges
  - Two ways of counting intrusions:
      - Uniform circles
      - Anisotropic regions

Requirements
---------------
1. Numpy
2. Matplotlib, optional (only used in examples)

Installation
--------------
```sh
git clone https://github.com/makokal/navmet.git
cd navmet
[sudo] python setup.py install
```

Or simply add the path (useful for devel sessions)


Usage
--------
See `examples` folder



Roadmap
---------
[ ] Adding more objective and subjectibve metrics
[ ] Various timings for timestamped trajectories
[ ] 'Energy type' metrics
[ ] Elliptical regions as in Proxemics

