"""
RatingSystems

A package with several rating systems for players and/or teams. Works with teams 
of any size and matches with any number of teams.

The systems currently supported are:
- Elo: https://en.wikipedia.org/wiki/Elo_rating_system
- Glicko: http://www.glicko.net/glicko.html
- Glicko-2: http://www.glicko.net/glicko.html
- OpenSkill: https://openskill.me/en/stable/
- TrueSkill: https://www.microsoft.com/en-us/research/project/trueskill-ranking-system
  (uses the implementation by Heungsub Lee: https://trueskill.org/) 
- JRS: No reference yet.
"""

from setuptools import find_packages,setup
import os

__dir__ = os.path.dirname(__file__)
about = {}
with open(os.path.join(__dir__, 'ratingsystems', '__about__.py')) as f:
    exec(f.read(), about)
    

setup(name='ratingsystems',
      version=about['__version__'],
      # license=about['__license__'],
      description=about['__description__'],
      author=about['__author__'],
      author_email=about['__author_email__'],
      url=about['__url__'],
      packages=find_packages(where="./ratingsystems/"),
      install_requires=[
      'numpy',
      'scipy',
      'openskill',
      'pandas',
      'tqdm',
      'trueskill',
      ]
     )