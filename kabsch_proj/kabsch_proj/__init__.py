"""
kabsch_proj
The Kabsch algorithm, is a method for calculating the optimal rotation matrix that minimizes the RMSD (root mean squared deviation) between two paired sets of points.
"""

# Make Python 2 and 3 imports work the same
# Safe to remove with Python 3-only code
from __future__ import absolute_import

# Add imports here
from .kabsch_algorithm import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
