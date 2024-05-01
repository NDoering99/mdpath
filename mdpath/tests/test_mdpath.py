"""
Unit and regression test for the mdpath package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import mdpath


def test_mdpath_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "mdpath" in sys.modules
