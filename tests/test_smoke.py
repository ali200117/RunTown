"""Smoke test: proves the package is importable and the test setup works.

This is deliberately trivial. Its job is to fail loudly if the project layout,
the virtual environment or the editable install is broken - long before any
real logic depends on them.
"""

import kinetirun


def test_package_is_importable():
    assert kinetirun.__version__ == "0.1.0"
