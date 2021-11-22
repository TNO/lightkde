# Do not edit the version number manually, it is automatically updated using a
# semantic release package: ````python-semantic-release````. If you need to indicate the
# version somewhere, e.g. documentation, then import it from here.

__version__ = "1.0.1"

# for convenience
from lightkde.lightkde import kde_1d, kde_2d

__all__ = ["kde_1d", "kde_2d"]
