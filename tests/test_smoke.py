"""Smoke tests for the scaffold: package imports and config are sane."""

from noxus import __version__
from noxus.config import TANGSHAN


def test_version_present():
    assert __version__


def test_tangshan_bbox_is_well_formed():
    min_lon, min_lat, max_lon, max_lat = TANGSHAN.as_bbox()
    assert min_lon < max_lon
    assert min_lat < max_lat
    assert TANGSHAN.name == "tangshan"
