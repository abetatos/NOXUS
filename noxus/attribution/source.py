"""Source attribution of the NO2 column to the cluster.

The technique exists in the atmospheric literature (point-source NOx catalogues derived from the
divergence of the TROPOMI NO2 flux). Attribution is only reliable for large, isolated emitters;
co-located traffic, heating, and other industry confound the steel-specific component, so the
attributed quantity is treated as a relative activity proxy, not an emissions estimate.
"""

from __future__ import annotations


def attribute(no2_field):
    """Reduce a gridded NO2 field over the region to a per-overpass activity proxy.

    Not yet implemented — this is a scaffold. The implementation will mask to the cluster footprint,
    estimate and subtract a regional background, and integrate the residual column as the proxy.
    """
    raise NotImplementedError("Source attribution is not yet implemented")
