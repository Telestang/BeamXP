"""A second attempt at glyph detection, built beside the first rather than in it.

Nothing here is imported by ``beamxp`` or by ``mesh_segmentation_transform``.
The dependency runs one way only -- ``glyph_lab.corpus`` borrows the existing
archive/DAE resolution to *extract* a corpus, once, offline -- so that the
pipeline and its tuning app can be rebuilt without disturbing the shipping one.

The intent is that working parts graduate into the app later and supersede the
current detection; until then this is a sandbox, in the same sense
``spatial_classifier`` is.
"""
