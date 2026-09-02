"""Glyph detection: the front end the texture-correction exporter runs on.

Built beside the first pipeline rather than in it, and as of 2026-09-02 it is
the one production uses.  ``RhdTextureConfig.detector`` selects between them
and defaults to ``"glyph_lab"``; ``"legacy"`` still reaches the MSER/contrast
pipeline in ``mesh_segmentation_transform.annotate_texture_regions``, which the
tuning harness continues to drive, and which is kept reachable for one release
so a conversion that regresses can be compared against what produced it.

The order is the point of the rebuild: threshold, throw out texels that were
never marks, and only then group what is left.  The first pipeline grouped
first and argued afterwards, and a component welded together by noise cannot be
taken apart by any later stage.  Detection runs per coalesced UV domain, in
numpy and scipy only -- no cv2, no GPU.

Which way the imports run, and why it matters:

* ``mesh_segmentation_transform.glyph_lab_detection`` is the only module on the
  export side that imports this package.  It builds a ``CorpusTexture`` in
  memory through ``glyph_lab.source`` and hands back rectangles, texels and an
  optional fitted shape.
* ``glyph_lab.corpus`` and ``scripts/build_glyph_corpus.py`` borrow the
  existing archive/DAE resolution to *extract* a fixed corpus, once, offline.
  That is a script-time dependency and nothing in detection shares it.
* ``glyph_lab.uv_geometry`` reads three defaults off ``RhdTextureConfig`` for
  the viewer's chart-frame overlay, so **no module reachable from ``detect``
  or ``detect_normal`` may import it** -- that would close a cycle through the
  export path.  ``glyph_lab.hulls`` exists for exactly this reason: it holds
  the hull helpers the detector needs, free of any configuration.
"""
