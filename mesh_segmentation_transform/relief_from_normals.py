#!/usr/bin/env python3
"""Render a tangent-space normal map as an image glyph detection can read.

Some marks are moulded into the trim and never printed on it -- the ardente's
"AIRBAG" and "ARDENTE" are both relief and nothing else -- so the colour map
cannot find them and a plan derived from colour alone leaves them backwards
when the geometry is mirrored.  This module makes the second detection source
those marks need.

What makes moulded lettering legible is the shadow it casts.  A stroke is not a
different colour from the trim and not, to look at, a different height: it is a
bright edge beside a dark one, and that is the whole of the signal.  So the
render lights the surface from a low angle and shows what the light does.  Flat
material returns a constant and cancels to zero; relief is whatever the light
made of it.

Two earlier renders are kept because they are useful to compare against, and
both fail for the same reason -- they discard the shading:

*   ``slope`` takes the magnitude of (x, y).  Magnitude has no sign, so the lit
    side and the shadowed side collapse together and a glyph comes out as a
    hollow outline.  It also puts grain above the marks: over the scintilla
    interior the 90th percentile of slope is 78 levels against 15-22 for a
    faint moulded edge.
*   ``height`` integrates the gradient field back to a surface
    (Frankot-Chellappa).  That is the right object mathematically and a mark
    does become a solid plateau, but a plateau is uniform inside, so the edges
    that carry the signal are exactly what it throws away.  Measured on the
    ardente, "ARDENTE" reached a prominence of 0.03 against its surround this
    way, against 0.30 shaded.

Panel form is removed before lighting rather than after.  The curve of a fascia
swings the normal much further than a letter does, so lighting the map as
authored mostly lights the dashboard; subtracting a blurred copy of x and y
leaves the local relief standing on flat ground.

The defaults are a starting point, not an answer.  Tune them in the harness:
    python mesh_segmentation_transform/annotate_texture_tuning_app.py
with "Detect on" set to Relief.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODE_SHADED = "shaded"
MODE_HEIGHT = "height"
MODE_SLOPE = "slope"
RELIEF_MODES = (MODE_SHADED, MODE_HEIGHT, MODE_SLOPE)


@dataclass(frozen=True, slots=True)
class ReliefConfig:
    """Everything tuneable about turning a normal map into a detectable image.

    The pipeline is: reconstruct (or not), suppress grain, remove panel form,
    divide by a scale, apply gain, optionally equalise.  Every stage can be
    switched off by setting its parameter to zero, so the raw slope render the
    first attempt used is still reachable.
    """

    # "shaded" lights the surface from a low angle and shows what the light
    # does.  "height" integrates to a plateau, which is uniform inside and so
    # throws the edges away.  "slope" gives magnitude without sign, so the
    # bright and dark sides collapse into one outline; that is the tuned normal
    # detection default because the downstream edge front-end works on outlines.
    mode: str = MODE_SLOPE

    # Where the light comes from.  Elevation is degrees above the surface, and
    # low is the point: a raking light casts long shadows off shallow relief,
    # which is what makes a mark that is a few levels deep visible at all.
    light_elevation_degrees: float = 25.0
    light_azimuth_degrees: float = 135.0
    # A single light leaves a stroke running along it unlit.  Above one, that
    # many lights are spread evenly around the compass and the strongest
    # response at each texel is kept, so no stroke is invisible for its
    # direction -- at the cost of the shadows reading less like one scene.
    light_count: int = 1
    # There is deliberately no contrast control here.  One was tried, applied
    # to the shading before the scaling below, and it did exactly nothing: the
    # divisor is a percentile of the same field, so multiplying the field
    # multiplies the divisor and the two cancel.  ``gain`` is the contrast
    # control, and it works because it is applied after the normalisation.

    # Low-pass applied first, in texture pixels.  Aimed at block-compression
    # dither and the finest grain, both of which sit at 1-2 px.  A glyph stroke
    # on a 4k interior atlas is 6-8 px, so a small sigma costs it little.
    grain_blur_sigma: float = 1.0

    # How the panel's own shape is taken out of the height field.
    #
    # "tophat" is the morphological one: the field minus its opening by a disc,
    # which keeps whatever is smaller than the disc and discards whatever is
    # larger.  It removes background by *size* rather than by frequency, so a
    # curved fascia goes without the lettering on it being attenuated -- the
    # tension a Gaussian high-pass cannot escape, where a sigma small enough to
    # flatten the panel is close enough to a 30 px letter to eat it.  This is
    # the standard tool for small bright structures on uneven background, and
    # it also leaves a stroke as a band of uniform height, which is what the
    # stroke-width transform needs and what a high-pass does not give.
    #
    # "highpass" is the Gaussian difference.  Top-hat discards everything
    # larger than its disc, which at a useful radius takes the soft shading
    # across a panel with it and leaves flat grey carrying only the finest
    # detail; the high-pass at a large sigma keeps that shading.  Top-hat
    # measures better on prominence -- 0.56 against 0.30 on the ardente's
    # ARDENTE -- but that is a ratio, and it improved partly because the
    # surround flattened, not because the mark got easier to see.
    form_removal: str = "highpass"
    # Radius of the disc, in texture pixels.  Must be larger than the stroke
    # width and smaller than the panel features: 6-8 px strokes and 30 px
    # letters on a 4k interior atlas put it comfortably in the teens.
    tophat_radius_px: int = 15
    # Raised marks survive a white top-hat and engraved ones a black top-hat.
    # "both" returns their difference, so raised comes out positive, engraved
    # negative and flat material zero -- the same signed convention the shaded
    # mode uses.
    tophat_polarity: str = "both"

    # High-pass, as the sigma of the blur that gets subtracted.  This is what
    # removes panel form -- the curve of the dash, the roll of a bolster --
    # while leaving marks behind.  It must stay clear of the mark's own size or
    # it takes the mark with it: ardente lettering is ~30 px tall, and a sigma
    # of 24 measurably attenuated it where 60 did not.  Zero disables.
    form_blur_sigma: float = 60.0

    # Divide by the local spread rather than a global one, in texture pixels.
    # A moulded mark is small against panel form but large against the flat it
    # sits on, so a local divisor is the only way one threshold can read both.
    # Must be comfortably wider than the mark or it normalises the mark away.
    # Zero falls back to the global percentile below.
    local_scale_sigma: float = 0.0
    # Floor on the local spread, as a percentile of the spread over the whole
    # map.  Without it, genuinely flat areas divide by nearly nothing and
    # amplify their own quantisation into convincing-looking noise.
    local_scale_floor_percentile: float = 60.0
    # Used when local_scale_sigma is zero: the percentile of the absolute
    # detail that is treated as unit scale.
    global_scale_percentile: float = 99.5

    # Levels per unit of scale.  With a global divisor, 127 puts the chosen
    # percentile at full scale; with a local divisor the units are local
    # standard deviations and 48-70 is the useful range.
    gain: float = 127.0

    # Contrast-limited local equalisation, applied last.  This is the stage
    # aimed squarely at the problem the defaults do not solve: a mark that is
    # shallow in absolute terms but obvious within its own panel.  Zero
    # disables it.  Tile count is across the whole atlas, so 64 on a 4096 map
    # is a 64 px tile -- roughly two glyph strokes.
    clahe_clip_limit: float = 0.0
    clahe_tile_count: int = 64

    # Slope mode only: the slope, in encoded levels away from the neutral 128,
    # at which the render saturates.
    slope_saturation_levels: float = 90.0

    # Engraving and embossing differ only in sign, and a detector keyed to
    # bright-on-dark will find one and miss the other.  Flip to check.
    invert: bool = True


DEFAULT_RELIEF_CONFIG = ReliefConfig()


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def normal_gradients(normal_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover the height gradient a tangent-space normal map encodes.

    A surface normal (x, y, z) comes from the height gradient as
    (-dh/du, -dh/dv, 1) normalised, so the gradient is (-x/z, -y/z).  z is
    rebuilt from unit length rather than read, because BeamNG ships BC5 and
    leaves blue at zero.
    """
    x = normal_rgb[:, :, 0].astype(np.float32) / 127.5 - 1.0
    y = normal_rgb[:, :, 1].astype(np.float32) / 127.5 - 1.0
    z = np.sqrt(np.clip(1.0 - x * x - y * y, 1e-6, 1.0))
    return -x / z, -y / z


def height_from_normals(normal_rgb: np.ndarray) -> np.ndarray:
    """Least-squares integration of the gradient field (Frankot-Chellappa).

    Solves the Poisson equation in the Fourier domain, which is one FFT pair
    and costs about six seconds on a 4096 square.  The result is relative --
    there is no absolute height in a normal map -- and carries a large
    low-frequency component that the high-pass downstream removes.
    """
    p, q = normal_gradients(normal_rgb)
    height, width = p.shape
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    denominator = (2 * np.pi * fx) ** 2 + (2 * np.pi * fy) ** 2
    denominator[0, 0] = 1.0  # the mean is unconstrained; pinned to zero below
    numerator = (-1j * 2 * np.pi * fx) * np.fft.fft2(p) + (
        -1j * 2 * np.pi * fy
    ) * np.fft.fft2(q)
    spectrum = numerator / denominator
    spectrum[0, 0] = 0.0
    return np.real(np.fft.ifft2(spectrum)).astype(np.float32)


def fast_blur(array: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that stays usable at the sigmas the high-pass needs.

    A sigma of 60 over a 4096 square takes long enough to make the harness
    unpleasant to tune with, and the result is smooth by construction, so it is
    computed on a decimated copy and scaled back up.
    """
    if sigma <= 0:
        return array
    factor = max(int(sigma / 4), 1)
    if factor == 1:
        return cv2.GaussianBlur(array, (0, 0), sigma)
    height, width = array.shape[:2]
    small = cv2.resize(
        array, (max(width // factor, 1), max(height // factor, 1)),
        interpolation=cv2.INTER_AREA,
    )
    small = cv2.GaussianBlur(small, (0, 0), sigma / factor)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def detail_normals(
    normal_rgb: np.ndarray,
    config: ReliefConfig = DEFAULT_RELIEF_CONFIG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit normals with the panel's own shape taken out of them.

    Lighting the map as authored mostly lights the dash: the curve of a fascia
    swings the normal far further than a moulded letter does, so the letter
    arrives as a ripple on a gradient.  Subtracting a blurred copy of x and y
    removes that shape and leaves the local relief standing on flat ground,
    which is what the shading is supposed to be about.
    """
    x = normal_rgb[:, :, 0].astype(np.float32) / 127.5 - 1.0
    y = normal_rgb[:, :, 1].astype(np.float32) / 127.5 - 1.0
    if config.grain_blur_sigma > 0:
        x = cv2.GaussianBlur(x, (0, 0), config.grain_blur_sigma)
        y = cv2.GaussianBlur(y, (0, 0), config.grain_blur_sigma)
    # Through the same form removal the height path uses, so the choice of
    # remover and its size mean something in every mode.  They were applied
    # only to the height field once, which left three knobs dead in the mode
    # that is the default -- a knob that cannot move the result is worse than
    # no knob, because it invites tuning that cannot work.
    x = remove_form(x, config)
    y = remove_form(y, config)
    z = np.sqrt(np.clip(1.0 - x * x - y * y, 1e-6, 1.0))
    return x, y, z


def shade(
    normal_rgb: np.ndarray,
    config: ReliefConfig = DEFAULT_RELIEF_CONFIG,
) -> np.ndarray:
    """Light the relief from a low angle and return the deviation from flat.

    Zero is flat material, positive is a face turned towards the light and
    negative one turned away, so an embossed stroke comes out as a bright band
    against a dark one -- the shadow that makes it readable.  Returned signed
    and centred on zero so everything downstream can treat flat as background.
    """
    x, y, z = detail_normals(normal_rgb, config)
    elevation = np.radians(config.light_elevation_degrees)
    count = max(int(config.light_count), 1)
    best: np.ndarray | None = None
    for index in range(count):
        azimuth = np.radians(
            config.light_azimuth_degrees + index * (360.0 / count)
        )
        lx = float(np.cos(elevation) * np.cos(azimuth))
        ly = float(np.cos(elevation) * np.sin(azimuth))
        lz = float(np.sin(elevation))
        # Flat ground returns lz, so subtracting it puts flat at zero and
        # leaves only what the relief did to the light.
        response = (x * lx + y * ly + z * lz) - lz
        if best is None:
            best = response
        else:
            best = np.where(np.abs(response) > np.abs(best), response, best)
    assert best is not None
    return best


def relief_field(
    normal_rgb: np.ndarray,
    config: ReliefConfig = DEFAULT_RELIEF_CONFIG,
) -> np.ndarray:
    """The scalar field the render is built from, before any scaling.

    Separated out so the harness can report honest statistics about a region --
    how deep a mark actually is -- without those numbers being distorted by the
    gain and equalisation that only exist to make it visible.
    """
    if config.mode == MODE_SHADED:
        return shade(normal_rgb, config)

    if config.mode == MODE_SLOPE:
        x = normal_rgb[:, :, 0].astype(np.float32) - 128.0
        y = normal_rgb[:, :, 1].astype(np.float32) - 128.0
        field = np.hypot(x, y)
    else:
        field = height_from_normals(normal_rgb)

    if config.grain_blur_sigma > 0:
        field = cv2.GaussianBlur(field, (0, 0), config.grain_blur_sigma)
    return remove_form(field, config)


def remove_form(field: np.ndarray, config: ReliefConfig) -> np.ndarray:
    """Take the panel's own shape out, leaving the marks standing on flat."""
    if config.form_removal == "tophat":
        radius = max(int(config.tophat_radius_px), 1)
        element = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
        )
        if config.tophat_polarity == "raised":
            return cv2.morphologyEx(field, cv2.MORPH_TOPHAT, element)
        if config.tophat_polarity == "engraved":
            return -cv2.morphologyEx(field, cv2.MORPH_BLACKHAT, element)
        return cv2.morphologyEx(field, cv2.MORPH_TOPHAT, element) - cv2.morphologyEx(
            field, cv2.MORPH_BLACKHAT, element
        )
    if config.form_blur_sigma > 0:
        return field - fast_blur(field, config.form_blur_sigma)
    return field


def render_relief(
    normal_rgb: np.ndarray,
    config: ReliefConfig = DEFAULT_RELIEF_CONFIG,
) -> np.ndarray:
    """Render a normal map as a three-channel greyscale for the detector.

    Three channels because everything downstream -- MSER, the magic wand, the
    hull tests -- takes a BGR image.  The channels are identical.
    """
    field = relief_field(normal_rgb, config)

    if config.local_scale_sigma > 0:
        spread = np.sqrt(
            np.clip(fast_blur(field * field, config.local_scale_sigma), 0, None)
        )
        floor = float(
            np.percentile(spread, np.clip(config.local_scale_floor_percentile, 0, 100))
        )
        scale = np.maximum(spread, max(floor, 1e-6))
    else:
        percentile = float(
            np.percentile(np.abs(field), np.clip(config.global_scale_percentile, 0, 100))
        )
        scale = max(percentile, 1e-6)

    normalised = field / scale
    if config.invert:
        normalised = -normalised
    grey = np.clip(normalised * config.gain + 128.0, 0, 255).astype(np.uint8)

    if config.clahe_clip_limit > 0:
        tiles = max(int(config.clahe_tile_count), 1)
        grey = cv2.createCLAHE(
            clipLimit=float(config.clahe_clip_limit), tileGridSize=(tiles, tiles)
        ).apply(grey)

    return np.repeat(grey[:, :, None], 3, axis=2)


def region_relief_report(
    normal_rgb: np.ndarray,
    bounds: tuple[int, int, int, int],
    config: ReliefConfig = DEFAULT_RELIEF_CONFIG,
) -> dict[str, float]:
    """Measure how much relief a rectangle actually carries.

    Reported in the field's own units rather than the rendered ones, so a mark
    can be compared against its neighbours without the gain in the way.  This
    is the number to look at when deciding whether a region the detector missed
    was too shallow to see or merely too close to something louder.
    """
    field = relief_field(normal_rgb, config)
    x, y, w, h = bounds
    height, width = field.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, width), min(y + h, height)
    if x1 <= x0 or y1 <= y0:
        return {}
    inside = field[y0:y1, x0:x1]
    pad = max(max(w, h), 8)
    rx0, ry0 = max(x0 - pad, 0), max(y0 - pad, 0)
    rx1, ry1 = min(x1 + pad, width), min(y1 + pad, height)
    ring = np.ones((ry1 - ry0, rx1 - rx0), dtype=bool)
    ring[y0 - ry0 : y1 - ry0, x0 - rx0 : x1 - rx0] = False
    surround = field[ry0:ry1, rx0:rx1][ring]
    inner = float(np.percentile(inside, 99) - np.percentile(inside, 1))
    outer = (
        float(np.percentile(surround, 99) - np.percentile(surround, 1))
        if surround.size
        else 0.0
    )
    return {
        "inner_spread": inner,
        "ring_spread": outer,
        "prominence": inner / outer if outer > 1e-9 else 0.0,
    }
