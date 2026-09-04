"""The labelling modal: every region at one stage, judged by hand.

2026-09-04: the picture the stage viewer draws answers "what survived"; it
cannot answer "should it have". That question is one region at a time, and
there are hundreds of them, so it needs a list rather than a canvas -- crops
big enough to judge, three verdicts, and a keyboard that keeps up.

Deliberately its own window and its own module. The main app is already
1,500 lines of one class, and a modal that shares its state is how a viewer
turns into a workbench that can no longer be reasoned about. This one is
handed what it needs and hands back a directory.

The rules it follows:

- **The current stage, not the finished set.** Labelling the survivors teaches
  a filter nothing about what it should have dropped, and the stage picker is
  already how the question is framed.
- **Nothing is labelled by default.** A region starts unset and stays out of
  the export until someone decides. Defaulting to a verdict produces a corpus
  of whatever the default was.
- **The parameters travel with the labels.** A verdict is only meaningful
  against the settings that produced the region.
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from glyph_lab.labelling import (
    DEFAULT_CONTEXT_PX,
    FALSE_POSITIVE,
    IGNORE,
    TRUE_POSITIVE,
    LabelledRegion,
    export_session,
    region_statistics,
    write_region_crops,
)

BACKGROUND = "#1b1b1b"
FOREGROUND = "#e6e6e6"
UNSET = ""
# How much of the crop to show. A 20-texel glyph is unreadable at 1:1 on a
# modern display, and the judgement is about shape, so it is scaled up to a
# fixed box rather than shown at its own size.
THUMBNAIL_PX = 96
# Rows built per idle slice, so a page appears at once rather than freezing
# the window for as long as it takes to build.
BATCH = 24
# Rows per page.
#
# 2026-09-04: paged because Tk addresses canvas coordinates as signed 16-bit.
# A frame taller than 32,767 px is silently unreachable below that line: on a
# 557-region session the body grew to 55,000 px and everything past roughly
# the 335th row could not be scrolled to. The rows existed and were counted --
# the tally said 557 with none unset -- so it read as "the list stops" rather
# than as anything being wrong, and a bulk "all true" had quietly labelled
# hundreds of regions nobody could see.
#
# 60 rows at ~110 px is about 6,600 px, comfortably inside the limit with room
# for a taller thumbnail.
PAGE = 60


@dataclass(frozen=True, slots=True)
class LabelSubject:
    """One texture's regions at the stage being labelled."""

    name: str
    stage: str
    colour: np.ndarray
    edges: np.ndarray | None
    regions: tuple           # of glyph_lab.regions.Region


def thumbnail(
    colour: np.ndarray, mask: np.ndarray, bounds: tuple[int, int, int, int],
    context_px: int = DEFAULT_CONTEXT_PX,
) -> Image.Image:
    """The region's colour with its own texels tinted, scaled to a fixed box.

    The tint is what makes the list judgeable at speed: without it a crop of a
    busy panel says nothing about which texels the detector actually claimed,
    and that -- not whether there is a mark somewhere in the window -- is the
    question being answered.
    """
    from glyph_lab.labelling import crop_window, placed_mask

    window = crop_window(bounds, colour.shape[:2], context_px)
    left, top, width, height = window
    if width <= 0 or height <= 0:
        return Image.new("RGB", (THUMBNAIL_PX, THUMBNAIL_PX), BACKGROUND)
    patch = np.asarray(
        colour[top: top + height, left: left + width], dtype=np.float32
    ).copy()
    drawn = placed_mask(bounds, mask, window)
    # Toward green, not replaced by it: the mark's own colour still has to be
    # visible, since "is this ink or is this the material" is half the call.
    patch[drawn] = patch[drawn] * 0.45 + np.array([40.0, 230.0, 90.0]) * 0.55
    image = Image.fromarray(np.clip(patch, 0, 255).astype(np.uint8))
    scale = max(1, min(THUMBNAIL_PX // max(width, height, 1), 8))
    if scale > 1:
        image = image.resize((width * scale, height * scale), Image.NEAREST)
    return image


class LabelWindow(tk.Toplevel):
    """A scrolling list of crops, three radio buttons each, and an export."""

    def __init__(
        self, master: tk.Misc, subjects: list[LabelSubject],
        parameters: dict, initial_dir: Path | None = None,
    ) -> None:
        super().__init__(master)
        self.title("Label detections")
        self.geometry("900x800")
        self.configure(background=BACKGROUND)
        self._subjects = subjects
        self._parameters = parameters
        self._initial_dir = initial_dir
        # Every region, in one flat order, so a page is a slice of it.  The
        # verdicts live here rather than in the widgets: a page's rows are
        # destroyed when you leave it, and a verdict must not go with them.
        self._all: list[tuple[LabelSubject, int]] = [
            (subject, index)
            for subject in subjects
            for index in range(len(subject.regions))
        ]
        self._choices: dict[tuple[str, int], tk.StringVar] = {}
        self._photos: list[ImageTk.PhotoImage] = []
        self._pending: list[tuple[LabelSubject, int]] = []
        self._built = 0
        self._page = 0

        header = ttk.Frame(self)
        header.pack(side="top", fill="x", padx=8, pady=6)
        self._summary = ttk.Label(header, text="", anchor="w")
        self._summary.pack(side="left", fill="x", expand=True)
        ttk.Button(header, text="Export…", command=self.export).pack(side="right")
        # Scoped to the page on purpose.  A bulk verdict over the whole set
        # labels regions nobody has looked at, and reads afterwards exactly
        # like one that was reviewed.
        ttk.Button(
            header, text="Page: all false",
            command=lambda: self._set_page(FALSE_POSITIVE),
        ).pack(side="right", padx=(0, 6))
        ttk.Button(
            header, text="Page: all true",
            command=lambda: self._set_page(TRUE_POSITIVE),
        ).pack(side="right", padx=(0, 6))

        nav = ttk.Frame(self)
        nav.pack(side="bottom", fill="x", padx=8, pady=6)
        ttk.Button(nav, text="← Previous", command=self.previous_page).pack(
            side="left",
        )
        ttk.Button(nav, text="Next →", command=self.next_page).pack(
            side="left", padx=(6, 0),
        )
        self._page_label = ttk.Label(nav, text="", anchor="w")
        self._page_label.pack(side="left", padx=12)
        self._jump = tk.StringVar(master=self, value="")
        ttk.Label(nav, text="go to region").pack(side="left", padx=(12, 4))
        jump_box = ttk.Entry(nav, textvariable=self._jump, width=8)
        jump_box.pack(side="left")
        jump_box.bind("<Return>", lambda _e: self.jump_to_region())

        shell = tk.Canvas(self, background=BACKGROUND, highlightthickness=0)
        scroll = ttk.Scrollbar(self, orient="vertical", command=shell.yview)
        self._body = ttk.Frame(shell)
        self._body.bind(
            "<Configure>",
            lambda _e: shell.configure(scrollregion=shell.bbox("all")),
        )
        shell.create_window((0, 0), window=self._body, anchor="nw")
        shell.configure(yscrollcommand=scroll.set)
        shell.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self._shell = shell
        shell.bind_all("<MouseWheel>", self._on_wheel)

        self.show_page(0)

    # -- building ---------------------------------------------------------

    @property
    def pages(self) -> int:
        return max(1, (len(self._all) + PAGE - 1) // PAGE)

    def show_page(self, number: int) -> None:
        """Rebuild the body with one page's rows and nothing else.

        The rows are destroyed rather than hidden.  Keeping every page's
        widgets alive is what put the body past Tk's coordinate limit in the
        first place, and a thumbnail per region is real memory besides.
        """
        self._page = max(0, min(int(number), self.pages - 1))
        for child in self._body.pack_slaves():
            child.destroy()
        self._photos.clear()
        start = self._page * PAGE
        self._pending = list(self._all[start: start + PAGE])
        self._built = 0
        self._shell.yview_moveto(0.0)
        self._refresh_summary()
        self.after_idle(self._build_batch)

    def next_page(self) -> None:
        if self._page + 1 < self.pages:
            self.show_page(self._page + 1)

    def previous_page(self) -> None:
        if self._page > 0:
            self.show_page(self._page - 1)

    def jump_to_region(self) -> None:
        """Open the page holding the nth region, counting from 1."""
        try:
            wanted = int(float(self._jump.get()))
        except (TypeError, ValueError):
            return
        wanted = max(1, min(wanted, len(self._all)))
        self.show_page((wanted - 1) // PAGE)

    def _build_batch(self) -> None:
        """Add a slice of rows, then yield so the window stays alive."""
        if not self.winfo_exists():
            return
        for _ in range(BATCH):
            if not self._pending:
                self._refresh_summary()
                return
            subject, index = self._pending.pop(0)
            self._add_row(subject, index)
            self._built += 1
        self._refresh_summary()
        self.after_idle(self._build_batch)

    def _add_row(self, subject: LabelSubject, index: int) -> None:
        region = subject.regions[index]
        row = ttk.Frame(self._body)
        row.pack(fill="x", padx=6, pady=3)

        photo = ImageTk.PhotoImage(
            thumbnail(subject.colour, region.mask, region.bounds)
        )
        # Held on the window: a PhotoImage with no reference is collected and
        # the label goes blank, which is the oldest bug in Tk.
        self._photos.append(photo)
        tk.Label(row, image=photo, background=BACKGROUND).pack(side="left")

        texels, fill, aspect = region_statistics(region.bounds, region.mask)
        x, y, width, height = region.bounds
        ttk.Label(
            row,
            text=(f"{subject.name}  #{index}\n"
                  f"{width}x{height} at ({x},{y})\n"
                  f"{texels} texels  fill {fill:.2f}  aspect {aspect:.2f}"),
            width=42,
        ).pack(side="left", padx=8)

        # Reused, never replaced: leaving a page destroys its rows, and a
        # fresh variable on the way back would silently drop the verdict this
        # region already has.  master=self for the reason the main window
        # documents -- an unmastered Variable binds to whichever Tk was made
        # first in the process and is finalised against it.
        key = (subject.name, index)
        choice = self._choices.get(key)
        if choice is None:
            choice = tk.StringVar(master=self, value=UNSET)
            self._choices[key] = choice
        for text, value in (
            ("true", TRUE_POSITIVE), ("false", FALSE_POSITIVE), ("ignore", IGNORE),
        ):
            ttk.Radiobutton(
                row, text=text, value=value, variable=choice,
                command=self._refresh_summary,
            ).pack(side="left", padx=4)

    # -- state ------------------------------------------------------------

    def _set_page(self, value: str) -> None:
        """Fill in the unset rows on this page, and no others.

        Not a reset -- it is there to clear a page of obvious noise quickly,
        and overwriting considered verdicts would make it too dangerous to
        use.  And not the whole set: a bulk verdict over regions nobody has
        seen produces a corpus that reads as reviewed and is not.
        """
        start = self._page * PAGE
        for subject, index in self._all[start: start + PAGE]:
            choice = self._choices.get((subject.name, index))
            if choice is not None and not choice.get():
                choice.set(value)
        self._refresh_summary()

    def _tally(self) -> dict[str, int]:
        counts = {TRUE_POSITIVE: 0, FALSE_POSITIVE: 0, IGNORE: 0, UNSET: 0}
        for choice in self._choices.values():
            counts[choice.get()] = counts.get(choice.get(), 0) + 1
        return counts

    def _tally_all(self) -> dict[str, int]:
        """Over every region, not merely the ones with a widget right now."""
        counts = {TRUE_POSITIVE: 0, FALSE_POSITIVE: 0, IGNORE: 0, UNSET: 0}
        for subject, index in self._all:
            choice = self._choices.get((subject.name, index))
            counts[choice.get() if choice is not None else UNSET] += 1
        return counts

    def _refresh_summary(self) -> None:
        counts = self._tally_all()
        total = len(self._all)
        building = (
            "" if not self._pending
            else f"  (drawing {self._built}/{self._built + len(self._pending)})"
        )
        stage = self._subjects[0].stage if self._subjects else "-"
        self._summary.configure(
            text=(f"{total} region(s) at {stage}"
                  f"   true {counts[TRUE_POSITIVE]}"
                  f"   false {counts[FALSE_POSITIVE]}"
                  f"   ignore {counts[IGNORE]}"
                  f"   unset {counts[UNSET]}{building}")
        )
        start = self._page * PAGE
        last = min(start + PAGE, total)
        self._page_label.configure(
            text=f"page {self._page + 1} of {self.pages}"
                 f"   — regions {start + 1}-{last}"
        )

    def _on_wheel(self, event: tk.Event) -> None:
        self._shell.yview_scroll(int(-event.delta / 60), "units")

    # -- export -----------------------------------------------------------

    def export(self) -> None:
        counts = self._tally()
        if counts[UNSET] == len(self._choices):
            messagebox.showinfo(
                "Nothing labelled", "Give at least one region a verdict first.",
                parent=self,
            )
            return
        chosen = filedialog.askdirectory(
            parent=self, title="Write the labelled set to",
            initialdir=str(self._initial_dir) if self._initial_dir else None,
        )
        if not chosen:
            return
        out_dir = Path(chosen)
        entries: list[LabelledRegion] = []
        for subject in self._subjects:
            for index, region in enumerate(subject.regions):
                choice = self._choices.get((subject.name, index))
                # An unset region is not a verdict, and a row that never got
                # built was never looked at. Neither belongs in the dataset.
                if choice is None or not choice.get():
                    continue
                crops = write_region_crops(
                    out_dir, subject.name, index, region.bounds, region.mask,
                    subject.colour, subject.edges,
                )
                texels, fill, aspect = region_statistics(
                    region.bounds, region.mask
                )
                entries.append(
                    LabelledRegion(
                        texture=subject.name, stage=subject.stage, index=index,
                        label=choice.get(), bounds=tuple(region.bounds),
                        texels=texels, fill=fill, aspect=aspect,
                        domain=int(region.domain), crops=crops,
                    )
                )
        manifest = export_session(out_dir, self._parameters, entries)
        messagebox.showinfo(
            "Exported",
            f"{len(entries)} labelled region(s)\n{manifest}",
            parent=self,
        )
