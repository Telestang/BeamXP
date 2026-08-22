"""First-run prompt for the two folders BeamXP reads vehicles from.

Neither folder can be guessed reliably: the game may be installed outside
Steam, and the mods folder exists only once BeamNG has been run at least once.
Without them the app opens to an empty Model dropdown with no explanation, so
a launch that finds neither folder configured nor detectable says so up front
and offers to set them.

The prompt is advisory, not a gate. Nothing here insists on BeamNG's own
layout -- any folder holding vehicle zips is accepted -- and it can be
dismissed for good, since a user who only ever opens zips through Load Zip
has no need of either setting.
"""

from __future__ import annotations

from .shared import *

DISMISSED_SETTING = "folderSetupPromptDismissed"

INTRO = (
    "BeamXP lists vehicles from two folders. Point it at the ones you use, "
    "or set them later from the buttons at the top of the window."
)

# What each row explains about the folder it configures, and how deep the scan
# below it goes (see beamxp.core.inventory).
GAME_HINT = "The game's vehicles folder, e.g. BeamNG.drive/content/vehicles. Zips directly inside it are listed."
MODS_HINT = "Where your mods live, e.g. BeamNG.drive/current/mods. Zips inside it and one folder down are listed."


class FolderSetupMixin:
    """The launch-time 'set your folders' prompt and its dismissal setting."""

    def _folder_setup_prompt_dismissed(self) -> bool:
        return bool(self.settings.get(DISMISSED_SETTING))

    def _maybe_prompt_for_folders(self) -> None:
        """Offer the prompt when a folder is missing and it is still wanted.

        Called once at startup, after the window is up so the dialog centres on
        the real geometry rather than the pre-maximise placeholder.
        """
        if self._folder_setup_prompt_dismissed():
            return
        if not self._missing_folder_names():
            return
        self._open_folder_setup_dialog()

    def _open_folder_setup_dialog(self) -> None:
        modal = tk.Toplevel(self)
        modal.title("Set up your BeamNG folders")
        modal.transient(self)
        modal.resizable(False, False)
        modal.columnconfigure(0, weight=1)

        body = ttk.Frame(modal, padding=(14, 12, 14, 6))
        body.grid(row=0, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)

        ttk.Label(body, text=INTRO, wraplength=460, justify="left").grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )

        # Row per folder: name, current state, hint, and its own browse button.
        # Both are shown even when only one is missing, so a wrongly detected
        # folder can be corrected in the same place it is first explained.
        rows: dict[str, dict[str, object]] = {}
        for index, (key, title, hint) in enumerate(
            (
                ("game", "Game vehicles folder", GAME_HINT),
                ("mods", "Mods folder", MODS_HINT),
            ),
            start=1,
        ):
            frame = ttk.LabelFrame(body, text=title, padding=(10, 6, 10, 8))
            frame.grid(row=index, column=0, sticky="ew", pady=(0, 8))
            frame.columnconfigure(0, weight=1)
            state = ttk.Label(frame, text="", wraplength=340, justify="left")
            state.grid(row=0, column=0, sticky="w")
            ttk.Button(
                frame,
                text="Browse...",
                command=lambda k=key, m=modal: self._folder_setup_browse(k, m),
            ).grid(row=0, column=1, sticky="e", padx=(10, 0))
            ttk.Label(frame, text=hint, wraplength=440, justify="left", foreground="#606060").grid(
                row=1, column=0, columnspan=2, sticky="w", pady=(6, 0)
            )
            rows[key] = {"state": state}

        self._folder_setup_rows = rows
        self._folder_setup_modal = modal

        footer = ttk.Frame(modal, padding=(14, 0, 14, 12))
        footer.grid(row=1, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        dismiss_var = tk.BooleanVar(value=self._folder_setup_prompt_dismissed())
        ttk.Checkbutton(
            footer, text="Don't show this again", variable=dismiss_var
        ).grid(row=0, column=0, sticky="w")

        def close() -> None:
            self.settings[DISMISSED_SETTING] = bool(dismiss_var.get())
            core.save_app_settings(self.settings)
            self._folder_setup_rows = {}
            self._folder_setup_modal = None
            modal.destroy()

        ttk.Button(footer, text="Done", command=close).grid(row=0, column=1, sticky="e")

        self._refresh_folder_setup_rows()
        modal.protocol("WM_DELETE_WINDOW", close)
        modal.bind("<Escape>", lambda _event: close())
        modal.bind("<Return>", lambda _event: close())
        self._place_modal_on_app_monitor(modal)
        modal.grab_set()
        modal.focus_set()

    def _refresh_folder_setup_rows(self) -> None:
        """Restate both rows from the settings the main window is working off."""
        rows = getattr(self, "_folder_setup_rows", None)
        if not rows:
            return
        for key, folder in (
            ("game", self._game_vehicles_folder()),
            ("mods", self._mods_folder()),
        ):
            row = rows.get(key)
            if row is None:
                continue
            if self._folder_is_usable(folder):
                text, colour = str(folder), "#20602a"
            else:
                text, colour = "Not set", "#a04000"
            row["state"].configure(text=text, foreground=colour)

    def _folder_setup_browse(self, key: str, modal: tk.Toplevel) -> None:
        """Pick one of the two folders from inside the prompt.

        The grab is released around the picker so the native dialog is not
        held behind the modal, and taken back once the choice is made.
        """
        titles = {
            "game": "Select the folder holding the game's vehicle zips",
            "mods": "Select the folder holding your mod zips",
        }
        current = self._game_vehicles_folder() if key == "game" else self._mods_folder()
        initial = existing_initial_dir(current, core.WORKSPACE_DIR)
        modal.grab_release()
        try:
            path = filedialog.askdirectory(
                parent=modal, title=titles[key], initialdir=initial, mustexist=True
            )
        finally:
            if modal.winfo_exists():
                modal.grab_set()
        if not path:
            return
        if key == "game":
            self.settings["gameVehiclesFolder"] = path
        else:
            self.mods_folder_var.set(path)
            self.settings["modsFolder"] = path
            self.settings["lastModsFolder"] = path
        core.save_app_settings(self.settings)
        self._refresh_folder_setup_rows()
        self._refresh_folder_buttons()
        self._start_inventory_scan()
