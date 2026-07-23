"""Owns the project's annotations and mediates the Annotate view/page.

Annotations are a *project-global* resource (a point series can span many
sections), so they do not fit the per-section draft/save machinery in
:class:`~verso.gui.views.base_canvas_view.BaseCanvasView`. This controller holds
the loaded ``list[PointSeries]``, the active selection, and a single dirty flag;
it turns :class:`~verso.gui.widgets.properties.annotate_page.AnnotatePage` intent
signals into model mutations and pushes the result to both the page (manager UI)
and the :class:`~verso.gui.views.annotate_view.AnnotateView` (canvas rendering).

Persistence goes through :mod:`verso.engine.io.annotation_io`; the ``annotations/``
folder is synced on save.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from verso.engine.annotations import annotation_images, points_in_polygon
from verso.engine.io.annotation_io import (
    guess_point_columns,
    load_annotations,
    load_points_csv,
    save_annotations,
)
from verso.engine.model.annotation import (
    Annotation,
    AnnotationPoint,
    AreaAnnotation,
    PointSeries,
)

if TYPE_CHECKING:
    from verso.gui.jobs import BackgroundJob
    from verso.gui.main_window import MainWindow
    from verso.gui.widgets.properties.annotate_page import AnnotatePage


class AnnotationController:
    """Project annotations: state, page/view coordination, and persistence."""

    # Cycled to give each new annotation a distinct default colour.
    _PALETTE: tuple[tuple[int, int, int], ...] = (
        (255, 64, 64),
        (64, 160, 255),
        (80, 200, 120),
        (255, 180, 40),
        (200, 100, 255),
        (0, 200, 200),
    )

    # Shallow undo history: each entry is a closure that restores the pre-edit
    # state (a point series' points, or an area's per-section mask).
    _UNDO_LIMIT = 20

    def __init__(self, window: MainWindow) -> None:
        self._window = window
        self._state = window._state
        self._annotations: list[Annotation] = []
        self._active: int = -1
        self._dirty: bool = False
        self._undo_stack: list[Callable[[], None]] = []
        # Background-load bookkeeping. ``_load_gen`` is bumped on every (re)load so
        # a worker that finishes after the project changed again is ignored;
        # ``_load_jobs`` keeps in-flight jobs referenced until their thread finishes
        # (each job owns the worker QObject, which would otherwise be collected
        # mid-run once this method's local goes out of scope).
        self._load_gen: int = 0
        self._load_jobs: set[BackgroundJob] = set()
        # Coalesces the O(N) filmstrip-marker rescan during a burst of point adds:
        # markers only change when a section gains its first point for the series,
        # so a stream of clicks pays the rescan once, after it settles.
        self._marker_timer = QTimer()
        self._marker_timer.setSingleShot(True)
        self._marker_timer.setInterval(1000)
        self._marker_timer.timeout.connect(self.refresh_filmstrip_markers)

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def connect_page(self, page: AnnotatePage) -> None:
        page.manager.new_point_requested.connect(self.new_point_series)
        page.manager.new_area_requested.connect(self.new_area)
        page.manager.import_requested.connect(self.import_csv)
        page.manager.delete_requested.connect(self.delete_active)
        page.manager.active_changed.connect(self.set_active)
        page.manager.visibility_changed.connect(self.set_visibility)
        page.selected.color_changed.connect(self.set_color)
        page.selected.opacity_changed.connect(self.set_opacity)
        page.selected.point_size_changed.connect(self.set_point_size)
        page.selected.rename_requested.connect(self.rename_active)
        page.save_requested.connect(self.save)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def is_dirty(self) -> bool:
        return self._dirty

    def load_for_project(self) -> None:
        """(Re)load annotations from disk for the current project (synchronous).

        Kept synchronous for scripting/tests and as the fallback the async path
        degrades to. The GUI project-open path calls :meth:`load_for_project_async`
        instead so a huge point series never blocks the window.
        """
        self._reset_for_load()
        self._install_loaded(self._read_annotations())

    def load_for_project_async(self) -> None:
        """(Re)load annotations off the UI thread so project-open never blocks.

        Clears the current annotations immediately (the view/manager show an empty
        set right away), then parses ``annotations/`` in a worker thread and
        installs the result when it finishes. ``load_annotations`` builds one
        object per point, so a 500k+ point series takes a couple of seconds —
        doing it here keeps the window responsive on open. Degrades to a
        synchronous load when there is nothing to parse.

        The worker start is deferred to the next event-loop turn: parsing is
        pure-Python and CPU-bound, so it holds the GIL in bursts and would starve
        the window's *first* paint (a white window on startup) if it began during
        this call, which runs before the first paint. Posting it after the paint
        events queued by project-open lets the populated window render first, then
        the parse runs in the background.
        """
        self._reset_for_load()
        self._refresh()  # paint the empty set now; the worker fills it in later
        project = self._state.project
        path = self._state.project_path
        if project is None or path is None:
            return
        gen = self._load_gen
        QTimer.singleShot(200, lambda: self._begin_load(path.parent, gen))

    def _begin_load(self, project_dir: Path, gen: int) -> None:
        """Start the deferred background parse, unless a newer load superseded it."""
        if gen != self._load_gen:
            return
        from verso.gui.jobs import AnnotationLoadWorker, BackgroundJob

        job = BackgroundJob(self._window, AnnotationLoadWorker(project_dir, gen), silent=True)
        self._load_jobs.add(job)
        # Both callbacks are bound methods of this main-thread controller, so Qt
        # delivers them as queued connections back on the UI thread.
        job.start(self._on_async_loaded, self._prune_finished_jobs)

    def _on_async_loaded(self, annotations: list[Annotation], gen: int) -> None:
        """Install annotations parsed by the worker, unless the load was superseded."""
        if gen != self._load_gen:
            return  # a newer (re)load started while this one was parsing
        # If the user began creating annotations during the load (a narrow race:
        # the app opens in Overview and the load is brief), keep their in-memory
        # work rather than clobbering it with the on-disk set.
        if self._dirty or self._annotations:
            return
        self._install_loaded(annotations)

    def _prune_finished_jobs(self) -> None:
        """Drop references to loader jobs whose thread has stopped."""
        self._load_jobs = {j for j in self._load_jobs if j.is_running()}

    def _reset_for_load(self) -> None:
        """Clear annotation state and invalidate any in-flight background load."""
        self._load_gen += 1
        self._annotations = []
        self._active = -1
        self._dirty = False
        self._undo_stack.clear()

    def _read_annotations(self) -> list[Annotation]:
        """Load annotations for the current project synchronously (empty on error)."""
        path = self._state.project_path
        if self._state.project is None or path is None:
            return []
        try:
            return load_annotations(path.parent)
        except Exception:
            return []

    def _install_loaded(self, annotations: list[Annotation]) -> None:
        """Adopt a freshly-loaded annotation list and refresh the UI."""
        self._annotations = annotations
        self._active = 0 if annotations else -1
        self._refresh()

    def shutdown(self) -> None:
        """Stop in-flight background loads before the window is destroyed."""
        self._load_gen += 1  # ignore any result still arriving
        for job in list(self._load_jobs):
            job.stop()
        self._load_jobs.clear()

    def _active_annotation(self) -> Annotation | None:
        if 0 <= self._active < len(self._annotations):
            return self._annotations[self._active]
        return None

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def new_point_series(self) -> None:
        if self._state.project is None:
            return
        color = self._PALETTE[len(self._annotations) % len(self._PALETTE)]
        title = self._unique_title("points")
        self._annotations.append(PointSeries(title=title, color=color))
        self._active = len(self._annotations) - 1
        self._undo_stack.clear()
        self._mark_dirty()
        self._refresh()

    def new_area(self) -> None:
        if self._state.project is None:
            return
        color = self._PALETTE[len(self._annotations) % len(self._PALETTE)]
        title = self._unique_title("area")
        self._annotations.append(AreaAnnotation(title=title, color=color))
        self._active = len(self._annotations) - 1
        self._undo_stack.clear()
        self._mark_dirty()
        self._refresh()

    def import_csv(self) -> None:
        if self._state.project is None:
            return
        path_str, _ = QFileDialog.getOpenFileName(
            self._window, "Import points from CSV", "", "CSV files (*.csv);;All files (*)"
        )
        if not path_str:
            return
        path = Path(path_str)

        try:
            with open(path, newline="", encoding="utf-8") as fh:
                headers = next(csv.reader(fh), [])
        except OSError as exc:
            QMessageBox.critical(self._window, "Cannot read CSV", str(exc))
            return

        guess = guess_point_columns(headers)
        x_col, y_col, image_col = guess["x"], guess["y"], guess["image"]
        if x_col is None or y_col is None:
            from verso.gui.dialogs.annotation_csv import AnnotationCsvDialog

            dlg = AnnotationCsvDialog(headers, guess, self._window)
            if dlg.exec() != AnnotationCsvDialog.DialogCode.Accepted:
                return
            x_col, y_col, image_col = dlg.result_columns()

        default_image = self._current_image_key()
        try:
            points = load_points_csv(path, x_col, y_col, image_col, default_image)
        except (OSError, KeyError) as exc:
            QMessageBox.critical(self._window, "Cannot import points", str(exc))
            return

        color = self._PALETTE[len(self._annotations) % len(self._PALETTE)]
        title = self._unique_title(path.stem or "annotation")
        self._annotations.append(PointSeries(title=title, color=color, points=points))
        self._active = len(self._annotations) - 1
        self._undo_stack.clear()
        self._mark_dirty()
        self._refresh()
        self._state.show_status(f"Imported {len(points)} point(s) from {path.name}")

    def delete_active(self) -> None:
        ann = self._active_annotation()
        if ann is None:
            return
        resp = QMessageBox.question(
            self._window,
            "Delete annotation",
            f"Delete annotation “{ann.title}”?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return
        del self._annotations[self._active]
        self._active = min(self._active, len(self._annotations) - 1)
        self._undo_stack.clear()
        self._mark_dirty()
        self._refresh()

    def set_active(self, index: int) -> None:
        if index == self._active:
            return
        self._active = index
        self._refresh()

    def set_visibility(self, index: int, visible: bool) -> None:
        if not (0 <= index < len(self._annotations)):
            return
        ann = self._annotations[index]
        if ann.visible == visible:
            return
        ann.visible = visible
        self._mark_dirty()
        self._sync_ui()  # visibility doesn't move the presence markers

    def set_color(self, color: tuple[int, int, int]) -> None:
        ann = self._active_annotation()
        if ann is None or ann.color == color:
            return
        ann.color = color
        self._mark_dirty()
        self._refresh()

    def set_opacity(self, opacity: float) -> None:
        # Opacity is area-only; point scatters always render fully opaque.
        ann = self._active_annotation()
        if not isinstance(ann, AreaAnnotation) or ann.opacity == opacity:
            return
        ann.opacity = opacity
        self._mark_dirty()
        self._sync_ui()  # opacity slider drag: no marker rescan

    def set_point_size(self, size: int) -> None:
        ann = self._active_annotation()
        if not isinstance(ann, PointSeries) or ann.point_size == size:
            return
        ann.point_size = size
        self._mark_dirty()
        self._sync_ui()  # point-size slider drag: no marker rescan

    def rename_active(self, title: str) -> None:
        ann = self._active_annotation()
        title = title.strip()
        if ann is None or not title or ann.title == title:
            return
        ann.title = self._unique_title(title, exclude=self._active)
        self._mark_dirty()
        self._sync_ui()  # rename doesn't move the presence markers

    # ------------------------------------------------------------------
    # Point editing (Annotate canvas: click to add, lasso to remove)
    # ------------------------------------------------------------------

    def add_point(self, x: float, y: float) -> None:
        """Append a point (original-res px) to the active point series."""
        ann = self._active_annotation()
        section = self._state.current_section
        if not isinstance(ann, PointSeries) or section is None:
            return
        self._push_points_undo(ann)
        image = section.image_key
        ann.points.append(AnnotationPoint(x=x, y=y, image=image))
        self._mark_dirty()
        # Render the new point now (cheap: the view folds it into its cache), but
        # defer the filmstrip-marker rescan so a burst of clicks coalesces to one.
        self._sync_ui()
        self._marker_timer.start()

    def remove_in_polygon(self, polygon: list[tuple[float, float]]) -> None:
        """Remove active-annotation points on the current section inside ``polygon``.

        The polygon is in original-resolution pixels (the view converts from
        display space). Only points belonging to the visible section are hit —
        you can only lasso what you can see.
        """
        ann = self._active_annotation()
        section = self._state.current_section
        if not isinstance(ann, PointSeries) or section is None or len(polygon) < 3:
            return
        image = section.image_key.lower()
        # Find the current section's points (and their coords) in a single pass.
        # A point's basename is normalised once per *distinct* image string, not
        # once per point: a series can hold hundreds of thousands of points that
        # share only a handful of image strings, so a per-point basename() call
        # was ~1.2 s of pure string work on a 740k-point series.
        norm: dict[str, str] = {}
        on_section: list[int] = []
        xs: list[float] = []
        ys: list[float] = []
        for i, p in enumerate(ann.points):
            key = norm.get(p.image)
            if key is None:
                key = norm[p.image] = os.path.basename(p.image).lower()
            if key == image:
                on_section.append(i)
                xs.append(p.x)
                ys.append(p.y)
        if not on_section:
            return
        coords = np.column_stack((xs, ys))
        inside = points_in_polygon(coords, np.array(polygon, dtype=float))
        remove = {on_section[k] for k in np.nonzero(inside)[0]}
        if not remove:
            return
        self._push_points_undo(ann)
        ann.points = [p for i, p in enumerate(ann.points) if i not in remove]
        self._mark_dirty()
        self._refresh()
        self._state.show_status(f"Removed {len(remove)} point(s)")

    # ------------------------------------------------------------------
    # Area editing (Annotate canvas: brush / freehand mask painting)
    # ------------------------------------------------------------------

    def active_area(self) -> AreaAnnotation | None:
        ann = self._active_annotation()
        return ann if isinstance(ann, AreaAnnotation) else None

    def begin_area_edit(self) -> None:
        """Snapshot the active area's current-section mask before a stroke (undo)."""
        area = self.active_area()
        section = self._state.current_section
        if area is None or section is None:
            return
        image = section.image_key
        had = image in area.masks
        old = area.masks.get(image)
        old_copy = old.copy() if old is not None else None
        index = self._active

        def restore() -> None:
            if index < 0 or index >= len(self._annotations):
                return
            target = self._annotations[index]
            if not isinstance(target, AreaAnnotation):
                return
            if had and old_copy is not None:
                target.masks[image] = old_copy
            else:
                target.masks.pop(image, None)
            self._active = index

        self._push_undo(restore)

    def commit_area_edit(self) -> None:
        """Finalise a stroke the view painted in place: mark dirty + refresh."""
        if self.active_area() is None:
            return
        self._mark_dirty()
        self._refresh()

    # ------------------------------------------------------------------
    # Undo (closure-based; unifies point edits and area mask edits)
    # ------------------------------------------------------------------

    def undo(self) -> None:
        """Undo the last point or area edit by running its restore closure."""
        if not self._undo_stack:
            return
        restore = self._undo_stack.pop()
        restore()
        self._mark_dirty()
        self._refresh()

    def _push_points_undo(self, series: PointSeries) -> None:
        index = self._active
        old = list(series.points)

        def restore() -> None:
            series.points = old
            self._active = index

        self._push_undo(restore)

    def _push_undo(self, restore: Callable[[], None]) -> None:
        self._undo_stack.append(restore)
        if len(self._undo_stack) > self._UNDO_LIMIT:
            self._undo_stack.pop(0)

    def save(self) -> bool:
        """Write annotations to the project's ``annotations/`` folder."""
        path = self._state.project_path
        if self._state.project is None or path is None:
            if self._dirty:
                QMessageBox.information(
                    self._window,
                    "Save project first",
                    "Save the project before saving annotations so they have a home on disk.",
                )
            return False
        try:
            save_annotations(path.parent, self._annotations)
        except OSError as exc:
            QMessageBox.critical(self._window, "Cannot save annotations", str(exc))
            return False
        self._dirty = False
        self._window._props.annotate.set_dirty(False)
        self._state.show_status("Saved annotations")
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_image_key(self) -> str:
        section = self._state.current_section
        return section.image_key if section is not None else ""

    def _unique_title(self, base: str, exclude: int = -1) -> str:
        existing = {a.title for i, a in enumerate(self._annotations) if i != exclude}
        if base not in existing:
            return base
        i = 2
        while f"{base} {i}" in existing:
            i += 1
        return f"{base} {i}"

    def _mark_dirty(self) -> None:
        self._dirty = True

    def _refresh(self) -> None:
        """Re-sync the manager/view *and* the filmstrip presence markers.

        Used by every edit that can change which sections the active annotation
        touches, or which annotation is active, or its colour. Cosmetic edits that
        cannot move the markers (opacity, point size, rename, visibility) call
        :meth:`_sync_ui` instead, so a slider drag never rescans the point series.
        """
        self._sync_ui()
        self.refresh_filmstrip_markers()

    def _sync_ui(self) -> None:
        """Push the annotation list + dirty flag to the manager and canvas view."""
        self._window._props.annotate.set_annotations(self._annotations, self._active)
        self._window._props.annotate.set_dirty(self._dirty)
        self._window._annotate.set_annotations(self._annotations, self._active)

    def refresh_filmstrip_markers(self) -> None:
        """Flag the filmstrip tiles that carry the selected annotation.

        Unlike Prep/Align/Warp (whose per-section status dots are owned by
        :class:`~verso.gui.widgets.filmstrip_status.FilmstripStatusPresenter`),
        annotations are project-global: this paints a square marker in the active
        annotation's colour on every section it appears in, so the user can spot
        which images to visit without hunting. A no-op outside the Annotate view
        so it never clobbers another view's dots; the presenter repaints those on
        the way back in.
        """
        if self._window._current_mode != "annotate":
            return
        project = self._state.project
        filmstrip = self._window._filmstrip
        if project is None:
            filmstrip.set_statuses([], shape="square")
            return
        ann = self._active_annotation()
        if ann is None:
            filmstrip.set_statuses([None] * len(project.sections), shape="square")
            return
        images = annotation_images(ann)
        color = "#{:02x}{:02x}{:02x}".format(*ann.color)
        colors = [
            color if s.image_key.lower() in images else None for s in project.sections
        ]
        filmstrip.set_statuses(colors, shape="square")
