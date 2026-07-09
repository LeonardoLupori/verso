"""Central-widget, dock, shortcut, status-bar and signal wiring for MainWindow.

Pure one-time assembly split out of the window (menus/toolbar live in
:mod:`verso.gui.menus`). Each function receives the window, constructs its
widgets, and stashes the handles the window keeps (``_stack``, ``_overview``,
``_prep``, ``_panel``, ``_align``, ``_warp``, ``_props``, ``_filmstrip``,
``_statusbar`` …) back onto it. No behaviour lives here — the slots these signals
connect to remain methods on the window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QDockWidget, QStackedWidget, QWidget

from verso.gui.utils import require
from verso.gui.views.align_view import AlignView
from verso.gui.views.overview_view import OverviewView
from verso.gui.views.prep_view import PrepView
from verso.gui.views.warp_view import WarpView
from verso.gui.widgets.filmstrip import Filmstrip
from verso.gui.widgets.filmstrip_status import FilmstripStatusPresenter
from verso.gui.widgets.properties import PropertiesPanel
from verso.gui.widgets.section_canvas_panel import SectionCanvasPanel

if TYPE_CHECKING:
    from verso.gui.main_window import MainWindow


def build_central(window: MainWindow) -> None:
    """Build the stacked Overview / Prep / Align / Warp views."""
    window._stack = QStackedWidget()
    window.setCentralWidget(window._stack)

    window._overview = OverviewView(window._state)
    window._prep = PrepView(window._state)
    # Shared canvas + region bar + section/atlas/channels state.  Reparented
    # into whichever of AlignView / WarpView is currently active so zoom,
    # pan, and the channel-layer cache survive mode switches.
    window._panel = SectionCanvasPanel()
    window._align = AlignView(window._panel, window._state)
    window._warp = WarpView(window._panel, window._state)

    window._stack.addWidget(window._overview)  # 0
    window._stack.addWidget(window._prep)  # 1
    window._stack.addWidget(window._align)  # 2
    window._stack.addWidget(window._warp)  # 3

    # Park the panel inside AlignView's slot immediately.  If we left it as
    # a free-floating child of MainWindow (the default when ``SectionCanvasPanel``
    # is constructed without a layout parent), it renders at (0, 0) of the
    # main window and covers the menubar until the first Align/Warp switch.
    window._align.activate()


def build_docks(window: MainWindow) -> None:
    """Build the right (Properties) and bottom (Filmstrip) docks."""
    # Right: properties panel
    window._props = PropertiesPanel()
    right_dock = QDockWidget("Properties", window)
    right_dock.setWidget(window._props)
    right_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
    right_dock.setTitleBarWidget(QWidget())  # hide title bar
    window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, right_dock)
    window._right_dock = right_dock

    # Bottom: filmstrip
    window._filmstrip = Filmstrip()
    bottom_dock = QDockWidget("Filmstrip", window)
    bottom_dock.setWidget(window._filmstrip)
    bottom_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
    bottom_dock.setTitleBarWidget(QWidget())
    window.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, bottom_dock)
    window._bottom_dock = bottom_dock
    window._filmstrip_status = FilmstripStatusPresenter(window._state, window._filmstrip)

    # PropertiesPanel no longer pins its width, so its sizeHint inflates
    # the dock past what we want at launch.  Force the initial width here;
    # the user can still drag the splitter to resize.
    window.resizeDocks([right_dock], [270], Qt.Orientation.Horizontal)


def build_shortcuts(window: MainWindow) -> None:
    """Register the left/right arrow section-stepping shortcuts."""
    window._section_shortcuts = []
    for key, delta in (
        (Qt.Key.Key_Left, -1),
        (Qt.Key.Key_Right, 1),
    ):
        shortcut = QShortcut(QKeySequence(key), window)
        shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        shortcut.activated.connect(lambda d=delta: window._step_section(d))
        window._section_shortcuts.append(shortcut)


def build_status_bar(window: MainWindow) -> None:
    """Create the status bar up front and make it a touch shorter.

    QMainWindow builds the status bar lazily, on the first ``showMessage``
    call — so without this it appears to pop in below the filmstrip the
    first time the user saves, shifting the whole layout.  Instantiating it
    here reserves the space from launch so it is always visible.  The font
    is shrunk a point and the size grip dropped to keep it compact.
    """
    bar = require(window.statusBar())
    window._statusbar = bar
    bar.setSizeGripEnabled(False)
    font = bar.font()
    font.setPointSizeF(max(1.0, font.pointSizeF() - 1.0))
    bar.setFont(font)
    bar.setMaximumHeight(bar.fontMetrics().height() + 4)


def connect_signals(window: MainWindow) -> None:
    """Wire every state / view / property signal to the window's slots."""
    # State → views
    window._state.project_changed.connect(window._on_project_changed)
    window._state.section_changed.connect(window._on_section_changed)
    window._state.atlas_changed.connect(window._on_atlas_loaded)
    window._state.atlas_error.connect(window._on_atlas_error)
    window._state.dirty_changed.connect(window._on_dirty_changed)
    # Controllers emit these instead of poking window internals. A batch op
    # mutates the visible section in place, so reload the active view too.
    # Lambdas defer _statusbar lookup (built after connect_signals).
    window._state.sections_changed.connect(lambda: window.sync_dependent_ui(reload_active=True))
    window._state.status_message.connect(lambda m: window._statusbar.showMessage(m, 3000))
    window._state.deepslice_running_changed.connect(
        lambda running: window._update_deepslice_enabled(running=running)
    )
    # The section list changed (add/remove/reorder) → rebuild list-dependent UI.
    window._state.structure_changed.connect(window._on_structure_changed)

    # Overview interactions
    window._overview.section_activated.connect(window._on_section_activated)
    window._overview.section_selected.connect(window._state.set_section)
    window._overview.sections_reordered.connect(window._project.on_sections_reordered)
    window._overview.remove_requested.connect(window._project.remove_sections)
    window._overview.images_dropped.connect(window._project.on_images_dropped)

    # Filmstrip
    window._filmstrip.section_selected.connect(window._state.set_section)
    window._filmstrip.thumbnail_loaded.connect(window._on_thumbnail_loaded)

    # Properties
    flip = window._props.prep.flip
    flip.flip_h_changed.connect(window._on_flip_h_changed)
    flip.flip_v_changed.connect(window._on_flip_v_changed)
    mask = window._props.prep.mask_box
    mask.visibility_changed.connect(window._prep.set_mask_visible)
    mask.opacity_changed.connect(window._prep.set_mask_opacity)
    mask.color_changed.connect(window._prep.set_mask_color)
    mask.negative_changed.connect(window._prep.set_mask_negative)
    mask.draw_mode_changed.connect(window._prep.set_draw_mode)
    window._prep.draw_mode_changed.connect(mask.set_draw_mode)
    mask.brush_size_changed.connect(window._prep.set_brush_size)
    window._prep.brush_size_changed.connect(mask.set_brush_size)
    mask.autodetect_requested.connect(window._on_prep_autodetect_requested)
    mask.clear_requested.connect(window._on_prep_clear_mask_requested)
    mask.erode_requested.connect(lambda px: window._prep.apply_morph(px, "erode"))
    mask.expand_requested.connect(lambda px: window._prep.apply_morph(px, "expand"))
    # Overlay lives in both Align and Warp pages with independent state.
    for overlay in (window._props.align.overlay, window._props.warp.overlay):
        overlay.opacity_changed.connect(window._on_opacity_changed)
        overlay.color_changed.connect(window._panel.set_outline_color)
        overlay.mode_changed.connect(window._panel.set_overlay_mode)

    # PrepView edits
    window._prep.mask_negative_changed.connect(mask.set_negative)
    window._prep.mask_visibility_changed.connect(mask.set_visible_state)

    # AlignView navigator drives the anchoring; alignments_updated fires
    # when the user explicitly saves or clears, triggering re-interpolation.
    window._align.anchoring_changed.connect(window._on_anchoring_changed)
    window._align.alignments_updated.connect(window._on_alignments_updated)
    window._props.warp.cp.style_changed.connect(window._warp.on_cp_style_changed)
    window._props.warp.cp.autogen_requested.connect(window._jobs.auto_generate_warp_cps)
    window._props.warp.cp.edit_params_requested.connect(window._jobs.edit_elastix_params)

    # Local-changes bars (Save / Clear edits / Reset) and per-view dirty signals.
    # SaveController owns the parameterized save/revert/clear dispatch.
    view_bindings = (
        ("prep", window._prep, window._props.prep),
        ("align", window._align, window._props.align),
        ("warp", window._warp, window._props.warp),
    )
    for step, view, page in view_bindings:
        window._saves.register(step, view, page)
        # The views mutate AppState directly (the single source of truth) and
        # SaveController drives the save bars off AppState.dirty_changed — no
        # per-view dirty signal to mirror here.
        page.save_bar.save_requested.connect(lambda s=step: window._saves.on_save(s))
        page.save_bar.revert_requested.connect(lambda s=step: window._saves.on_revert(s))
        page.save_bar.reset_requested.connect(lambda s=step: window._saves.on_clear(s))

    # A prep save/clear that flips the section invalidates its alignment+warp.
    window._prep.alignment_invalidated.connect(window._project.on_prep_invalidated_alignment)
    # CP add/delete changes the warp dot even when the dirty flag is unchanged
    # (e.g. removing the last CP → gray).
    window._warp.cp_changed.connect(window._refresh_current_step_dot)
