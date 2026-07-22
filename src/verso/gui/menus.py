"""Menu-bar and view-switcher toolbar construction for :class:`MainWindow`.

Pure widget construction split out of the window. ``build_menus`` and
``build_toolbar`` receive the window, create the actions/buttons, wire them to the
window's handlers and controllers, and stash the handles the window needs to keep
(``_act_*``, ``_view_buttons``, ``_project_label``) back onto it. No behaviour
lives here — moving a menu item is a one-file edit that never touches the window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QLabel, QPushButton, QSizePolicy, QToolBar, QWidget

from verso.gui.utils import require

if TYPE_CHECKING:
    from verso.gui.main_window import MainWindow

# Indices into the central QStackedWidget; shared with MainWindow._switch_view.
VIEW_OVERVIEW = 0
VIEW_PREP = 1
VIEW_ALIGN = 2
VIEW_WARP = 3
VIEW_ANNOTATE = 4


def build_menus(window: MainWindow) -> None:
    """Build the File / Image / Batch / Export / Help menus on ``window``."""
    mb = require(window.menuBar())

    file_menu = require(mb.addMenu("&File"))

    act_new = QAction("&New Project…", window)
    act_new.setShortcut(QKeySequence.StandardKey.New)
    act_new.triggered.connect(window._project.new_project)
    file_menu.addAction(act_new)

    act_open = QAction("&Open Project…", window)
    act_open.setShortcut(QKeySequence.StandardKey.Open)
    act_open.triggered.connect(window._project.open_project)
    file_menu.addAction(act_open)

    file_menu.addSeparator()

    import_menu = require(file_menu.addMenu("&Import"))

    act_open_qn = QAction("&QuickNII XML file…", window)
    act_open_qn.triggered.connect(window._project.open_quicknii)
    import_menu.addAction(act_open_qn)

    act_open_va = QAction("&VisuAlign JSON file…", window)
    act_open_va.triggered.connect(window._project.open_visualign)
    import_menu.addAction(act_open_va)

    import_menu.addSeparator()

    act_import_settings = QAction("&Settings from VERSO project file…", window)
    act_import_settings.triggered.connect(window._project.import_settings_from_project)
    import_menu.addAction(act_import_settings)

    file_menu.addSeparator()

    act_save = QAction("&Save all", window)
    act_save.setShortcut(QKeySequence.StandardKey.Save)
    act_save.setToolTip("Save all unsaved edits across every slice (Ctrl+S)")
    act_save.triggered.connect(window._project.save_all)
    file_menu.addAction(act_save)

    act_save_as = QAction("Save project &as…", window)
    act_save_as.setShortcut(QKeySequence.StandardKey.SaveAs)
    act_save_as.triggered.connect(window._project.save_project_as)
    file_menu.addAction(act_save_as)

    file_menu.addSeparator()

    act_quit = QAction("&Quit", window)
    act_quit.setShortcut(QKeySequence.StandardKey.Quit)
    act_quit.triggered.connect(window.close)
    file_menu.addAction(act_quit)

    images_menu = require(mb.addMenu("&Image"))
    act_adjust = QAction("&Channels…", window)
    act_adjust.triggered.connect(window._open_brightness_dialog)
    images_menu.addAction(act_adjust)
    act_reorder = QAction("Reorder slices based on &filename…", window)
    act_reorder.triggered.connect(window._project.reorder_by_filename)
    images_menu.addAction(act_reorder)
    images_menu.addSeparator()
    act_add_images = QAction("&Add images to project…", window)
    act_add_images.triggered.connect(window._project.add_images)
    images_menu.addAction(act_add_images)

    batch_menu = require(mb.addMenu("&Batch"))

    preprocess_menu = require(batch_menu.addMenu("&Preprocess"))
    act_batch_mask = QAction("Autodetect slice mask for &all slices", window)
    act_batch_mask.triggered.connect(window._jobs.batch_autodetect_masks)
    preprocess_menu.addAction(act_batch_mask)
    preprocess_menu.addSeparator()
    window._act_clear_all_slice_masks = QAction("Clear all &slice masks…", window)
    window._act_clear_all_slice_masks.setEnabled(False)
    window._act_clear_all_slice_masks.triggered.connect(window._jobs.clear_all_slice_masks)
    preprocess_menu.addAction(window._act_clear_all_slice_masks)

    align_menu = require(batch_menu.addMenu("&Align"))
    window._act_deepslice = QAction("Run &DeepSlice", window)
    window._act_deepslice.setEnabled(False)
    window._act_deepslice.triggered.connect(window._jobs.run_deepslice)
    align_menu.addAction(window._act_deepslice)

    window._act_default_proposal = QAction("&Default proposal", window)
    window._act_default_proposal.setEnabled(False)
    window._act_default_proposal.triggered.connect(window._jobs.revert_to_default_proposal)
    align_menu.addAction(window._act_default_proposal)

    align_menu.addSeparator()
    window._act_clear_all_alignments = QAction("&Clear all alignments…", window)
    window._act_clear_all_alignments.setEnabled(False)
    window._act_clear_all_alignments.triggered.connect(window._jobs.clear_all_alignments)
    align_menu.addAction(window._act_clear_all_alignments)

    warp_menu = require(batch_menu.addMenu("&Warp"))
    window._act_batch_auto_cp = QAction("&Auto-generate control points for all slices…", window)
    window._act_batch_auto_cp.setEnabled(False)
    window._act_batch_auto_cp.triggered.connect(window._jobs.batch_auto_generate_warps)
    warp_menu.addAction(window._act_batch_auto_cp)
    warp_menu.addSeparator()
    window._act_clear_manual_cps = QAction("Clear all &manual control points…", window)
    window._act_clear_manual_cps.setEnabled(False)
    window._act_clear_manual_cps.triggered.connect(window._jobs.clear_all_manual_cps)
    warp_menu.addAction(window._act_clear_manual_cps)
    window._act_clear_auto_cps = QAction("Clear all a&utomatic control points…", window)
    window._act_clear_auto_cps.setEnabled(False)
    window._act_clear_auto_cps.triggered.connect(window._jobs.clear_all_auto_cps)
    warp_menu.addAction(window._act_clear_auto_cps)

    export_menu = require(mb.addMenu("&Export"))
    act_export_images = QAction("Images with atlas &overlay…", window)
    act_export_images.triggered.connect(window._export.export_images_with_overlay)
    export_menu.addAction(act_export_images)

    act_export_stack = QAction("Aligned section &stack…", window)
    act_export_stack.triggered.connect(window._export.export_aligned_stack)
    export_menu.addAction(act_export_stack)

    export_menu.addSeparator()

    quint_menu = require(export_menu.addMenu("For &QUINT"))

    act_export_qn_xml = QAction("QuickNII &XML…", window)
    act_export_qn_xml.triggered.connect(window._export.export_quicknii_xml)
    quint_menu.addAction(act_export_qn_xml)

    act_export_qn = QAction("QuickNII &JSON…", window)
    act_export_qn.triggered.connect(window._export.export_quicknii)
    quint_menu.addAction(act_export_qn)

    act_export_va = QAction("&VisuAlign JSON…", window)
    act_export_va.triggered.connect(window._export.export_visualign)
    quint_menu.addAction(act_export_va)

    export_menu.addSeparator()

    quantify_menu = require(export_menu.addMenu("&Quantify"))

    act_q_intensity = QAction("&Intensity…", window)
    act_q_intensity.triggered.connect(window._export.quantify_intensity)
    quantify_menu.addAction(act_q_intensity)

    act_q_dots = QAction("&Dots annotations…", window)
    act_q_dots.triggered.connect(window._export.quantify_dots)
    quantify_menu.addAction(act_q_dots)

    act_q_area = QAction("&Area annotations…", window)
    act_q_area.triggered.connect(window._export.quantify_area)
    quantify_menu.addAction(act_q_area)

    help_menu = require(mb.addMenu("&Help"))
    act_atlas_info = QAction("&Atlas info…", window)
    act_atlas_info.triggered.connect(window._show_atlas_info)
    help_menu.addAction(act_atlas_info)
    act_project_info = QAction("&Project info…", window)
    act_project_info.triggered.connect(window._show_project_info)
    help_menu.addAction(act_project_info)
    help_menu.addSeparator()
    act_open_logs = QAction("Open &log folder", window)
    act_open_logs.triggered.connect(window._open_log_folder)
    help_menu.addAction(act_open_logs)
    help_menu.addSeparator()
    act_about = QAction("&About VERSO…", window)
    act_about.triggered.connect(window._show_about)
    help_menu.addAction(act_about)


def build_toolbar(window: MainWindow) -> None:
    """Build the top view-switcher toolbar (Overview / Prep / Align / Warp)."""
    tb = QToolBar("Views")
    tb.setMovable(False)
    tb.setFloatable(False)
    tb.setStyleSheet(
        "QToolBar { background: #2a2a2a; border-bottom: 1px solid #444; "
        "spacing: 4px; padding: 4px; }"
    )
    window.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

    window._view_buttons = []
    view_specs = [
        ("Overview", VIEW_OVERVIEW),
        ("Preprocess", VIEW_PREP),
        ("Align", VIEW_ALIGN),
        ("Warp", VIEW_WARP),
        ("Annotate", VIEW_ANNOTATE),
    ]
    for label, idx in view_specs:
        btn = QPushButton(label)
        btn.setCheckable(True)
        btn.setFixedHeight(28)
        btn.setStyleSheet(
            "QPushButton { border-radius: 4px; padding: 2px 14px; color: #ccc;"
            " background: #3a3a3a; border: 1px solid #555; }"
            "QPushButton:checked { background: #1e5a8a; color: #fff; border-color: #1e5a8a; }"
            "QPushButton:hover:!checked { background: #4a4a4a; }"
            "QPushButton:disabled { color: #555; background: #2e2e2e; border-color: #3a3a3a; }"
        )
        btn.clicked.connect(lambda _checked, i=idx: window._switch_view(i))
        if idx != VIEW_OVERVIEW:
            btn.setEnabled(False)
        window._view_buttons.append(btn)
        tb.addWidget(btn)

    spacer = QWidget()
    spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    tb.addWidget(spacer)

    window._project_label = QLabel("")
    window._project_label.setStyleSheet("color: #888; font-size: 11px; padding-right: 8px;")
    tb.addWidget(window._project_label)
