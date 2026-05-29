"""Context-sensitive right-side properties panel.

Contains a QStackedWidget with one page per view mode.
MainWindow switches pages via set_mode().
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QStackedWidget, QVBoxLayout, QWidget

from verso.gui.widgets.properties.align_page import AlignPage
from verso.gui.widgets.properties.overview_page import OverviewPage
from verso.gui.widgets.properties.prep_page import PrepPage
from verso.gui.widgets.properties.warp_page import WarpPage


class PropertiesPanel(QWidget):
    """Outer container that switches between the per-mode properties pages."""

    flip_h_changed = pyqtSignal(bool)
    flip_v_changed = pyqtSignal(bool)
    mask_visibility_changed = pyqtSignal(bool)
    lr_visibility_changed = pyqtSignal(bool)
    mask_opacity_changed = pyqtSignal(float)
    mask_color_changed = pyqtSignal(tuple)
    mask_negative_changed = pyqtSignal(bool)
    mask_draw_mode_changed = pyqtSignal(str)  # "freehand" | "brush"
    brush_size_changed = pyqtSignal(int)
    autodetect_requested = pyqtSignal()
    clear_mask_requested = pyqtSignal()
    erode_mask_requested = pyqtSignal(int)
    expand_mask_requested = pyqtSignal(int)
    lr_set_all_left_requested = pyqtSignal()
    lr_set_all_right_requested = pyqtSignal()
    lr_draw_mode_toggled = pyqtSignal(bool)
    lr_apply_requested = pyqtSignal()
    lr_cancel_requested = pyqtSignal()
    lr_clear_requested = pyqtSignal()
    lr_opacity_changed = pyqtSignal(float)
    lr_left_color_changed = pyqtSignal(tuple)
    lr_right_color_changed = pyqtSignal(tuple)
    opacity_changed = pyqtSignal(float)
    overlay_color_changed = pyqtSignal(tuple)  # (r, g, b) — outline color
    overlay_mode_changed = pyqtSignal(str)  # "annotation" | "outline" | "reference"
    cp_style_changed = pyqtSignal(int, str, str)  # size, shape, color

    _MODES = ("overview", "prep", "align", "warp")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(130)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self._stack = QStackedWidget()
        self._overview_page = OverviewPage()
        self._prep_page = PrepPage()
        self._align_page = AlignPage()
        self._warp_page = WarpPage()

        self._stack.addWidget(self._overview_page)
        self._stack.addWidget(self._prep_page)
        self._stack.addWidget(self._align_page)
        self._stack.addWidget(self._warp_page)
        layout.addWidget(self._stack)

        self._wire_prep_signals()
        self._wire_align_warp_signals()

    def _wire_prep_signals(self) -> None:
        flip = self._prep_page.flip
        flip.flip_h_changed.connect(self.flip_h_changed)
        flip.flip_v_changed.connect(self.flip_v_changed)

        mask = self._prep_page.mask
        mask.visibility_changed.connect(self.mask_visibility_changed)
        mask.opacity_changed.connect(self.mask_opacity_changed)
        mask.color_changed.connect(self.mask_color_changed)
        mask.negative_changed.connect(self.mask_negative_changed)
        mask.draw_mode_changed.connect(self.mask_draw_mode_changed)
        mask.brush_size_changed.connect(self.brush_size_changed)
        mask.autodetect_requested.connect(self.autodetect_requested)
        mask.clear_requested.connect(self.clear_mask_requested)
        mask.erode_requested.connect(self.erode_mask_requested)
        mask.expand_requested.connect(self.expand_mask_requested)

        hemi = self._prep_page.hemisphere
        hemi.visibility_changed.connect(self.lr_visibility_changed)
        hemi.set_all_left_requested.connect(self.lr_set_all_left_requested)
        hemi.set_all_right_requested.connect(self.lr_set_all_right_requested)
        hemi.draw_mode_toggled.connect(self.lr_draw_mode_toggled)
        hemi.apply_requested.connect(self.lr_apply_requested)
        hemi.cancel_requested.connect(self.lr_cancel_requested)
        hemi.clear_requested.connect(self.lr_clear_requested)
        hemi.opacity_changed.connect(self.lr_opacity_changed)
        hemi.left_color_changed.connect(self.lr_left_color_changed)
        hemi.right_color_changed.connect(self.lr_right_color_changed)

    def _wire_align_warp_signals(self) -> None:
        # Each page owns its own Overlay/Proposal with independent UI state.
        # Both pages forward overlay changes to the same panel-level signals.
        for page in (self._align_page, self._warp_page):
            page.overlay.opacity_changed.connect(self.opacity_changed)
            page.overlay.color_changed.connect(self.overlay_color_changed)
            page.overlay.mode_changed.connect(self.overlay_mode_changed)
        self._warp_page.cp.style_changed.connect(self.cp_style_changed)

    def set_mode(self, mode: str) -> None:
        self._stack.setCurrentIndex(self._MODES.index(mode))
        # Push the newly-active page's overlay settings to the canvas so it
        # matches the buttons the user is now looking at.
        if mode == "align":
            self._align_page.overlay.emit_current_state()
        elif mode == "warp":
            self._warp_page.overlay.emit_current_state()

    def update_section(self, section, mode: str) -> None:
        if mode == "overview":
            self._overview_page.update_section(section)
        elif mode == "prep":
            self._prep_page.update_section(section)
        elif mode == "align":
            self._align_page.update_section(section)
        elif mode == "warp":
            self._warp_page.update_section(section)

    def update_ap_plot(self, sections: list, current_index: int) -> None:
        self._align_page.ap_plot.update_plot(sections, current_index)

    def set_mask_negative(self, negative: bool) -> None:
        self._prep_page.mask.set_negative(negative)

    def set_mask_visible(self, visible: bool) -> None:
        self._prep_page.mask.set_visible_state(visible)

    def set_lr_status(self, text: str) -> None:
        self._prep_page.hemisphere.set_status(text)

    def set_lr_draw_active(self, active: bool) -> None:
        self._prep_page.hemisphere.set_draw_active(active)

    def apply_cp_style(self, size: int, shape: str, color: str) -> None:
        """Initialise CP style widgets from saved settings (no signal emitted)."""
        self._warp_page.cp.apply_style(size, shape, color)

    def set_brush_size(self, size: int) -> None:
        self._prep_page.mask.set_brush_size(size)
