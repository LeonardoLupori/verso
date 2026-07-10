"""Selected-annotation section — editing controls for the active annotation.

Shares a colour/title/opacity header across both annotation types, then shows
type-specific controls below it: point size and Add/Remove tools for point
series, Freehand/Brush tools and a brush-size slider for areas. A pure view:
it emits intent signals and renders whatever :meth:`update_selected` is handed.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QColorDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from verso.engine.model.annotation import Annotation, AreaAnnotation
from verso.gui.utils import colored_svg_pixmap
from verso.gui.widgets.properties._common import color_swatch_style, make_segmented_buttons


class _RenameLineEdit(QLineEdit):
    """A label-like line edit that becomes editable on double-click.

    Emits :attr:`rename_committed` with the new text when editing finishes
    (Return or focus-out), then returns to its read-only, frameless look.
    """

    rename_committed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFrame(False)
        self._editing = False
        self.editingFinished.connect(self._commit)

    def mouseDoubleClickEvent(self, event) -> None:
        if self.isReadOnly():
            self.setReadOnly(False)
            self.setFrame(True)
            self._editing = True
            self.selectAll()
            self.setFocus()
        super().mouseDoubleClickEvent(event)

    def _commit(self) -> None:
        if not self._editing:
            return
        self._editing = False
        self.setReadOnly(True)
        self.setFrame(False)
        self.rename_committed.emit(self.text())


class EditAnnotationsBox(QGroupBox):
    color_changed = pyqtSignal(tuple)  # (r, g, b)
    opacity_changed = pyqtSignal(float)
    rename_requested = pyqtSignal(str)
    point_size_changed = pyqtSignal(int)
    tool_changed = pyqtSignal(str)  # "add" | "remove"
    area_tool_changed = pyqtSignal(str)  # "brush" | "freehand"
    brush_size_changed = pyqtSignal(int)

    def __init__(self) -> None:
        super().__init__("Edit annotation")
        self._color_rgb: tuple[int, int, int] = (255, 64, 64)
        layout = QVBoxLayout(self)

        # Fixed caption/value widths so the Opacity, Point size, and Brush
        # sliders (never all shown together, but each paired with Opacity)
        # start and end at the same x position.
        fm = self.fontMetrics()
        self._caption_width = (
            max(fm.horizontalAdvance(t) for t in ("Opacity", "Point size", "Brush")) + 4
        )
        self._value_width = max(fm.horizontalAdvance(t) for t in ("1.00", "40", "200")) + 4

        # Row 1: colour swatch + type icon + title (double-click to rename)
        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(20, 20)
        self._color_btn.setToolTip("Colour")
        self._color_btn.clicked.connect(self._on_color)
        self._type_icon = QLabel()
        self._type_icon.setPixmap(colored_svg_pixmap("scatter.svg", "#aaaaaa", 16))
        self._type_icon.setFixedSize(18, 18)
        self._type_icon.setToolTip("Point series")
        self._title_edit = _RenameLineEdit()
        self._title_edit.setToolTip("Double-click to rename")
        self._title_edit.rename_committed.connect(self.rename_requested)
        title_row = QHBoxLayout()
        title_row.addWidget(self._color_btn)
        title_row.addWidget(self._type_icon)
        title_row.addWidget(self._title_edit, stretch=1)
        layout.addLayout(title_row)

        # Row 2: element count (Points/Sections, depending on type)
        self._count_caption = QLabel("Points")
        self._count_label = QLabel("-")
        self._count_label.setStyleSheet("color: #888;")
        count_row = QHBoxLayout()
        count_row.addWidget(self._count_caption)
        count_row.addWidget(self._count_label, stretch=1)
        layout.addLayout(count_row)

        # Row 3: opacity slider
        self._opacity_value = QLabel("1.00")
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(100)
        self._opacity_slider.setMinimumWidth(20)
        self._opacity_slider.valueChanged.connect(self._emit_opacity)
        layout.addLayout(self._slider_row("Opacity", self._opacity_slider, self._opacity_value))

        layout.addWidget(self._build_point_controls())
        layout.addWidget(self._build_area_controls())

    def _slider_row(self, caption: str, slider: QSlider, value_label: QLabel) -> QHBoxLayout:
        """A caption + slider + numeric-readout row with a fixed-width caption
        and readout, so sliders on different rows line up."""
        caption_label = QLabel(caption)
        caption_label.setFixedWidth(self._caption_width)
        value_label.setFixedWidth(self._value_width)
        row = QHBoxLayout()
        row.addWidget(caption_label)
        row.addWidget(slider, stretch=1)
        row.addWidget(value_label)
        return row

    def _build_point_controls(self) -> QWidget:
        """Point-series-only controls: point size + Add/Remove tools."""
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)

        self._size_value = QLabel("3")
        self._size_slider = QSlider(Qt.Orientation.Horizontal)
        self._size_slider.setRange(1, 20)
        self._size_slider.setValue(3)
        self._size_slider.setMinimumWidth(20)
        self._size_slider.valueChanged.connect(self._emit_point_size)
        v.addLayout(self._slider_row("Point size", self._size_slider, self._size_value))

        tool_row, self._tool_btns, tool_group = make_segmented_buttons(
            self,
            [("add", "Add"), ("remove", "Remove")],
            tooltips=["Click to add a point (A)", "Drag a lasso to remove points (R)"],
            initial_key="add",
        )
        tool_group.buttonClicked.connect(self._on_tool_clicked)
        v.addLayout(tool_row)

        self._point_widget = box
        return box

    def _build_area_controls(self) -> QWidget:
        """Area-only controls: Freehand/Brush tools, erase hint, brush size."""
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)

        area_tool_row, self._area_tool_btns, area_group = make_segmented_buttons(
            self,
            [("freehand", "Freehand"), ("brush", "Brush")],
            tooltips=["Fill an outline (F)", "Paint the mask (B)"],
            initial_key="brush",
        )
        area_group.buttonClicked.connect(self._on_area_tool_clicked)
        v.addLayout(area_tool_row)

        self._area_hint = QLabel("Hold Shift to erase")
        self._area_hint.setStyleSheet("color: #888; font-size: 11px;")
        self._area_hint.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        v.addWidget(self._area_hint)

        self._brush_value = QLabel("20")
        self._brush_slider = QSlider(Qt.Orientation.Horizontal)
        self._brush_slider.setRange(5, 200)
        self._brush_slider.setValue(20)
        self._brush_slider.setMinimumWidth(20)
        self._brush_slider.valueChanged.connect(self._emit_brush_size)
        v.addLayout(self._slider_row("Brush", self._brush_slider, self._brush_value))

        self._area_widget = box
        return box

    # ------------------------------------------------------------------
    # Population (driven by AnnotatePage / AnnotationController)
    # ------------------------------------------------------------------

    def update_selected(self, ann: Annotation) -> None:
        """Render *ann*'s properties and show the controls for its type."""
        is_area = isinstance(ann, AreaAnnotation)

        icon = "area.svg" if is_area else "scatter.svg"
        self._type_icon.setPixmap(colored_svg_pixmap(icon, "#aaaaaa", 16))
        self._type_icon.setToolTip("Area" if is_area else "Point series")

        self._color_rgb = ann.color
        self._color_btn.setStyleSheet(color_swatch_style(ann.color))

        self._title_edit.setText(ann.title)

        self._opacity_slider.blockSignals(True)
        self._opacity_slider.setValue(round(ann.opacity * 100))
        self._opacity_slider.blockSignals(False)
        self._opacity_value.setText(f"{ann.opacity:.2f}")

        if is_area:
            n = sum(1 for m in ann.masks.values() if m.any())
            self._count_caption.setText("Sections")
            self._count_label.setText(str(n))
        else:
            self._count_caption.setText("Points")
            self._count_label.setText(str(len(ann.points)))
            self._size_slider.blockSignals(True)
            self._size_slider.setValue(ann.point_size)
            self._size_slider.blockSignals(False)
            self._size_value.setText(str(ann.point_size))

        self._point_widget.setVisible(not is_area)
        self._area_widget.setVisible(is_area)

    def clear(self) -> None:
        """Blank the controls when no annotation is selected."""
        self._title_edit.setText("")
        self._count_label.setText("-")

    def set_tool(self, tool: str) -> None:
        """Reflect the active point tool in the segmented buttons (no signal)."""
        btn = self._tool_btns.get(tool)
        if btn is not None and not btn.isChecked():
            btn.blockSignals(True)
            btn.setChecked(True)
            btn.blockSignals(False)

    def set_area_tool(self, tool: str) -> None:
        """Reflect the active area tool in the segmented buttons (no signal)."""
        btn = self._area_tool_btns.get(tool)
        if btn is not None and not btn.isChecked():
            btn.blockSignals(True)
            btn.setChecked(True)
            btn.blockSignals(False)

    def set_brush_size(self, size: int) -> None:
        """Reflect the brush size in the slider (no signal)."""
        size = max(5, min(200, int(size)))
        self._brush_slider.blockSignals(True)
        self._brush_slider.setValue(size)
        self._brush_slider.blockSignals(False)
        self._brush_value.setText(str(size))

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _emit_opacity(self) -> None:
        opacity = self._opacity_slider.value() / 100.0
        self._opacity_value.setText(f"{opacity:.2f}")
        self.opacity_changed.emit(opacity)

    def _emit_point_size(self) -> None:
        size = self._size_slider.value()
        self._size_value.setText(str(size))
        self.point_size_changed.emit(size)

    def _emit_brush_size(self) -> None:
        size = self._brush_slider.value()
        self._brush_value.setText(str(size))
        self.brush_size_changed.emit(size)

    def _on_color(self) -> None:
        color = QColorDialog.getColor(QColor(*self._color_rgb), self, "Annotation colour")
        if color.isValid():
            self._color_rgb = (color.red(), color.green(), color.blue())
            self._color_btn.setStyleSheet(color_swatch_style(self._color_rgb))
            self.color_changed.emit(self._color_rgb)

    def _on_tool_clicked(self, btn: QPushButton) -> None:
        for key, b in self._tool_btns.items():
            if b is btn:
                self.tool_changed.emit(key)
                return

    def _on_area_tool_clicked(self, btn: QPushButton) -> None:
        for key, b in self._area_tool_btns.items():
            if b is btn:
                self.area_tool_changed.emit(key)
                return
