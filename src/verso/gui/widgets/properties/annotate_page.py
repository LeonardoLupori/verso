"""Properties page for the Annotate view — the annotation manager.

Lets the user create annotations (empty or imported from a CSV), delete them,
pick the active one from a dropdown, and edit its visibility, colour, opacity and
title. A Save button (mirrored by Ctrl+S) persists the annotation set.

The page is a pure view: it emits intent signals and renders whatever
:meth:`set_annotations` is handed. All state lives in
:class:`~verso.gui.controllers.annotation_controller.AnnotationController`.
"""

from __future__ import annotations

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor
from PyQt6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from verso.engine.model.annotation import Annotation, AreaAnnotation
from verso.engine.model.project import Section
from verso.gui.utils import colored_svg_pixmap
from verso.gui.widgets.properties._common import (
    color_swatch_style,
    colored_icon,
    eye_icon,
    make_segmented_buttons,
)


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


class AnnotatePage(QWidget):
    new_point_requested = pyqtSignal()
    new_area_requested = pyqtSignal()
    import_requested = pyqtSignal()
    delete_requested = pyqtSignal()
    active_changed = pyqtSignal(int)
    visibility_changed = pyqtSignal(bool)
    color_changed = pyqtSignal(tuple)  # (r, g, b)
    opacity_changed = pyqtSignal(float)
    rename_requested = pyqtSignal(str)
    tool_changed = pyqtSignal(str)  # "add" | "remove"
    save_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color_rgb: tuple[int, int, int] = (255, 64, 64)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        outer.addWidget(scroll, stretch=1)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(8)
        scroll.setWidget(content)

        layout.addWidget(self._build_manager_box())
        layout.addWidget(self._build_selected_box())
        layout.addStretch()

        self._save_box = self._build_save_box()
        layout.addWidget(self._save_box)

        self.set_annotations([], -1)
        self.set_dirty(False)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_manager_box(self) -> QGroupBox:
        box = QGroupBox("Annotations")
        v = QVBoxLayout(box)

        actions = QHBoxLayout()
        actions.setSpacing(4)
        self._new_btn = QPushButton("New ▾")
        self._new_btn.setToolTip("Create an empty annotation")
        new_menu = QMenu(self._new_btn)
        act_points = QAction("Point series", new_menu)
        act_points.triggered.connect(self.new_point_requested)
        act_area = QAction("Area", new_menu)
        act_area.triggered.connect(self.new_area_requested)
        new_menu.addAction(act_points)
        new_menu.addAction(act_area)
        self._new_btn.setMenu(new_menu)
        self._import_btn = QPushButton("From CSV…")
        self._import_btn.setToolTip("Create a point series from a CSV of points")
        self._import_btn.clicked.connect(self.import_requested)
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setToolTip("Delete the selected annotation")
        self._delete_btn.clicked.connect(self.delete_requested)
        for btn in (self._new_btn, self._import_btn, self._delete_btn):
            actions.addWidget(btn)
        v.addLayout(actions)

        self._combo = QComboBox()
        self._combo.currentIndexChanged.connect(self._on_combo_changed)
        v.addWidget(self._combo)
        return box

    def _build_selected_box(self) -> QGroupBox:
        box = QGroupBox("Selected annotation")
        form = QFormLayout(box)

        title_row = QHBoxLayout()
        title_row.setSpacing(6)
        self._type_icon = QLabel()
        self._type_icon.setPixmap(colored_svg_pixmap("scatter.svg", "#aaaaaa", 16))
        self._type_icon.setFixedSize(18, 18)
        self._type_icon.setToolTip("Point series")
        self._eye_btn = QPushButton()
        self._eye_btn.setCheckable(True)
        self._eye_btn.setChecked(True)
        self._eye_btn.setFixedSize(24, 24)
        self._eye_btn.setFlat(True)
        self._eye_btn.setIcon(eye_icon(True))
        self._eye_btn.setIconSize(QSize(16, 16))
        self._eye_btn.toggled.connect(self._on_eye_toggled)
        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(20, 20)
        self._color_btn.setToolTip("Colour")
        self._color_btn.clicked.connect(self._on_color)
        title_row.addWidget(self._type_icon)
        title_row.addWidget(self._eye_btn)
        title_row.addWidget(self._color_btn)
        title_row.addStretch()
        form.addRow(title_row)

        self._title_edit = _RenameLineEdit()
        self._title_edit.setToolTip("Double-click to rename")
        self._title_edit.rename_committed.connect(self.rename_requested)
        form.addRow("Title:", self._title_edit)

        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(100)
        self._opacity_slider.valueChanged.connect(
            lambda val: self.opacity_changed.emit(val / 100.0)
        )
        form.addRow("Opacity:", self._opacity_slider)

        self._count_label = QLabel("-")
        self._count_label.setStyleSheet("color: #888;")
        self._count_caption = QLabel("Points:")
        form.addRow(self._count_caption, self._count_label)

        # Point-series tools (Add/Remove). Hidden while an Area is selected; area
        # brush/freehand tools are added in D5c.
        tool_row, self._tool_btns, tool_group = make_segmented_buttons(
            self,
            [("add", "Add"), ("remove", "Remove")],
            tooltips=["Click to add a point (A)", "Drag a lasso to remove points (R)"],
            initial_key="add",
        )
        tool_group.buttonClicked.connect(self._on_tool_clicked)
        self._tool_caption = QLabel("Tool:")
        self._tool_widget = QWidget()
        self._tool_widget.setLayout(tool_row)
        form.addRow(self._tool_caption, self._tool_widget)

        self._selected_box = box
        return box

    def _build_save_box(self) -> QGroupBox:
        box = QGroupBox("Local changes")
        v = QVBoxLayout(box)
        self._save_btn = QPushButton("Save annotations")
        self._save_btn.setIcon(colored_icon("save.svg", "#ffffff"))
        self._save_btn.setIconSize(QSize(14, 14))
        self._save_btn.setToolTip("Write annotations to the project's annotations/ folder")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self.save_requested)
        v.addWidget(self._save_btn)
        return box

    # ------------------------------------------------------------------
    # Population (driven by AnnotationController)
    # ------------------------------------------------------------------

    def set_annotations(self, annotations: list[Annotation], active_index: int) -> None:
        """Rebuild the dropdown and reflect the active annotation's controls."""
        self._combo.blockSignals(True)
        self._combo.clear()
        for ann in annotations:
            self._combo.addItem(ann.title)
        if 0 <= active_index < len(annotations):
            self._combo.setCurrentIndex(active_index)
        self._combo.blockSignals(False)

        has_active = 0 <= active_index < len(annotations)
        self._delete_btn.setEnabled(has_active)
        self._selected_box.setEnabled(has_active)
        if has_active:
            self._update_selected(annotations[active_index])
        else:
            self._title_edit.setText("")
            self._count_label.setText("-")

    def _update_selected(self, ann: Annotation) -> None:
        is_area = isinstance(ann, AreaAnnotation)

        icon = "area.svg" if is_area else "scatter.svg"
        self._type_icon.setPixmap(colored_svg_pixmap(icon, "#aaaaaa", 16))
        self._type_icon.setToolTip("Area" if is_area else "Point series")

        self._eye_btn.blockSignals(True)
        self._eye_btn.setChecked(ann.visible)
        self._eye_btn.setIcon(eye_icon(ann.visible))
        self._eye_btn.blockSignals(False)

        self._color_rgb = ann.color
        self._color_btn.setStyleSheet(color_swatch_style(ann.color))

        self._title_edit.setText(ann.title)

        self._opacity_slider.blockSignals(True)
        self._opacity_slider.setValue(round(ann.opacity * 100))
        self._opacity_slider.blockSignals(False)

        if is_area:
            n = sum(1 for m in ann.masks.values() if m.any())
            self._count_caption.setText("Sections:")
            self._count_label.setText(str(n))
        else:
            self._count_caption.setText("Points:")
            self._count_label.setText(str(len(ann.points)))

        # Point tools apply only to point series (area tools arrive in D5c).
        self._tool_caption.setVisible(not is_area)
        self._tool_widget.setVisible(not is_area)

    def set_dirty(self, dirty: bool) -> None:
        self._save_btn.setEnabled(bool(dirty))

    def update_section(self, section: Section | None) -> None:
        """No section-derived controls (annotations are project-global)."""

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _on_combo_changed(self, index: int) -> None:
        if index >= 0:
            self.active_changed.emit(index)

    def _on_eye_toggled(self, checked: bool) -> None:
        self._eye_btn.setIcon(eye_icon(checked))
        self.visibility_changed.emit(checked)

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

    def set_tool(self, tool: str) -> None:
        """Reflect the active tool in the segmented buttons (no signal)."""
        btn = self._tool_btns.get(tool)
        if btn is not None and not btn.isChecked():
            btn.blockSignals(True)
            btn.setChecked(True)
            btn.blockSignals(False)
