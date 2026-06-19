"""Position-along-slicing-axis strip chart (Align view)."""

from __future__ import annotations

import pyqtgraph as pg
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QGroupBox, QSizePolicy, QVBoxLayout


class SlicingPositionBox(QGroupBox):
    def __init__(self) -> None:
        super().__init__("Slicing position")
        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        # Blend the plot into the group box (and every other section), which
        # paints with the app palette's Window color. pyqtgraph ignores the Qt
        # palette, so we feed it that color explicitly rather than hardcoding.
        bg = self.palette().color(QPalette.ColorRole.Window)
        self._plot = pg.PlotWidget(background=bg)
        self._plot.setFixedHeight(200)
        self._plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        pi = self._plot.getPlotItem()
        pi.hideAxis("top")
        pi.hideAxis("right")
        pi.getAxis("bottom").setLabel("Section", color="#aaa")
        # Neutral until set_axis_name() supplies the project's slicing axis.
        pi.getAxis("left").setLabel("Position (mm)", color="#aaa")
        pi.getAxis("bottom").setTextPen(pg.mkPen("#aaa"))
        pi.getAxis("left").setTextPen(pg.mkPen("#aaa"))
        pi.setMenuEnabled(False)
        layout.addWidget(self._plot)

    def set_axis_name(self, name: str) -> None:
        """Update the y-axis label to match the project's slicing axis (AP / ML / DV).

        The group box title stays axis-agnostic ("Slicing position"); only the
        y-axis label reflects the actual interpolation axis and units.
        """
        self._plot.getPlotItem().getAxis("left").setLabel(f"{name} (mm)", color="#aaa")

    def update_plot(self, sections: list, current_index: int) -> None:
        """Redraw the position strip chart."""
        from verso.engine.model.alignment import AlignmentStatus

        pi = self._plot.getPlotItem()
        pi.clear()

        if not sections:
            return

        x_complete, y_complete = [], []
        x_progress, y_progress = [], []
        x_none, y_none = [], []

        for i, section in enumerate(sections):
            pos = section.alignment.position_mm
            if pos is None or all(v == 0.0 for v in (section.alignment.anchoring or [])):
                continue
            s = section.alignment.status
            if s == AlignmentStatus.COMPLETE:
                x_complete.append(i)
                y_complete.append(pos)
            elif s == AlignmentStatus.IN_PROGRESS:
                x_progress.append(i)
                y_progress.append(pos)
            else:
                x_none.append(i)
                y_none.append(pos)

        def _add_scatter(xs, ys, color, size=6) -> None:
            if not xs:
                return
            pi.addItem(
                pg.ScatterPlotItem(
                    x=xs,
                    y=ys,
                    symbol="o",
                    size=size,
                    brush=pg.mkBrush(*color),
                    pen=pg.mkPen(None),
                )
            )

        _add_scatter(x_none, y_none, (130, 130, 130, 180))
        _add_scatter(x_progress, y_progress, (255, 193, 7, 220))
        _add_scatter(x_complete, y_complete, (76, 175, 80, 220))

        if 0 <= current_index < len(sections):
            section = sections[current_index]
            pos = section.alignment.position_mm
            if pos is not None and any(v != 0.0 for v in (section.alignment.anchoring or [])):
                pi.addItem(
                    pg.ScatterPlotItem(
                        x=[current_index],
                        y=[pos],
                        symbol="o",
                        size=11,
                        brush=pg.mkBrush(255, 255, 255, 230),
                        pen=pg.mkPen(None),
                    )
                )
