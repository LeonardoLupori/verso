"""AP position strip chart (Align view)."""

from __future__ import annotations

import pyqtgraph as pg
from PyQt6.QtWidgets import QGroupBox, QSizePolicy, QVBoxLayout


class APPlotBox(QGroupBox):
    def __init__(self) -> None:
        super().__init__("AP position")
        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        self._plot = pg.PlotWidget(background="#1a1a1a")
        self._plot.setFixedHeight(200)
        self._plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        pi = self._plot.getPlotItem()
        pi.hideAxis("top")
        pi.hideAxis("right")
        pi.getAxis("bottom").setLabel("Section", color="#aaa")
        pi.getAxis("left").setLabel("AP (mm)", color="#aaa")
        pi.getAxis("bottom").setTextPen(pg.mkPen("#aaa"))
        pi.getAxis("left").setTextPen(pg.mkPen("#aaa"))
        pi.setMenuEnabled(False)
        layout.addWidget(self._plot)

    def update_plot(self, sections: list, current_index: int) -> None:
        """Redraw the AP position strip chart."""
        from verso.engine.model.alignment import AlignmentStatus

        pi = self._plot.getPlotItem()
        pi.clear()

        if not sections:
            return

        x_complete, y_complete = [], []
        x_progress, y_progress = [], []
        x_none, y_none = [], []

        for i, section in enumerate(sections):
            ap = section.alignment.ap_position_mm
            if ap is None or all(v == 0.0 for v in (section.alignment.anchoring or [])):
                continue
            s = section.alignment.status
            if s == AlignmentStatus.COMPLETE:
                x_complete.append(i)
                y_complete.append(ap)
            elif s == AlignmentStatus.IN_PROGRESS:
                x_progress.append(i)
                y_progress.append(ap)
            else:
                x_none.append(i)
                y_none.append(ap)

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
            ap = section.alignment.ap_position_mm
            if ap is not None and any(v != 0.0 for v in (section.alignment.anchoring or [])):
                pi.addItem(
                    pg.ScatterPlotItem(
                        x=[current_index],
                        y=[ap],
                        symbol="o",
                        size=11,
                        brush=pg.mkBrush(255, 255, 255, 230),
                        pen=pg.mkPen(None),
                    )
                )
