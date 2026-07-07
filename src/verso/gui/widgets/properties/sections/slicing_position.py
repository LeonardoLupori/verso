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
        pi.getAxis("bottom").setLabel("Slice index", color="#aaa")
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

    # Translucency for the dots so they blend into the plot background; the hues
    # themselves come from status.STATUS_COLOR (the filmstrip's traffic lights).
    _DOT_ALPHA = 220

    def update_plot(self, sections: list, current_index: int, dirty_ids: set[str]) -> None:
        """Redraw the position strip chart.

        Dots are coloured by each section's *align* step status via
        :func:`section_step_status`, so an unsaved (dirty) section shows yellow
        exactly as its Align filmstrip dot does.  ``dirty_ids`` is the set of
        section ids with unsaved align edits (from the GUI edit registry).
        """
        from verso.engine.model.status import STATUS_COLOR, section_step_status

        pi = self._plot.getPlotItem()
        pi.clear()

        if not sections:
            return

        # Group dots by align status. STATUS_COLOR is ordered gray → yellow →
        # green, so drawing in its order keeps the "further along" dots on top.
        buckets = {status: ([], []) for status in STATUS_COLOR}
        for section in sections:
            pos = section.alignment.position_mm
            if pos is None or all(v == 0.0 for v in (section.alignment.anchoring or [])):
                continue
            # X is the physical slice index so spacing matches how interpolation
            # is parameterized (uneven gaps stay uneven), not the list rank.
            status = section_step_status(section, "align", dirty=section.id in dirty_ids)
            xs, ys = buckets[status]
            xs.append(section.slice_index)
            ys.append(pos)

        for status, (xs, ys) in buckets.items():
            if not xs:
                continue
            color = pg.mkColor(STATUS_COLOR[status])
            color.setAlpha(self._DOT_ALPHA)
            pi.addItem(
                pg.ScatterPlotItem(
                    x=xs,
                    y=ys,
                    symbol="o",
                    size=6,
                    brush=pg.mkBrush(color),
                    pen=pg.mkPen(None),
                )
            )

        # Highlight the current slice: keep its status colour but ring it in white
        # and draw it larger, so the selection is obvious without hiding the state.
        if 0 <= current_index < len(sections):
            section = sections[current_index]
            pos = section.alignment.position_mm
            if pos is not None and any(v != 0.0 for v in (section.alignment.anchoring or [])):
                status = section_step_status(section, "align", dirty=section.id in dirty_ids)
                color = pg.mkColor(STATUS_COLOR[status])
                color.setAlpha(self._DOT_ALPHA)
                pi.addItem(
                    pg.ScatterPlotItem(
                        x=[section.slice_index],
                        y=[pos],
                        symbol="o",
                        size=10,
                        brush=pg.mkBrush(color),
                        pen=pg.mkPen("w", width=2),
                    )
                )
