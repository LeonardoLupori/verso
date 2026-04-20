"""Drawing tools for masks and warp control points on the canvas."""

from __future__ import annotations

from enum import Enum, auto

import numpy as np
from PyQt6.QtCore import QPointF, pyqtSignal
from PyQt6.QtWidgets import QGraphicsScene


class ToolMode(Enum):
    NONE = auto()
    DRAW_MASK = auto()
    ERASE_MASK = auto()
    ADD_CONTROL_POINT = auto()
    MOVE_CONTROL_POINT = auto()
    DELETE_CONTROL_POINT = auto()


class ROIToolController:
    """Translates raw scene mouse events into mask / control-point edits.

    Attach to a :class:`~verso.gui.widgets.canvas.SectionCanvas` and connect
    :attr:`mask_changed` / :attr:`control_points_changed` to update the model.
    """

    mask_changed = pyqtSignal(np.ndarray)
    control_points_changed = pyqtSignal(list)

    def __init__(self, scene: QGraphicsScene) -> None:
        self._scene = scene
        self._mode = ToolMode.NONE
        self._brush_radius: int = 20

    @property
    def mode(self) -> ToolMode:
        return self._mode

    def set_mode(self, mode: ToolMode) -> None:
        self._mode = mode

    @property
    def brush_radius(self) -> int:
        return self._brush_radius

    def set_brush_radius(self, radius: int) -> None:
        self._brush_radius = max(1, radius)

    def handle_press(self, pos: QPointF) -> None:
        raise NotImplementedError

    def handle_move(self, pos: QPointF) -> None:
        raise NotImplementedError

    def handle_release(self, pos: QPointF) -> None:
        raise NotImplementedError
