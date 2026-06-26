"""Self-contained property sections (one per QGroupBox)."""

from verso.gui.widgets.properties.sections.control_points import ControlPointsBox
from verso.gui.widgets.properties.sections.flip import FlipBox
from verso.gui.widgets.properties.sections.mask import MaskBox
from verso.gui.widgets.properties.sections.overlay import OverlayBox
from verso.gui.widgets.properties.sections.save_bar import SaveBarBox
from verso.gui.widgets.properties.sections.slicing_position import SlicingPositionBox

__all__ = [
    "ControlPointsBox",
    "FlipBox",
    "MaskBox",
    "OverlayBox",
    "SaveBarBox",
    "SlicingPositionBox",
]
