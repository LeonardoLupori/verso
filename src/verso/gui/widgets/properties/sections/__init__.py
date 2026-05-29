"""Self-contained property sections (one per QGroupBox)."""

from verso.gui.widgets.properties.sections.ap_plot import APPlotBox
from verso.gui.widgets.properties.sections.control_points import ControlPointsBox
from verso.gui.widgets.properties.sections.flip import FlipBox
from verso.gui.widgets.properties.sections.hemisphere import HemisphereBox
from verso.gui.widgets.properties.sections.mask import MaskBox
from verso.gui.widgets.properties.sections.overlay import OverlayBox
from verso.gui.widgets.properties.sections.proposal import ProposalBox

__all__ = [
    "APPlotBox",
    "ControlPointsBox",
    "FlipBox",
    "HemisphereBox",
    "MaskBox",
    "OverlayBox",
    "ProposalBox",
]
