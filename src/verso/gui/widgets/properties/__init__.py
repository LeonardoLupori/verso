"""Properties panel package.

Public API: ``PropertiesPanel``.  Internally split into one widget per
``QGroupBox`` in ``sections/`` and one page widget per view mode.
"""

from verso.gui.widgets.properties.panel import PropertiesPanel

__all__ = ["PropertiesPanel"]
