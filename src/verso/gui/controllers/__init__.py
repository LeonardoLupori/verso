"""Controllers that own self-contained MainWindow subsystems.

Each controller is a plain (non-``QObject``) class constructed by
:class:`~verso.gui.main_window.MainWindow` after its widgets exist. Controllers
receive the window as ``window`` and reach shared widgets and the "refresh
dependent UI" cluster through a small set of coordinator methods on the window,
rather than poking view internals. This keeps the signal/slot mediator pattern
intact while splitting the workflow logic out of the god object.
"""

from __future__ import annotations

from verso.gui.controllers.export_controller import ExportController
from verso.gui.controllers.job_controller import JobController
from verso.gui.controllers.project_controller import ProjectController
from verso.gui.controllers.save_controller import SaveController

__all__ = ["ExportController", "JobController", "ProjectController", "SaveController"]
