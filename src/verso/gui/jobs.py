"""Background workers and a small runner for off-the-UI-thread jobs.

The workers here are plain ``QObject``s moved onto a ``QThread`` by
:class:`BackgroundJob`, which also owns the indeterminate progress dialog and
the standard signal wiring so call sites in the main window stay short.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from PyQt6.QtCore import QObject, QThread, pyqtBoundSignal, pyqtSignal
from PyQt6.QtWidgets import QProgressDialog, QWidget

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from verso.engine.elastix import ElastixWorker
    from verso.engine.model.alignment import ControlPoint
    from verso.engine.model.elastix import ElastixParams
    from verso.engine.model.project import Project


class DeepSliceWorker(QObject):
    done = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        project: Project,
        reverse_section_order: bool = False,
        bad_section_ids: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._project = project
        self._reverse_section_order = reverse_section_order
        self._bad_section_ids = bad_section_ids or []

    def run(self) -> None:
        try:
            from verso.engine.deepslice import DeepSliceOptions, run_deepslice_suggestions

            result = run_deepslice_suggestions(
                self._project,
                DeepSliceOptions(
                    species="mouse",
                    reverse_section_order=self._reverse_section_order,
                    bad_section_ids=self._bad_section_ids,
                ),
            )
            self.done.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class AnnotationLoadWorker(QObject):
    """Parse a project's annotations off the UI thread.

    ``load_annotations`` builds one object per point, so a series with hundreds
    of thousands of points takes a couple of seconds — long enough to freeze the
    window if run on project open. This worker does it in a thread and hands the
    parsed list back; see the annotation controller's ``load_for_project_async``.

    ``done`` carries the load generation the controller stamped this worker with,
    so a result that arrives after a newer (re)load started can be ignored.
    """

    done = pyqtSignal(object, int)  # (list[Annotation], generation)

    def __init__(self, project_dir: Path, generation: int) -> None:
        super().__init__()
        self._project_dir = project_dir
        self._generation = generation

    def run(self) -> None:
        from verso.engine.io.annotation_io import load_annotations

        try:
            annotations = load_annotations(self._project_dir)
        except Exception:
            annotations = []
        self.done.emit(annotations, self._generation)


class BatchMaskWorker(QObject):
    done = pyqtSignal(int, list)

    def __init__(self, sections: list, working_scale: float) -> None:
        super().__init__()
        self._sections = sections
        self._working_scale = working_scale
        # Detected masks held in RAM (section.id -> bool array), drained by the
        # main thread into the resident prep-draft store on completion.  Nothing
        # is written to disk until the user saves.
        self.results: dict[str, np.ndarray] = {}

    def run(self) -> None:
        errors: list[str] = []
        completed = 0

        from verso.engine.io.image_io import ensure_working_copy
        from verso.engine.preprocessing import detect_foreground

        for section in self._sections:
            try:
                image = ensure_working_copy(section, self._working_scale)
                if image is None:
                    errors.append(f"{Path(section.original_path).name}: no readable image")
                    continue
                self.results[section.id] = detect_foreground(image)
                completed += 1
            except Exception as exc:
                errors.append(f"{Path(section.original_path).name}: {exc}")
        self.done.emit(completed, errors)


class AutoCPWorker(QObject):
    """Drive elastix control-point generation off the UI thread.

    Loads images and slices the atlas template here (safe in a thread), then
    hands the prepared arrays to a persistent warm child process
    (:class:`ElastixWorker`) that runs the native registration. Passing arrays
    means the child never reloads the atlas, and keeping it warm means only the
    first run pays the optimizer's cold-start cost.
    """

    done = pyqtSignal(int, list)

    def __init__(
        self,
        worker: ElastixWorker,
        sections: list,
        atlas,
        working_scale: float,
        params: ElastixParams,
    ) -> None:
        super().__init__()
        self._worker = worker
        self._sections = sections
        self._atlas = atlas
        self._working_scale = working_scale
        self._params = params
        # section.id -> list[ControlPoint], drained by the main thread on completion.
        self.results: dict[str, list[ControlPoint]] = {}

    def run(self) -> None:
        from verso.engine.elastix import prepare_registration_inputs

        try:
            inputs, errors = prepare_registration_inputs(
                self._sections, self._atlas, self._working_scale
            )
            if inputs:
                results, gen_errors = self._worker.generate(inputs, self._atlas.shape, self._params)
                self.results = results
                errors = errors + gen_errors
            self.done.emit(len(self.results), errors)
        except Exception as exc:
            self.done.emit(0, [str(exc)])


class QuantifyWorker(QObject):
    """Runs a quantification call off the UI thread.

    Takes a zero-argument ``run_fn`` (a closure over the chosen
    ``quantify_*`` function and its arguments). Emits the result on ``done`` or the
    error message on ``error`` (precondition failures raise ``QuantificationError``,
    whose message is user-facing).
    """

    done = pyqtSignal(object)  # result dict
    error = pyqtSignal(str)

    def __init__(self, run_fn: Callable[[], object]) -> None:
        super().__init__()
        self._run_fn = run_fn

    def run(self) -> None:
        try:
            self.done.emit(self._run_fn())
        except Exception as exc:  # QuantificationError + any I/O failure
            self.error.emit(str(exc))


class JobWorker(Protocol):
    """Structural type for workers driven by :class:`BackgroundJob`.

    Satisfied by any ``QObject`` exposing a ``done`` signal and a ``run`` slot
    (the three workers above). An optional ``error`` signal is discovered at
    runtime via ``getattr``, so it is not part of the protocol.

    ``done`` is declared as a read-only property returning ``pyqtSignal`` —
    the *declared* type of each concrete worker's ``done = pyqtSignal(...)``
    attribute. A read-only property is checked covariantly, so ``pyqtSignal``
    matches ``pyqtSignal`` and the workers satisfy the protocol structurally.
    (A plain attribute would demand an *invariant* match and, more importantly,
    the type checker does not apply the ``pyqtSignal`` descriptor's ``__get__``
    when matching protocol members, so it compares the raw ``pyqtSignal`` type
    rather than the ``pyqtBoundSignal`` an instance access resolves to.) The
    bound signal — the thing with ``.connect`` — is recovered with a ``cast``
    at the single connect site in :meth:`BackgroundJob.start`.
    """

    @property
    def done(self) -> pyqtSignal: ...

    def run(self) -> None: ...
    def moveToThread(self, thread: QThread) -> None: ...
    def deleteLater(self) -> None: ...


class BackgroundJob[W: JobWorker]:
    """Runs a worker ``QObject`` on a ``QThread`` behind a progress dialog.

    Wires the standard lifecycle: ``started`` → ``worker.run``, the worker's
    ``done`` signal → quit, and ``thread.finished`` → ``deleteLater`` plus
    dialog teardown. The worker must expose a ``done`` signal; an optional
    ``error`` signal is routed to quit (and to ``on_error`` if given) too.

    Pass ``silent=True`` for a job with no progress dialog — used when the work
    should run without the user noticing (e.g. loading annotations on project
    open); ``title``/``message``/``modal``/``min_width`` are then ignored.
    """

    def __init__(
        self,
        parent: QWidget,
        worker: W,
        *,
        title: str = "",
        message: str = "",
        modal: bool = False,
        min_width: int | None = None,
        silent: bool = False,
    ) -> None:
        self.worker: W = worker
        self._thread = QThread(parent)
        worker.moveToThread(self._thread)

        self._progress: QProgressDialog | None = None
        if not silent:
            progress = QProgressDialog(message, "", 0, 0, parent)
            progress.setWindowTitle(title)
            progress.setCancelButton(None)
            progress.setMinimumDuration(0)
            progress.setAutoClose(False)
            progress.setAutoReset(False)
            progress.setModal(modal)
            if min_width is not None:
                progress.setMinimumWidth(min_width)
            self._progress = progress

    def start(
        self,
        on_done: Callable,
        on_finished: Callable,
        on_error: Callable | None = None,
    ) -> None:
        """Show the dialog and start the worker.

        ``on_done`` receives the worker's ``done`` arguments; ``on_finished`` is
        called (no args) once the thread has fully stopped; ``on_error``, when
        given, receives the worker's ``error`` argument.
        """
        thread = self._thread
        worker = self.worker
        thread.started.connect(worker.run)
        # ``worker.done`` is declared ``pyqtSignal`` on the protocol; on the
        # concrete QObject worker it is the bound signal that carries ``.connect``.
        done = cast(pyqtBoundSignal, worker.done)
        done.connect(on_done)
        done.connect(thread.quit)
        error = getattr(worker, "error", None)
        if error is not None:
            if on_error is not None:
                error.connect(on_error)
            error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._teardown)
        thread.finished.connect(on_finished)
        if self._progress is not None:
            self._progress.show()
        thread.start()

    def is_running(self) -> bool:
        return self._thread.isRunning()

    def stop(self) -> None:
        """Quit the worker thread and block until it has finished."""
        self._thread.quit()
        self._thread.wait()

    def _teardown(self) -> None:
        if self._progress is not None:
            self._progress.close()
            self._progress.deleteLater()
