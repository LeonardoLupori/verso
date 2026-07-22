"""Persistent subprocess that runs elastix registrations off the host process.

Run as ``python -m verso.engine._elastix_worker --serve``. The elastix C++
optimizer segfaults when embedded in a host's worker thread (e.g. a Qt
``QThread``), so the GUI drives it through this child process instead
(see :class:`verso.engine.elastix.ElastixWorker`).

Lifecycle:
  1. On startup the child runs a tiny dummy registration to pay the one-time
     native template-instantiation cost (~15 s) up front, then prints a READY
     line. Subsequent real jobs reuse the warmed optimizer (~0.5 s each).
  2. It then reads job-directory paths from stdin, one per line. For each it
     loads the prepared arrays, registers every section, writes ``result.json``
     incrementally (so a mid-batch native crash preserves earlier results), and
     prints a DONE line.
  3. A ``QUIT`` line (or EOF) ends the loop.

Job directory contents (written by the parent):
  - ``job.json``: {"atlas_shape", "params", "sections": [{"id", "index",
    "anchoring", "manual_cps", "has_mask"}, ...]}
  - ``section_<i>.npy`` / ``template_<i>.npy`` / ``mask_<i>.npy`` (mask optional)

``result.json`` (written by the child):
  {"results": {section_id: [ControlPoint dict, ...]}, "errors": [str, ...]}
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
from pathlib import Path

from verso.engine.elastix import _WORKER_DONE, _WORKER_QUIT, _WORKER_READY

_log = logging.getLogger(__name__)


def _prewarm() -> None:
    """Run a tiny registration so the native optimizer instantiates up front."""
    try:
        import numpy as np

        from verso.engine.elastix import auto_control_points
        from verso.engine.model.elastix import ElastixParams

        h, w = 64, 64
        yy, xx = np.mgrid[0:h, 0:w]
        img = (np.exp(-(((yy - 32) / 15) ** 2 + ((xx - 32) / 15) ** 2)) * 255).astype(np.float32)
        anchoring = [0.0, 264.0, 0.0, 456.0, 0.0, 0.0, 0.0, 0.0, 320.0]
        auto_control_points(
            img,
            img,
            anchoring,
            (528, 320, 456),
            params=ElastixParams(n_resolutions=1, max_iterations=5, n_samples=128),
        )
    except Exception:
        # Warming is best-effort; real jobs will pay the cost if it failed.
        _log.warning(
            "Elastix prewarm failed; first real job will pay the cold-start cost", exc_info=True
        )


def _process_job(job_dir: Path) -> None:
    import numpy as np

    from verso.engine.elastix import auto_control_points
    from verso.engine.model.alignment import ControlPoint
    from verso.engine.model.elastix import ElastixParams

    job = json.loads((job_dir / "job.json").read_text())
    atlas_shape = tuple(job["atlas_shape"])
    params = ElastixParams.from_dict(job["params"])

    out: dict = {"results": {}, "errors": []}

    def flush() -> None:
        (job_dir / "result.json").write_text(json.dumps(out))

    flush()
    for sec in job["sections"]:
        i = sec["index"]
        name = sec["id"]
        try:
            section = np.load(job_dir / f"section_{i}.npy")
            template = np.load(job_dir / f"template_{i}.npy")
            mask = np.load(job_dir / f"mask_{i}.npy") if sec["has_mask"] else None
            manual = [ControlPoint.from_dict(d) for d in sec["manual_cps"]]
            cps = auto_control_points(
                section,
                template,
                sec["anchoring"],
                atlas_shape,
                mask=mask,
                manual_cps=manual,
                params=params,
            )
            out["results"][name] = [cp.to_dict() for cp in cps]
        except Exception as exc:
            _log.exception("Registration failed for section %s", name)
            out["errors"].append(f"{name}: {exc}")
        flush()  # incremental: a native crash on a later section keeps earlier results


def serve() -> None:
    # File-only logging: stdout is the READY/DONE IPC channel to the parent and
    # must carry nothing else, so never attach a stdout handler here.
    from verso.engine.logconf import configure_logging

    configure_logging(process_tag="elastix", console=False)
    _log.info("Elastix worker starting")

    _prewarm()
    sys.stdout.write(_WORKER_READY + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        cmd = line.strip()
        if not cmd or cmd == _WORKER_QUIT:
            break
        try:
            _process_job(Path(cmd))
        except Exception as exc:
            _log.exception("Elastix job failed: %s", cmd)
            # Ensure the parent always finds a result file so it isn't stuck.
            with contextlib.suppress(Exception):
                (Path(cmd) / "result.json").write_text(
                    json.dumps({"results": {}, "errors": [f"job failed: {exc}"]})
                )
        sys.stdout.write(_WORKER_DONE + "\n")
        sys.stdout.flush()


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "--serve":
        serve()
        return
    raise SystemExit("usage: python -m verso.engine._elastix_worker --serve")


if __name__ == "__main__":
    main()
