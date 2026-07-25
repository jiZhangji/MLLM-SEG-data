"""Opt-in hook for exporting masks from GSVA's official validation loop."""

from __future__ import annotations

import os
from pathlib import Path


if os.environ.get("FREEREF_GSVA_EXPORT") == "1":
    import solver

    from universal_freeref.gsva_export import GSVAOfficialExporter

    if not getattr(solver, "_freeref_gsva_hook_installed", False):
        _official_validate = solver.validate

        class _ExportingEngine:
            def __init__(self, engine, exporter):
                self._engine = engine
                self._exporter = exporter

            def __getattr__(self, name):
                return getattr(self._engine, name)

            def eval(self):
                self._engine.eval()
                return self

            def __call__(self, **input_dict):
                output = self._engine(**input_dict)
                self._exporter.record(input_dict, output)
                return output

        def _validate_and_export(val_loader, model_engine, epoch, args, logger):
            exporter = GSVAOfficialExporter(
                Path(os.environ["FREEREF_GSVA_EXPORT_DIR"]),
                os.environ.get("FREEREF_GSVA_METHOD", "GSVA-7B-official"),
                os.environ.get("FREEREF_GSVA_SPLIT", str(args.val_dataset).replace("|", "_")),
            )
            metrics = _official_validate(
                val_loader,
                _ExportingEngine(model_engine, exporter),
                epoch,
                args,
                logger,
            )
            report = exporter.finalize(metrics)
            logger.info(
                "FreeRef GSVA export: %d masks -> %s",
                report["samples"],
                report["manifest"],
            )
            return metrics

        solver.validate = _validate_and_export
        solver._freeref_gsva_hook_installed = True
        print("[FreeRef] GSVA final-mask export hook installed.", flush=True)
