"""Opt-in runtime hook that makes ReLA's official evaluator save predictions.

Python imports ``sitecustomize`` automatically during interpreter startup when
this directory is prepended to ``PYTHONPATH``.  The hook is deliberately inert
unless ``FREEREF_RELA_SAVE_PREDICTIONS=1`` is set by our ReLA runner.
"""

from __future__ import annotations

import os


if os.environ.get("FREEREF_RELA_SAVE_PREDICTIONS") == "1":
    import itertools

    import gres_model.evaluation.refer_evaluation as refer_evaluation
    from gres_model.evaluation.refer_evaluation import ReferEvaluator

    # The pinned official evaluator calls itertools.chain in distributed mode
    # but omits the module import.  Injecting it here preserves the evaluator's
    # one-rank distributed path without editing the third-party checkout.
    refer_evaluation.itertools = itertools

    if not getattr(ReferEvaluator, "_freeref_save_hook_installed", False):
        _official_init = ReferEvaluator.__init__

        def _freeref_init(
            self,
            dataset_name,
            distributed=True,
            output_dir=None,
            save_imgs=False,
        ):
            del save_imgs
            _official_init(
                self,
                dataset_name,
                distributed=distributed,
                output_dir=output_dir,
                save_imgs=True,
            )

        ReferEvaluator.__init__ = _freeref_init
        ReferEvaluator._freeref_save_hook_installed = True
        print("[FreeRef] ReLA prediction-saving hook installed.", flush=True)
