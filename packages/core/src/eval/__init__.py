"""Objective evaluation harness for stem-separation quality.

Reports SDR, fullness, and bleedless per stem against a reference corpus, with
deterministic inference settings recorded alongside every score.  See
``docs/evaluation_harness.md`` for the metric definitions and corpus format,
and ``~/Projects/upmixer-knowledge/techniques/evaluation.md`` for the
community context these metrics are drawn from.

This package gates separation-quality decisions elsewhere in the codebase
(model swaps, ensembling, phase-fix/debleed passes): those changes should be
measured here before shipping, per AGENTS.md's Knowledge Base section.
"""

from upmixer.eval.metrics import bleedless, fullness, sdr
from upmixer.eval.corpus import CorpusItem, ReferenceCorpus, synthetic_corpus
from upmixer.eval.harness import (
    EvaluationSkipped,
    evaluate_corpus,
    separate_for_eval,
    separate_tree_for_eval,
)
from upmixer.eval.origins import (
    OriginEvaluationResult,
    OriginViewOutput,
    separate_with_extra_origin,
)
from upmixer.eval.cascade import (
    CascadeArmOutput,
    CascadeEvaluationResult,
    separate_with_deux_cascade,
)
from upmixer.eval.types import ItemRunSettings, RunSettings
from upmixer.eval.report import CoverageRow, EvalReport, StemScore, format_report

__all__ = [
    "sdr",
    "fullness",
    "bleedless",
    "CorpusItem",
    "ReferenceCorpus",
    "synthetic_corpus",
    "RunSettings",
    "EvaluationSkipped",
    "ItemRunSettings",
    "OriginEvaluationResult",
    "OriginViewOutput",
    "separate_for_eval",
    "separate_with_extra_origin",
    "CascadeArmOutput",
    "CascadeEvaluationResult",
    "separate_with_deux_cascade",
    "separate_tree_for_eval",
    "evaluate_corpus",
    "EvalReport",
    "StemScore",
    "CoverageRow",
    "format_report",
]
