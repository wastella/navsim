"""Regression test for the stage-two-only summary branch in run_pdm_score.main().

Background: pseudo closed-loop score aggregation needs valid stage-one (original)
scenes to pair against stage-two (synthetic) ones. When that pairing is unavailable
(`pseudo_closed_loop_valid == False`, e.g. because only the two-stage synthetic split
was downloaded), main() falls back to saving raw per-scene scores instead of crashing.

If *every single* scenario also failed before scoring, the per-row "frame_type" column
is never added at all (it's only set on the success path in run_pdm_score()), so
`pdm_score_df["frame_type"]` would raise a KeyError while computing stage-two summary
stats. Before the fix, that KeyError was raised *before* the diagnostic
success/failure counts and failed-token list were logged, so the broad
`except Exception` swallowed it and the operator never saw which tokens failed or how
many. The fix moved that diagnostic logging above the code that can raise, and added
an explicit "frame_type" column check with a clear error message.

`run_pdm_score.main()` is a large, `@hydra.main`-decorated function coupled to heavy
IO/multiprocessing dependencies (SceneLoader, MetricCacheLoader, worker_map, ...), so
those are stubbed out here to drive `pdm_score_df` into the exact edge case (all rows
invalid, no "frame_type" column) and exercise the real branch directly -- no logic is
extracted or duplicated.
"""

import logging
from unittest.mock import MagicMock, patch

import pandas as pd
from omegaconf import OmegaConf

from navsim.common.dataclasses import PDMResults
from navsim.planning.script import run_pdm_score as run_pdm_score_module

# @hydra.main wraps the real function for CLI/config-composition purposes; the
# original function is reachable via __wrapped__ (functools.wraps), which lets us call
# it directly with an in-memory DictConfig instead of going through hydra's argv
# parsing and config file resolution.
_raw_main = run_pdm_score_module.main.__wrapped__

FAILED_TOKENS = ["tok1", "tok2"]


def _make_failed_score_row(token: str) -> pd.DataFrame:
    """Mirror exactly what run_pdm_score() builds for a token whose agent/scoring raised.

    Notably, no "frame_type" column is set -- that only happens on the success path.
    """
    row = pd.DataFrame([PDMResults.get_empty_results()])
    row["valid"] = False
    row["token"] = token
    return row


def _build_cfg(output_dir) -> OmegaConf:
    return OmegaConf.create(
        {
            "output_dir": str(output_dir),
            "navsim_log_path": "dummy_log_path",
            "synthetic_scenes_path": "dummy_synthetic_scenes_path",
            "metric_cache_path": "dummy_metric_cache_path",
            "train_test_split": {"scene_filter": {}},  # deliberately no reactive_all_mapping
            "verbose": False,
        }
    )


def _run_main_with_every_scenario_failed(cfg, caplog):
    """
    Drive run_pdm_score.main() end-to-end with every heavy dependency stubbed, so that:
      - every scenario "fails" before a frame_type is ever recorded (score_rows below),
      - pseudo-closed-loop aggregation also fails (cfg has no reactive_all_mapping),
    reproducing the exact edge case the fix addresses.
    """
    score_rows = [_make_failed_score_row(tok) for tok in FAILED_TOKENS]

    scene_loader = MagicMock()
    scene_loader.tokens = list(FAILED_TOKENS)
    scene_loader.get_tokens_list_per_log.return_value = {"log1": list(FAILED_TOKENS)}

    metric_cache_loader = MagicMock()
    metric_cache_loader.tokens = list(FAILED_TOKENS)

    with patch.object(run_pdm_score_module, "build_logger"), patch.object(
        run_pdm_score_module, "build_worker", return_value=MagicMock()
    ), patch.object(
        run_pdm_score_module, "SceneLoader", return_value=scene_loader
    ), patch.object(
        run_pdm_score_module, "MetricCacheLoader", return_value=metric_cache_loader
    ), patch.object(
        run_pdm_score_module, "instantiate", return_value=MagicMock()
    ), patch.object(
        run_pdm_score_module, "worker_map", return_value=score_rows
    ):
        with caplog.at_level(logging.INFO, logger=run_pdm_score_module.logger.name):
            _raw_main(cfg)


def test_all_scenarios_failed_does_not_raise_and_still_saves_csv(tmp_path, caplog):
    """The stage-two-only branch must not crash when every scenario failed."""
    cfg = _build_cfg(tmp_path)

    # Must complete without raising -- pytest fails the test automatically if it does.
    _run_main_with_every_scenario_failed(cfg, caplog)

    csv_files = list(tmp_path.glob("*.csv"))
    assert len(csv_files) == 1, "raw per-scene scores must still be saved even though the summary can't be computed"

    saved_df = pd.read_csv(csv_files[0])
    assert set(saved_df["token"]) == set(FAILED_TOKENS)
    assert (~saved_df["valid"]).all()


def test_failure_diagnostics_are_logged_even_though_frame_type_is_missing(tmp_path, caplog):
    """
    Regression test for the actual fix: the failed-scenario counts and failed-token
    list must be surfaced via logger.info even when "frame_type" is entirely absent
    from pdm_score_df (i.e. literally every scenario failed before scoring).

    On the pre-fix code, this diagnostic logger.info() call lived *after* the line
    that raises KeyError on the missing "frame_type" column, so the broad
    `except Exception` around it swallowed the KeyError before the diagnostics were
    ever logged -- the operator would see only a generic "failed to compute summary
    stats" warning, with no indication of which (or how many) tokens failed. This test
    fails against that code and passes against the fixed code, where the diagnostics
    are logged unconditionally before the code that can raise.
    """
    cfg = _build_cfg(tmp_path)

    _run_main_with_every_scenario_failed(cfg, caplog)

    info_text = "\n".join(record.message for record in caplog.records if record.levelno == logging.INFO)

    assert "Number of failed scenarios: 2" in info_text, (
        "failed-scenario count was not logged -- diagnostics are being swallowed by the "
        "exception raised when 'frame_type' is missing"
    )
    for token in FAILED_TOKENS:
        assert token in info_text, f"failed token {token!r} was not logged -- diagnostics are being swallowed"

    # The pre-existing generic warning for the (still expected) inner failure should
    # still fire -- we're not asserting the KeyError disappears, only that the
    # diagnostics logged *before* it are no longer lost.
    warning_text = "\n".join(record.message for record in caplog.records if record.levelno == logging.WARNING)
    assert "summary stats" in warning_text.lower()
