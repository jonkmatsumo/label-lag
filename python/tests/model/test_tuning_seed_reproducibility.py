"""Regression tests for Optuna seed persistence and resume reproducibility."""

from __future__ import annotations

from unittest.mock import patch

import optuna

from model.tuning import run_tuning_study

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_xy():
    """Tiny balanced dataset — enough for a trial to complete."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 40
    x_data = pd.DataFrame({"a": rng.random(n), "b": rng.random(n)})
    y = pd.Series([0] * (n // 2) + [1] * (n // 2))
    return x_data, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOptunaSeeds:
    def test_seed_persisted_in_user_attrs_on_first_run(self):
        """After the first run, study.user_attrs['seed'] must equal the seed arg."""
        x_data, y = _make_xy()
        captured: list[optuna.study.Study] = []

        # Ensure we use in-memory study by forcing storage_url=None
        from model.tuning import create_tuning_study

        def capturing_create(**kwargs):
            # Force storage_url to None for unit test
            kwargs["storage_url"] = None
            study = create_tuning_study(**kwargs)
            captured.append(study)
            return study

        with (
            patch("model.tuning.create_tuning_study", side_effect=capturing_create),
            patch("training.optuna_resume.get_optuna_storage_url", return_value=None),
            patch("mlflow.log_dict"),
            patch("mlflow.log_params"),
            patch("mlflow.active_run"),
            patch("mlflow.start_run"),
        ):
            run_tuning_study(x_data, y, x_data, y, n_trials=1, seed=99)

        assert captured, "create_tuning_study was never called"
        assert captured[0].user_attrs.get("seed") == 99

    def test_seed_not_overwritten_on_second_call(self):
        """If user_attrs already has 'seed', a second call must not overwrite it."""
        x_data, y = _make_xy()
        # In-memory study for testing
        study = optuna.create_study(direction="maximize")
        study.set_user_attr("seed", 7)

        call_count = 0

        def fake_create(**kwargs):
            nonlocal call_count
            call_count += 1
            return study

        with (
            patch("model.tuning.create_tuning_study", side_effect=fake_create),
            patch("training.optuna_resume.get_optuna_storage_url", return_value=None),
            patch("mlflow.log_dict"),
            patch("mlflow.log_params"),
            patch("mlflow.active_run"),
            patch("mlflow.start_run"),
        ):
            run_tuning_study(x_data, y, x_data, y, n_trials=1, seed=42)

        assert study.user_attrs["seed"] == 7, "Original seed must not be overwritten"

    def test_persisted_seed_overrides_caller_seed_on_resume(self):
        """When storage has a study with seed=7, passing seed=99 must use 7."""
        x_data, y = _make_xy()

        # Simulate an existing study in storage with seed=7
        # Use in-memory for unit test
        existing_study = optuna.create_study(direction="maximize")
        existing_study.set_user_attr("seed", 7)

        sampler_seeds: list[int] = []

        original_tpe = optuna.samplers.TPESampler

        def capturing_tpe(seed=None, **kwargs):
            sampler_seeds.append(seed)
            return original_tpe(seed=seed, **kwargs)

        with (
            patch("model.tuning.create_tuning_study", return_value=existing_study),
            patch("optuna.load_study", return_value=existing_study),
            patch("optuna.samplers.TPESampler", side_effect=capturing_tpe),
            patch("training.optuna_resume.get_optuna_storage_url", return_value=None),
            patch("mlflow.log_dict"),
            patch("mlflow.log_params"),
            patch("mlflow.active_run"),
            patch("mlflow.start_run"),
        ):
            run_tuning_study(
                x_data,
                y,
                x_data,
                y,
                n_trials=1,
                seed=99,
                job_id="job-abc",
                storage_url="sqlite:///fake.db",
            )

        assert sampler_seeds, "TPESampler was never instantiated"
        assert sampler_seeds[0] == 7, (
            f"Expected seed=7 (from persisted attrs), got seed={sampler_seeds[0]}"
        )
