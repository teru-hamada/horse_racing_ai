from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.public_api import (
    FeatureGenerator,
    FeaturePipeline,
    FeatureRegistry,
    FeatureSetDefinition,
)


class SampleGenerator(FeatureGenerator):
    name = "sample"
    version = "1"
    output_columns = ("sample_score",)

    def transform(self, targets, history, context):
        output = targets[["race_id", "horse_id"]].copy()
        output["sample_score"] = float(context.parameters["score"])
        return output


def test_pipeline_uses_registered_versioned_generator():
    targets = pd.DataFrame({"race_id": ["r1"], "horse_id": ["h1"]})
    definition = FeatureSetDefinition(
        name="baseline",
        version="1.0.0",
        generators=("sample",),
        parameters={"score": 3.5},
    )
    pipeline = FeaturePipeline(FeatureRegistry([SampleGenerator()]))

    result = pipeline.transform(targets, pd.DataFrame(), definition)

    assert result.to_dict("records") == [
        {"race_id": "r1", "horse_id": "h1", "sample_score": 3.5}
    ]


def test_registry_rejects_duplicate_feature_columns():
    registry = FeatureRegistry([SampleGenerator()])
    try:
        registry.register(SampleGenerator())
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("Duplicate generator was accepted")


def test_pipeline_rejects_changed_target_order():
    class ReorderedGenerator(SampleGenerator):
        name = "reordered"

        def transform(self, targets, history, context):
            output = super().transform(targets, history, context)
            return output.iloc[::-1].reset_index(drop=True)

    targets = pd.DataFrame(
        {"race_id": ["r1", "r2"], "horse_id": ["h1", "h2"]}
    )
    pipeline = FeaturePipeline(FeatureRegistry([ReorderedGenerator()]))
    definition = FeatureSetDefinition("baseline", "1", ("reordered",), {"score": 1})

    try:
        pipeline.transform(targets, pd.DataFrame(), definition)
    except ValueError as exc:
        assert "preserve target row order" in str(exc)
    else:
        raise AssertionError("Reordered output was accepted")
