from __future__ import annotations

from typing import Any, Dict, Optional

from stage1_quality_screening import Stage1QualityConfig, run_quality_screening


def process_image_pipeline(image: Any, config: Optional[Stage1QualityConfig] = None) -> Dict[str, Any]:
    stage1 = run_quality_screening(image, config)
    if not stage1["passed"]:
        return {"stage1": stage1, "stage2": None}

    # Placeholder for Stage 2 execution.
    return {"stage1": stage1, "stage2": {"status": "ready_for_stage2"}}
