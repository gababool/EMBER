from pathlib import Path

from pipeline.llm.inference import LLMInferenceCancelled
from pipeline.service import LLMReasoningStage, PipelineContext


def test_llm_stage_returns_cancelled_result(monkeypatch, tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("system prompt", encoding="utf-8")
    image_file = tmp_path / "image.jpg"
    image_file.write_bytes(b"fake image")

    def cancelled_inference(**_kwargs):
        raise LLMInferenceCancelled("cancelled")

    monkeypatch.setattr(
        "pipeline.llm.inference.run_llm_inference",
        cancelled_inference,
    )

    stage = LLMReasoningStage(prompt_file=prompt_file)
    result = stage.run(PipelineContext(image_path=Path(image_file)))

    assert result.passed is True
    assert result.stop_pipeline is False
    assert result.output["status"] == "cancelled"
    assert result.output["parsed"] == {}
