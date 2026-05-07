from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

from pipeline.object_detection.detections import (
    format_detection_context,
    resolve_yolo_input_mode,
)
from pipeline.object_detection.config import ANNOTATED_OUTPUT_DIR, ANNOTATED_OUTPUT_FILENAME


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT_FILE = SCRIPT_DIR / "llm" / "prompts" / "c1v1prompt.txt"


@dataclass
class PipelineContext:
    image_path: Path
    outputs: dict[str, Any] = field(default_factory=dict)

    def set_output(self, key: str, value: Any) -> None:
        self.outputs[key] = value

    def get_output(self, key: str, default: Any = None) -> Any:
        return self.outputs.get(key, default)


@dataclass(frozen=True)
class PipelineStageResult:
    stage_name: str
    label: str
    passed: bool
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    stop_pipeline: bool = False


@dataclass(frozen=True)
class PipelineEvent:
    event_type: str
    stage_name: str
    label: str
    message: str
    result: PipelineStageResult | None = None


@dataclass(frozen=True)
class PipelineRunResult:
    image_path: Path
    passed: bool
    stage_results: list[PipelineStageResult]
    outputs: dict[str, Any]


class PipelineStage(Protocol):
    name: str
    label: str

    def run(self, context: PipelineContext) -> PipelineStageResult:
        ...


class QualityScreeningStage:
    name = "quality_screening"
    label = "Quality Screening"

    def run(self, context: PipelineContext) -> PipelineStageResult:
        from pipeline.quality_screening.screening import run_quality_screening_from_path

        result = run_quality_screening_from_path(context.image_path)
        context.set_output(self.name, result)

        failed_checks = result.get("failed_checks", [])
        passed = bool(result.get("passed"))
        error = None
        if not passed:
            failed = ", ".join(failed_checks) if failed_checks else "unknown"
            error = f"Image failed quality screening: {failed}"

        return PipelineStageResult(
            stage_name=self.name,
            label=self.label,
            passed=passed,
            output=result,
            error=error,
            stop_pipeline=not passed,
        )


class ObjectDetectionStage:
    name = "object_detection"
    label = "Object Detection"

    def __init__(
        self,
        model_name: str = "best",
        show: bool = False,
        yolo_input_mode: str | None = None,
        use_annotation: bool | None = None,
    ):
        self.model_name = model_name
        self.show = show
        self.yolo_input_mode = resolve_yolo_input_mode(
            yolo_input_mode,
            use_annotation=use_annotation,
        )

    def run(self, context: PipelineContext) -> PipelineStageResult:
        from pipeline.object_detection.run import run as run_detection

        result = run_detection(
            image_path=str(context.image_path),
            model_name=self.model_name,
            show=self.show,
            save_annotation=self.yolo_input_mode == "annotated_image",
        )
        if result is None:
            result = {
                "passed": False,
                "error": "Object detection did not return a result.",
            }

        if self.yolo_input_mode == "annotated_image":
            annotated_image_path = Path(
                result.get(
                    "annotated_image_path",
                    ANNOTATED_OUTPUT_DIR / ANNOTATED_OUTPUT_FILENAME,
                )
            )
            if result.get("passed") and not annotated_image_path.exists():
                result["passed"] = False
                result["error"] = f"Annotated image not found: {annotated_image_path}"
        else:
            annotated_image_path = context.image_path

        result["annotated_image_path"] = str(annotated_image_path)
        result["yolo_input_mode"] = self.yolo_input_mode
        context.set_output(self.name, result)
        context.set_output("annotated_image_path", annotated_image_path)

        passed = bool(result.get("passed"))
        return PipelineStageResult(
            stage_name=self.name,
            label=self.label,
            passed=passed,
            output=result,
            error=result.get("error"),
            stop_pipeline=not passed,
        )


class LLMReasoningStage:
    name = "llm_reasoning"
    label = "LLM Reasoning"

    def __init__(
        self,
        model_name: str = "ministral",
        prompt_file: str | Path = DEFAULT_PROMPT_FILE,
        additional_context: str | None = None,
        context_file: str | Path | None = None,
        yolo_input_mode: str | None = None,
        use_annotation: bool | None = None,
    ):
        self.model_name = model_name
        self.prompt_file = Path(prompt_file)
        self.additional_context = additional_context
        self.context_file = Path(context_file) if context_file else None
        self.yolo_input_mode = resolve_yolo_input_mode(
            yolo_input_mode,
            use_annotation=use_annotation,
        )

    def run(self, context: PipelineContext) -> PipelineStageResult:
        from pipeline.llm.inference import load_system_prompt, run_llm_inference

        if self.yolo_input_mode == "annotated_image":
            image_path = context.get_output("annotated_image_path", context.image_path)
            additional_context = self.additional_context
        else:
            image_path = context.image_path
            detection_result = context.get_output("object_detection", {})
            detection_context = format_detection_context(
                detection_result,
                self.yolo_input_mode,
            )
            additional_context = _join_context_blocks(
                self.additional_context,
                detection_context,
            )
        system_prompt = load_system_prompt(str(self.prompt_file))
        result = run_llm_inference(
            model_name=self.model_name,
            image_path=str(image_path),
            system_prompt=system_prompt,
            prompt_file=str(self.prompt_file),
            additional_context=additional_context,
            context_file=str(self.context_file) if self.context_file else None,
        )
        result["yolo_input_mode"] = self.yolo_input_mode
        context.set_output(self.name, result)

        return PipelineStageResult(
            stage_name=self.name,
            label=self.label,
            passed=True,
            output=result,
        )


class PipelineRunner:
    def __init__(self, stages: Iterable[PipelineStage]):
        self.stages = list(stages)

    def iter_events(self, image_path: str | Path):
        context = PipelineContext(image_path=Path(image_path).resolve())

        for stage in self.stages:
            yield PipelineEvent(
                event_type="stage_started",
                stage_name=stage.name,
                label=stage.label,
                message=f"{stage.label} started",
            )

            try:
                result = stage.run(context)
            except Exception as exc:
                result = PipelineStageResult(
                    stage_name=stage.name,
                    label=stage.label,
                    passed=False,
                    error=str(exc),
                    stop_pipeline=True,
                )

            yield PipelineEvent(
                event_type="stage_finished",
                stage_name=stage.name,
                label=stage.label,
                message=f"{stage.label} finished",
                result=result,
            )

            if result.stop_pipeline:
                break

    def run(self, image_path: str | Path) -> PipelineRunResult:
        stage_results = []
        context_outputs = {}

        for event in self.iter_events(image_path):
            if event.result is not None:
                stage_results.append(event.result)
                context_outputs[event.result.stage_name] = event.result.output

        return PipelineRunResult(
            image_path=Path(image_path).resolve(),
            passed=all(result.passed for result in stage_results),
            stage_results=stage_results,
            outputs=context_outputs,
        )


def create_default_pipeline(
    yolo_model: str = "best",
    llm_model: str = "ministral",
    prompt_file: str | Path = DEFAULT_PROMPT_FILE,
    context_file: str | Path | None = None,
    skip_quality_screening: bool = False,
    yolo_input_mode: str | None = None,
    use_annotation: bool | None = None,
) -> PipelineRunner:
    resolved_yolo_input_mode = resolve_yolo_input_mode(
        yolo_input_mode,
        use_annotation=use_annotation,
    )
    stages = []
    if not skip_quality_screening:
        stages.append(QualityScreeningStage())
    stages.append(
        ObjectDetectionStage(
            model_name=yolo_model,
            yolo_input_mode=resolved_yolo_input_mode,
        )
    )
    stages.append(LLMReasoningStage(
        model_name=llm_model,
        prompt_file=prompt_file,
        context_file=context_file,
        yolo_input_mode=resolved_yolo_input_mode,
    ))
    return PipelineRunner(stages=stages)


def _join_context_blocks(*blocks: str | None) -> str | None:
    joined = "\n\n".join(block.strip() for block in blocks if block and block.strip())
    return joined or None
