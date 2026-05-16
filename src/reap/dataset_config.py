"""Dataset processor configuration, auto-detection, and config file loading."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from datasets import Dataset

logger = logging.getLogger(__name__)

ProcessorKind = Literal[
    "chat",
    "lm",
    "instruction",
    "prompt_completion",
    "mmlu",
    "passthrough",
]


@dataclass
class DatasetProcessorSpec:
    """Declarative specification for preprocessing a HuggingFace dataset."""

    processor: ProcessorKind = "chat"
    messages_field: str = "messages"
    text_field: str = "text"
    tools_field: str = "tools"
    category_field: str | None = None
    instruction_field: str = "instruction"
    output_field: str = "output"
    prompt_field: str = "prompt"
    completion_field: str = "completion"
    question_field: str = "question"
    choices_field: str = "choices"
    user_role: str = "user"
    assistant_role: str = "assistant"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DatasetProcessorSpec:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in raw.items() if k in known}
        if "processor" not in filtered and "type" in raw:
            filtered["processor"] = raw["type"]
        return cls(**filtered)


def _column_set(dataset: Dataset) -> set[str]:
    return set(dataset.column_names)


def infer_processor_spec(dataset: Dataset) -> DatasetProcessorSpec:
    """Infer a processor spec from dataset column names."""
    columns = _column_set(dataset)

    if "messages" in columns or "conversations" in columns:
        messages_field = "messages" if "messages" in columns else "conversations"
        category = _first_present(
            columns, ("category", "source", "lang", "subject", "subset")
        )
        logger.info(
            "Auto-detected chat dataset (messages_field=%s, category_field=%s)",
            messages_field,
            category,
        )
        return DatasetProcessorSpec(
            processor="chat",
            messages_field=messages_field,
            category_field=category,
        )

    if {"instruction", "output"}.issubset(columns):
        logger.info("Auto-detected instruction/output chat dataset")
        return DatasetProcessorSpec(
            processor="instruction",
            instruction_field="instruction",
            output_field="output",
        )

    if {"instruction", "response"}.issubset(columns):
        logger.info("Auto-detected instruction/response chat dataset")
        return DatasetProcessorSpec(
            processor="instruction",
            instruction_field="instruction",
            output_field="response",
        )

    if {"prompt", "completion"}.issubset(columns):
        logger.info("Auto-detected prompt/completion chat dataset")
        return DatasetProcessorSpec(
            processor="prompt_completion",
            prompt_field="prompt",
            completion_field="completion",
        )

    if {"question", "choices"}.issubset(columns):
        logger.info("Auto-detected MMLU-style chat dataset")
        return DatasetProcessorSpec(
            processor="mmlu",
            question_field="question",
            choices_field="choices",
            category_field="subject" if "subject" in columns else None,
        )

    if {"query", "answers"}.issubset(columns) and "tools" in columns:
        logger.info(
            "Auto-detected function-calling chat dataset (query/answers/tools)"
        )
        return DatasetProcessorSpec(
            processor="chat",
            messages_field="messages" if "messages" in columns else "query",
        )

    if "text" in columns:
        logger.info("Auto-detected language-modeling dataset (text_field=text)")
        return DatasetProcessorSpec(processor="lm", text_field="text")

    if "content" in columns:
        logger.info("Auto-detected language-modeling dataset (text_field=content)")
        return DatasetProcessorSpec(processor="lm", text_field="content")

    raise ValueError(
        f"Could not auto-detect dataset processor for columns: {sorted(columns)}. "
        "Register the dataset in DATASET_REGISTRY or add an entry to a dataset config file."
    )


def _first_present(columns: set[str], candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in columns:
            return name
    return None


def load_dataset_config_file(path: str | Path) -> dict[str, DatasetProcessorSpec]:
    """Load dataset processor specs from a JSON config file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Dataset config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Dataset config must be a JSON object, got {type(raw).__name__}"
        )

    entries = raw.get("datasets", raw)
    if not isinstance(entries, dict):
        raise ValueError(
            "Dataset config must contain a 'datasets' object or top-level dataset keys"
        )

    return {
        dataset_name: DatasetProcessorSpec.from_dict(spec)
        for dataset_name, spec in entries.items()
    }


def merge_dataset_configs(
    *config_maps: dict[str, DatasetProcessorSpec],
) -> dict[str, DatasetProcessorSpec]:
    merged: dict[str, DatasetProcessorSpec] = {}
    for config_map in config_maps:
        merged.update(config_map)
    return merged


def resolve_processor_spec(
    dataset_name: str,
    dataset: Dataset,
    file_configs: dict[str, DatasetProcessorSpec] | None = None,
) -> DatasetProcessorSpec:
    """Resolve spec from config file, else auto-detect from columns."""
    if file_configs and dataset_name in file_configs:
        logger.info("Using dataset config entry for '%s'", dataset_name)
        return file_configs[dataset_name]

    return infer_processor_spec(dataset)


def make_map_fn(spec: DatasetProcessorSpec) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build a row-mapping function from a processor spec."""

    def map_fn(sample: dict[str, Any]) -> dict[str, Any]:
        if spec.processor == "passthrough":
            return sample

        if spec.processor == "chat":
            return sample

        if spec.processor == "lm":
            return {spec.text_field: sample[spec.text_field]}

        if spec.processor == "instruction":
            return {
                "messages": [
                    {
                        "role": spec.user_role,
                        "content": sample[spec.instruction_field],
                    },
                    {
                        "role": spec.assistant_role,
                        "content": sample[spec.output_field],
                    },
                ],
            }

        if spec.processor == "prompt_completion":
            return {
                "messages": [
                    {"role": spec.user_role, "content": sample[spec.prompt_field]},
                    {
                        "role": spec.assistant_role,
                        "content": sample[spec.completion_field],
                    },
                ],
            }

        if spec.processor == "mmlu":
            return {
                "messages": [
                    {
                        "role": spec.user_role,
                        "content": (
                            f"{sample[spec.question_field]} "
                            f"Choose from the following options: "
                            f"{sample[spec.choices_field]}"
                        ),
                    },
                ],
            }

        raise ValueError(f"Unsupported processor kind: {spec.processor}")

    return map_fn


def processor_kind_to_base(spec: DatasetProcessorSpec) -> Literal["chat", "lm"]:
    if spec.processor in ("lm",):
        return "lm"
    return "chat"
