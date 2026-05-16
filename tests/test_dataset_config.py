import json

import pytest
from datasets import Dataset

from reap.dataset_config import (
    DatasetProcessorSpec,
    infer_processor_spec,
    load_dataset_config_file,
    make_map_fn,
    resolve_processor_spec,
)
from reap.data import (
    BaseDatasetProcessor,
    ConfiguredDatasetProcessor,
    resolve_dataset_processor,
)


def test_infer_instruction_output():
    dataset = Dataset.from_dict(
        {
            "instruction": ["Write hello world"],
            "output": ['print("hello")'],
        }
    )
    spec = infer_processor_spec(dataset)
    assert spec.processor == "instruction"
    mapped = make_map_fn(spec)(dataset[0])
    assert mapped["messages"][0]["role"] == "user"
    assert mapped["messages"][1]["role"] == "assistant"


def test_infer_messages_chat():
    dataset = Dataset.from_dict(
        {
            "messages": [
                [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]
            ],
        }
    )
    spec = infer_processor_spec(dataset)
    assert spec.processor == "chat"
    assert spec.messages_field == "messages"


def test_load_dataset_config_file(tmp_path):
    config_path = tmp_path / "datasets.json"
    config_path.write_text(
        json.dumps(
            {
                "datasets": {
                    "org/custom": {
                        "processor": "prompt_completion",
                        "prompt_field": "prompt",
                        "completion_field": "completion",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    configs = load_dataset_config_file(config_path)
    assert "org/custom" in configs
    assert configs["org/custom"].processor == "prompt_completion"


def test_resolve_processor_spec_prefers_config(tmp_path):
    dataset = Dataset.from_dict({"text": ["hello"]})
    config_path = tmp_path / "datasets.json"
    config_path.write_text(
        json.dumps(
            {
                "datasets": {
                    "org/custom": {
                        "processor": "lm",
                        "text_field": "text",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    configs = load_dataset_config_file(config_path)
    spec = resolve_processor_spec("org/custom", dataset, configs)
    assert spec.processor == "lm"


def test_resolve_dataset_processor_auto_detect():
    dataset = Dataset.from_dict(
        {
            "instruction": ["a"],
            "output": ["b"],
        }
    )
    proc_cls = resolve_dataset_processor("unregistered/example", dataset)
    assert issubclass(proc_cls, BaseDatasetProcessor)
    processor = proc_cls(
        dataset=dataset,
        tokenizer=_FakeTokenizer(),
        max_input_len=32,
        split_by_category=False,
        truncate=False,
        batch_size=1,
    )
    batches = processor.get_processed_dataset(batches_per_category=1)
    assert len(batches["all"]) == 1


def test_fit_encoded_sample_never_skips_long_rows():
    dataset = Dataset.from_dict(
        {
            "instruction": ["word " * 200],
            "output": ["ok"],
        }
    )
    proc_cls = ConfiguredDatasetProcessor.for_spec(
        DatasetProcessorSpec(processor="instruction")
    )
    processor = proc_cls(
        dataset=dataset,
        tokenizer=_FakeTokenizer(),
        max_input_len=16,
        split_by_category=False,
        truncate=False,
        batch_size=1,
        pack_samples=False,
    )
    batches = processor.get_processed_dataset(batches_per_category=1)
    assert len(batches["all"]) == 1
    assert batches["all"][0]["input_ids"].shape[-1] == 16


class _FakeTokenizer:
    model_max_length = 2048
    pad_token_id = 0

    def __call__(self, text, truncation=False, max_length=None, return_tensors="pt"):
        import torch

        tokens = torch.tensor([[len(text)] * min(len(text.split()), max_length or len(text.split()))])
        if truncation and max_length is not None:
            tokens = tokens[:, :max_length]
        return {"input_ids": tokens}

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, **kwargs):
        return " ".join(message["content"] for message in messages)
