"""Convert datasets to transformers BatchEncoded or vLLM TokensPrompt formats.

We follow the OpenAI spec for conversational datasets.

ie..,
messages = [
    {"role": "system", "content": "You are AGI"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "What is my purpose?"},
]

Includes the ability to select from specific categories within the dataset and convert
the dataset into either a language modelling dataset with attention applied to every
token or a prompt-completion dataset for training on completions only with SFTTrainer.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any
import os
import uuid
import json
import re
import random
import logging
import multiprocessing as mp


import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, BatchEncoding
from vllm import TokensPrompt

from reap.dataset_config import (
    DatasetProcessorSpec,
    load_dataset_config_file,
    make_map_fn,
    processor_kind_to_base,
    resolve_processor_spec,
)


logger = logging.getLogger(__name__)


def _resolve_map_num_proc(map_num_proc: int | None) -> int | None:
    """Return worker count for parallel data loading (None = single-process).

    Used for HuggingFace ``Dataset.map``, per-category batch building, and
    sample tokenization. Defaults to ``max(1, cpu_count - 1)`` when unset.
    """
    if map_num_proc is not None:
        return map_num_proc if map_num_proc > 1 else None
    cpus = os.cpu_count() or 1
    if cpus <= 1:
        return None
    return cpus - 1


# Process-pool worker state (set via pool initializers; not thread-safe).
_MP_PROCESSOR: Any = None
_MP_CATEGORY_DATASET: Dataset | None = None
_MP_PACKED_CONTEXT: _PackedBatchContext | None = None


@dataclass(frozen=True)
class _PackedBatchContext:
    category: str
    batches_per_category: int
    return_vllm_tokens_prompt: bool
    batch_size: int


def _set_mp_processor(processor: Any) -> None:
    global _MP_PROCESSOR
    _MP_PROCESSOR = processor


def _set_mp_packed_context(
    processor: Any,
    category_dataset: Dataset,
    packed_context: _PackedBatchContext,
) -> None:
    global _MP_PROCESSOR, _MP_CATEGORY_DATASET, _MP_PACKED_CONTEXT
    _MP_PROCESSOR = processor
    _MP_CATEGORY_DATASET = category_dataset
    _MP_PACKED_CONTEXT = packed_context


def _encode_sample_mp(sample: dict[str, Any]) -> torch.Tensor:
    if _MP_PROCESSOR is None:
        raise RuntimeError("Multiprocessing encode worker is missing processor state")
    return _MP_PROCESSOR._fit_encoded_sample(_MP_PROCESSOR._encode_sample(sample))


def _process_category_mp(
    args: tuple[str, int],
) -> tuple[str, list[TokensPrompt] | list[BatchEncoding]]:
    category, batches_per_category = args
    if _MP_PROCESSOR is None:
        raise RuntimeError("Multiprocessing category worker is missing processor state")
    batches = _MP_PROCESSOR._process_batches_for_category(
        category,
        batches_per_category,
        parallel_categories=False,
    )
    return category, batches


def _build_packed_batch_mp(_batch_idx: int) -> TokensPrompt | dict[str, torch.Tensor]:
    if (
        _MP_PROCESSOR is None
        or _MP_CATEGORY_DATASET is None
        or _MP_PACKED_CONTEXT is None
    ):
        raise RuntimeError("Multiprocessing packed-batch worker is missing state")
    return _MP_PROCESSOR._build_one_packed_batch(
        _MP_CATEGORY_DATASET,
        _MP_PACKED_CONTEXT.category,
        _MP_PACKED_CONTEXT.return_vllm_tokens_prompt,
    )


def _maybe_json_load(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def _normalize_message_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                else:
                    parts.append(json.dumps(item, sort_keys=True))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _normalize_messages_for_chat_template(messages):
    if not isinstance(messages, list):
        return messages

    normalized_messages = []
    for message in messages:
        if not isinstance(message, dict) or not message.get("tool_calls"):
            normalized_messages.append(message)
            continue

        normalized_message = dict(message)
        normalized_tool_calls = []
        for tool_call in message["tool_calls"]:
            if not isinstance(tool_call, dict) or not isinstance(
                tool_call.get("function"), dict
            ):
                normalized_tool_calls.append(tool_call)
                continue

            normalized_tool_call = dict(tool_call)
            normalized_function = dict(tool_call["function"])
            normalized_function["arguments"] = _maybe_json_load(
                normalized_function.get("arguments")
            )
            normalized_tool_call["function"] = normalized_function
            normalized_tool_calls.append(normalized_tool_call)

        normalized_message["tool_calls"] = normalized_tool_calls
        normalized_messages.append(normalized_message)

    return normalized_messages


@dataclass
class CompositeDatasetComponent:
    """A single component of a composite dataset specification.

    Attributes:
        name: HuggingFace dataset name (e.g., "open-r1/Mixture-of-Thoughts").
        split: HF dataset split to load (e.g., "train", "tool"). None means use
               the default split from DatasetArgs.
        subset: HF dataset config/subset name (e.g., "code", "math"). None means
                do not pass a subset to ``load_dataset``.
        num_batches: Number of batches to draw from this component.
    """

    name: str
    split: str | None
    subset: str | None
    num_batches: int


# Regex to parse a single component: <name>[<subset>](<split>):<num_batches>
# Examples:
#   "theblackcat102/evol-codealpaca-v1:4096"            -> name="theblackcat102/evol-codealpaca-v1", subset=None, split=None, num_batches=4096
#   "open-r1/Mixture-of-Thoughts[code]:4096"            -> name="open-r1/Mixture-of-Thoughts", subset="code", split=None, num_batches=4096
#   "SWE-bench/SWE-smith-trajectories(tool):4096"      -> name="SWE-bench/SWE-smith-trajectories", subset=None, split="tool", num_batches=4096
#   "dataset[subset](split):4096"                      -> name="dataset", subset="subset", split="split", num_batches=4096
_COMPOSITE_COMPONENT_RE = re.compile(
    r"^(?P<name>[^\[\]()[:,]+)"  # dataset name
    r"(?:\[(?P<subset>[^\]]+)\])?"  # optional [subset]
    r"(?:\((?P<split>[^\)]+)\))?"  # optional (split)
    r":(?P<num_batches>\d+)$"  # :num_batches (required for composite)
)


def parse_composite_dataset_spec(
    spec: str,
    default_split: str = "train",
    default_subset: str | None = None,
) -> list[CompositeDatasetComponent] | None:
    """Parse a composite dataset specification string.

    Returns a list of CompositeDatasetComponent if the spec is composite
    (contains comma-separated entries with :num_batches), or None if the spec
    is a single dataset name (backward-compatible).

    Format: ``name1[subset1](split1):N1,name2:N2,name3[subset3]:N3,...``

    Args:
        spec: The dataset specification string.
        default_split: The default split to use when no split is specified.
        default_subset: The default subset to use when no subset is specified.

    Returns:
        List of parsed components, or None if this is a plain single-dataset name.

    Raises:
        ValueError: If the spec looks like a composite spec but has parse errors.
    """
    # A composite spec must contain at least one colon followed by digits.
    # Single dataset names like "theblackcat102/evol-codealpaca-v1" won't match.
    if ":" not in spec:
        return None

    # Could be a single dataset with a colon in the name (unlikely for HF) —
    # but to be safe, also require at least one comma OR the entire string to
    # match the component pattern.
    parts = [p.strip() for p in spec.split(",")]

    components = []
    for i, part in enumerate(parts):
        m = _COMPOSITE_COMPONENT_RE.match(part)
        if m is None:
            if len(parts) == 1:
                # Single entry that doesn't match composite format — treat as
                # a plain dataset name (backward compatible).
                return None
            raise ValueError(
                f"Failed to parse composite dataset component {i}: '{part}'. "
                f"Expected format: <dataset_name>[<subset>](<split>):<num_batches>. "
                f"Full spec: '{spec}'"
            )
        name = m.group("name").strip()
        subset = m.group("subset")
        if subset is None:
            subset = default_subset
        split = m.group("split")
        if split is None:
            split = default_split
        num_batches = int(m.group("num_batches"))
        components.append(
            CompositeDatasetComponent(
                name=name,
                split=split,
                subset=subset,
                num_batches=num_batches,
            )
        )

    if not components:
        return None

    logger.info(
        f"Parsed composite dataset spec with {len(components)} components: "
        + ", ".join(
            f"{c.name}"
            f"{f'[{c.subset}]' if c.subset is not None else ''}"
            f"{f'({c.split})' if c.split is not None else ''}"
            f":{c.num_batches}"
            for c in components
        )
    )
    return components


def _load_raw_dataset(dataset_name, split, subset=None):
    """Load a raw HuggingFace dataset, handling special cases like C4."""
    try:
        if dataset_name == "allenai/c4":
            file_url = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz"
            return load_dataset(
                "json", data_files={"train": file_url}, split="train", streaming=False
            )
        else:
            load_kwargs = {}
            if subset is not None:
                load_kwargs = {"name": subset}
            return load_dataset(dataset_name, split=split, **load_kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load dataset '{dataset_name}' (subset={subset}, split={split}): {e}"
        )


_FILE_DATASET_CONFIGS: dict[str, dict[str, DatasetProcessorSpec]] = {}


def _get_file_dataset_configs(
    dataset_config_path: str | None,
) -> dict[str, DatasetProcessorSpec]:
    if dataset_config_path is None:
        return {}
    if dataset_config_path not in _FILE_DATASET_CONFIGS:
        _FILE_DATASET_CONFIGS[dataset_config_path] = load_dataset_config_file(
            dataset_config_path
        )
    return _FILE_DATASET_CONFIGS[dataset_config_path]


# --- Base Dataset Processors --------------------------------------------------


class BaseDatasetProcessor(ABC):
    category_field: str = "category"
    text_field: str = "text"
    completion_field: str = "completion"
    prompt_field: str = "prompt"
    messages_field: str = "messages"
    tools_field: str = "tools"
    all_categories_label: str = "all"

    def __init__(
        self,
        dataset: Dataset | DatasetDict,
        tokenizer: AutoTokenizer,
        pack_samples: bool = True,
        max_input_len: int | None = None,
        split: str | None = None,
        split_by_category: bool = True,
        return_vllm_tokens_prompt: bool = False,
        truncate: bool = False,
        select_only_categories: list[str] | str | None = None,
        batch_size: int = 1,
        map_num_proc: int | None = None,
    ):
        """Defines base functionality for all Dataset Processors.

        Args:
            dataset (Dataset | DatasetDict): _description_
            tokenizer (AutoTokenizer): _description_
            split (str | None, optional): _description_. Defaults to None.
            split_by_category (bool, optional): _description_. Defaults to True.
            return_vllm_tokens_prompt (bool, optional): If True, will return
                TokensPrompt objects instead of BatchEncoding. Defaults to False
            truncate (bool, optional): If True, apply tokenizer truncation to
                ``max_input_len`` during encoding. If False, encode full text and
                clip to ``max_input_len`` when building calibration batches (rows
                are never dropped for length).
            batch_size (int, optional): Number of samples per batch. Defaults to 1.
            map_num_proc (int | None, optional): Worker processes for parallel
                dataset mapping, category batch building, and tokenization.
                None uses ``max(1, cpu_count - 1)``. Set to 1 to disable
                multiprocessing.

        """
        if isinstance(dataset, DatasetDict):
            if split is None:
                split = list(dataset.keys())[0]
                logging.warning(
                    f"Using split '{split}' as default for dataset. Available "
                    f"splits: {list(dataset.keys())}",
                )
            dataset = dataset[split]
        if max_input_len is None:
            max_input_len = tokenizer.model_max_length
            logger.warning(
                f"max_input_len is set to {max_input_len} as per tokenizer's "
                f"model_max_length. This will be used for truncation.",
            )
        self.pack_samples = pack_samples
        self.max_input_len = max_input_len
        self.dataset = dataset
        self._mapped_dataset = None
        self.tokenizer = tokenizer
        self.split_by_category = split_by_category
        self.return_vllm_tokens_prompt = return_vllm_tokens_prompt
        self.truncate = truncate
        self.categories = self.get_categories()
        if isinstance(select_only_categories, str):
            select_only_categories = [select_only_categories]
        self.select_only_categories = select_only_categories
        self.batch_size = batch_size
        self.map_num_proc = map_num_proc
        self._row_map_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None
        if self.select_only_categories:
            logger.warning(
                "select_only_categories is not None but split_by_category "
                "was False. Setting split_by_category to True and processing "
                f"categories: {self.select_only_categories}"
            )
            self.split_by_category = True
            if self.category_field not in self.dataset.column_names:
                raise RuntimeError(
                    f"Category field '{self.category_field}' not found in dataset. "
                    "Please ensure the dataset has a category field.",
                )
            for category in self.select_only_categories:
                if category not in self.categories:
                    raise RuntimeError(
                        f"Category '{category}' not found in dataset. "
                        "Please ensure the dataset has the specified categories.",
                    )

    @staticmethod
    @abstractmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        """Map a row of the dataset to the desired output format.

        EG., map "prompts" and "completions" to "messages" for chat datasets.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses.",
        )

    @abstractmethod
    def _encode_sample(self, sample: dict) -> torch.Tensor:
        """Encode a sample from the desired category of the dataset into
        tokens.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses.",
        )

    def _fit_encoded_sample(self, encoded_sample: torch.Tensor) -> torch.Tensor:
        """Clip encoded tokens to ``max_input_len`` without dropping the row."""
        if encoded_sample.shape[-1] > self.max_input_len:
            return encoded_sample[:, : self.max_input_len]
        return encoded_sample

    def _map_dataset_rows(self) -> Dataset:
        map_fn = self._row_map_fn or self._map_fn
        num_proc = _resolve_map_num_proc(self.map_num_proc)
        map_kwargs: dict[str, Any] = {"desc": "Mapping dataset rows"}
        if num_proc is not None:
            map_kwargs["num_proc"] = num_proc
        return self.dataset.map(map_fn, **map_kwargs)

    def get_processed_dataset(
        self, batches_per_category: int
    ) -> dict[str, list[TokensPrompt]] | dict[str, list[BatchEncoding]]:
        """Get requests for each category in the dataset."""
        if self._mapped_dataset is None:
            self._mapped_dataset = self._map_dataset_rows()
        if self.split_by_category:
            categories = (
                self.categories
                if self.select_only_categories is None
                else self.select_only_categories
            )
            return self._process_all_categories(categories, batches_per_category)
        return {
            self.all_categories_label: self._process_batches_for_category(
                self.all_categories_label,
                batches_per_category,
            ),
        }

    def _process_all_categories(
        self,
        categories: list[str],
        batches_per_category: int,
    ) -> dict[str, list[TokensPrompt]] | dict[str, list[BatchEncoding]]:
        num_proc = _resolve_map_num_proc(self.map_num_proc)
        if num_proc is None or len(categories) <= 1:
            return {
                category: self._process_batches_for_category(
                    category, batches_per_category
                )
                for category in categories
            }

        max_workers = min(num_proc, len(categories))
        logger.info(
            "Processing %d categories with %d worker processes",
            len(categories),
            max_workers,
        )
        results: dict[str, list[TokensPrompt] | list[BatchEncoding]] = {}
        try:
            mp_context = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=mp_context,
                initializer=_set_mp_processor,
                initargs=(self,),
            ) as executor:
                futures = {
                    executor.submit(
                        _process_category_mp, (category, batches_per_category)
                    ): category
                    for category in categories
                }
                for future in as_completed(futures):
                    category, batches = future.result()
                    results[category] = batches
        except Exception as exc:
            logger.warning(
                "Parallel category processing failed (%s); falling back to "
                "single-process loading",
                exc,
            )
            return {
                category: self._process_batches_for_category(
                    category, batches_per_category
                )
                for category in categories
            }
        return results

    def _encode_samples_parallel(
        self,
        samples: list[dict[str, Any]],
        *,
        allow_parallel: bool = True,
    ) -> list[torch.Tensor]:
        if not samples:
            return []

        num_proc = _resolve_map_num_proc(self.map_num_proc) if allow_parallel else None
        if num_proc is None or len(samples) == 1:
            return [
                self._fit_encoded_sample(self._encode_sample(sample))
                for sample in samples
            ]

        max_workers = min(num_proc, len(samples))
        try:
            mp_context = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=mp_context,
                initializer=_set_mp_processor,
                initargs=(self,),
            ) as executor:
                chunksize = max(1, len(samples) // (max_workers * 4))
                return list(
                    executor.map(_encode_sample_mp, samples, chunksize=chunksize)
                )
        except Exception as exc:
            logger.warning(
                "Parallel sample encoding failed (%s); falling back to "
                "single-process encoding",
                exc,
            )
            return [
                self._fit_encoded_sample(self._encode_sample(sample))
                for sample in samples
            ]

    def get_categories(self) -> list[str]:
        """Get the unique categories in the dataset."""
        if self.category_field is None:
            logger.warning(
                "No category field specified for dataset, returning 'all' category."
            )
            return ["all"]
        return self.dataset.unique(self.category_field)

    def _process_batches_for_category(
        self,
        category: str,
        batches_per_category: int,
        parallel_categories: bool = True,
    ) -> list[TokensPrompt] | list[BatchEncoding]:
        if category != self.all_categories_label:
            category_dataset = self._mapped_dataset.filter(
                lambda sample: sample[self.category_field] == category,
            )
        else:
            category_dataset = self._mapped_dataset
            category = self.all_categories_label

        if self.pack_samples:
            return self._process_batches_for_category_packed(
                category,
                batches_per_category,
                category_dataset,
                parallel_categories=parallel_categories,
            )
        return self._process_batches_for_category_unpacked(
            category,
            batches_per_category,
            category_dataset,
            parallel_encoding=parallel_categories,
        )

    def _collate_batch(self, batch: list[dict[str, torch.Tensor]]) -> BatchEncoding:
        """Collate a list of tokenized samples into a padded batch.

        Args:
            batch: List of dicts, each containing:
                - "input_ids": Tensor with shape (1, seq_len)
                - "attention_mask": Tensor with shape (1, seq_len)

        Returns:
            BatchEncoding with:
                - input_ids: Tensor of shape (batch_size, max_seq_len)
                - attention_mask: Tensor of shape (batch_size, max_seq_len)
        """
        max_len = max(sample["input_ids"].shape[-1] for sample in batch)
        pad_token_id = self.tokenizer.pad_token_id or 0

        padded_input_ids = []
        padded_attention_masks = []

        for sample in batch:
            input_ids = sample["input_ids"]
            attention_mask = sample["attention_mask"]

            seq_len = input_ids.shape[-1]
            padding_len = max_len - seq_len

            if padding_len > 0:
                input_padding = torch.full(
                    (input_ids.shape[0], padding_len),
                    pad_token_id,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                mask_padding = torch.zeros(
                    (attention_mask.shape[0], padding_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                input_ids = torch.cat([input_ids, input_padding], dim=-1)
                attention_mask = torch.cat([attention_mask, mask_padding], dim=-1)

            padded_input_ids.append(input_ids)
            padded_attention_masks.append(attention_mask)

        return BatchEncoding(
            {
                "input_ids": torch.cat(padded_input_ids, dim=0),
                "attention_mask": torch.cat(padded_attention_masks, dim=0),
            }
        )

    def _sample_unique_indices(
        self,
        category: str,
        category_dataset: Dataset,
        num_samples: int,
        sampled: list[int],
    ) -> list[int]:
        """Draw up to ``num_samples`` unique row indices without replacement."""
        if len(category_dataset) == 0:
            return []

        sampled_set = set(sampled)
        new_indices: list[int] = []
        attempts = 0
        max_attempts = max(num_samples * 20, 1)
        while len(new_indices) < num_samples and attempts < max_attempts:
            attempts += 1
            if len(sampled_set) >= len(category_dataset):
                break
            sample_idx = random.randint(0, len(category_dataset) - 1)
            if sample_idx in sampled_set:
                continue
            sampled_set.add(sample_idx)
            new_indices.append(sample_idx)

        if len(new_indices) < num_samples:
            logger.warning(
                "Not enough unique samples in category '%s' to collect %d rows; "
                "only %d were available.",
                category,
                num_samples,
                len(new_indices),
            )
        sampled.extend(new_indices)
        return new_indices

    def _build_one_packed_batch(
        self,
        category_dataset: Dataset,
        category: str,
        return_vllm_tokens_prompt: bool,
    ) -> TokensPrompt | dict[str, torch.Tensor]:
        sampled: list[int] = []
        seq = torch.zeros((1, self.max_input_len), dtype=torch.long)
        seq_idx = 0
        attention_mask = torch.zeros((1, self.max_input_len), dtype=torch.long)
        while seq_idx < self.max_input_len:
            if len(sampled) >= len(category_dataset):
                logger.warning(
                    "Not enough samples to pack sequence to max_input_len in "
                    "category '%s'.",
                    category,
                )
                break
            new_indices = self._sample_unique_indices(
                category,
                category_dataset,
                num_samples=1,
                sampled=sampled,
            )
            if not new_indices:
                break
            sample = category_dataset[new_indices[0]]
            encoded_sample = self._fit_encoded_sample(self._encode_sample(sample))
            end_seq = seq_idx + encoded_sample.shape[-1]
            if end_seq > self.max_input_len:
                encoded_sample = encoded_sample[:, : (self.max_input_len - seq_idx)]
                end_seq = self.max_input_len
            seq[:, seq_idx:end_seq] = encoded_sample
            attention_mask[:, seq_idx:end_seq] = 1
            seq_idx = end_seq + 1

        if return_vllm_tokens_prompt:
            return TokensPrompt(prompt_token_ids=seq[0, :-1].tolist())
        return {"input_ids": seq, "attention_mask": attention_mask}

    def _process_batches_for_category_unpacked(
        self,
        category: str,
        batches_per_category: int,
        category_dataset: Dataset,
        parallel_encoding: bool = True,
    ) -> list[TokensPrompt] | list[BatchEncoding]:
        if self.return_vllm_tokens_prompt:
            target_samples = batches_per_category
        else:
            target_samples = batches_per_category * self.batch_size

        sampled: list[int] = []
        sample_indices = self._sample_unique_indices(
            category,
            category_dataset,
            target_samples,
            sampled,
        )
        if not sample_indices:
            return []

        samples = [category_dataset[sample_idx] for sample_idx in sample_indices]
        encoded_samples = self._encode_samples_parallel(
            samples,
            allow_parallel=parallel_encoding,
        )

        if self.return_vllm_tokens_prompt:
            return [
                TokensPrompt(prompt_token_ids=encoded_sample[0, :-1].tolist())
                for encoded_sample in encoded_samples
            ]

        processed_samples: list[BatchEncoding] = []
        current_batch: list[dict[str, torch.Tensor]] = []
        for encoded_sample in encoded_samples:
            attention_mask = torch.ones((1, encoded_sample.shape[-1]), dtype=torch.long)
            current_batch.append(
                {"input_ids": encoded_sample, "attention_mask": attention_mask}
            )
            if len(current_batch) >= self.batch_size:
                processed_samples.append(self._collate_batch(current_batch))
                current_batch = []

        if current_batch:
            processed_samples.append(self._collate_batch(current_batch))

        if len(processed_samples) < batches_per_category:
            logger.warning(
                "Not enough samples in category '%s' to reach %d data batches. "
                "Only %d batches were produced.",
                category,
                batches_per_category,
                len(processed_samples),
            )
        return processed_samples[:batches_per_category]

    def _process_batches_for_category_packed(
        self,
        category: str,
        batches_per_category: int,
        category_dataset: Dataset,
        parallel_categories: bool = True,
    ) -> list[TokensPrompt] | list[BatchEncoding]:
        num_proc = _resolve_map_num_proc(self.map_num_proc)
        use_parallel_batches = (
            num_proc is not None
            and batches_per_category > 1
            and not parallel_categories
        )

        if use_parallel_batches:
            packed_context = _PackedBatchContext(
                category=category,
                batches_per_category=batches_per_category,
                return_vllm_tokens_prompt=self.return_vllm_tokens_prompt,
                batch_size=self.batch_size,
            )
            max_workers = min(num_proc, batches_per_category)
            try:
                mp_context = mp.get_context("spawn")
                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    mp_context=mp_context,
                    initializer=_set_mp_packed_context,
                    initargs=(self, category_dataset, packed_context),
                ) as executor:
                    packed_items = list(
                        executor.map(
                            _build_packed_batch_mp,
                            range(batches_per_category),
                        )
                    )
            except Exception as exc:
                logger.warning(
                    "Parallel packed-batch building failed (%s); falling back to "
                    "single-process loading",
                    exc,
                )
                use_parallel_batches = False
            else:
                return self._collate_packed_batch_items(packed_items)

        processed_samples: list[TokensPrompt] | list[BatchEncoding] = []
        current_batch: list[dict[str, torch.Tensor]] = []
        while len(processed_samples) < batches_per_category:
            packed_item = self._build_one_packed_batch(
                category_dataset,
                category,
                self.return_vllm_tokens_prompt,
            )
            if self.return_vllm_tokens_prompt:
                processed_samples.append(packed_item)
            else:
                current_batch.append(packed_item)
                if len(current_batch) >= self.batch_size:
                    processed_samples.append(self._collate_batch(current_batch))
                    current_batch = []

        if current_batch and not self.return_vllm_tokens_prompt:
            processed_samples.append(self._collate_batch(current_batch))

        return processed_samples

    def _collate_packed_batch_items(
        self,
        packed_items: list[TokensPrompt | dict[str, torch.Tensor]],
    ) -> list[TokensPrompt] | list[BatchEncoding]:
        if self.return_vllm_tokens_prompt:
            return packed_items

        processed_samples: list[BatchEncoding] = []
        current_batch: list[dict[str, torch.Tensor]] = []
        for packed_item in packed_items:
            current_batch.append(packed_item)
            if len(current_batch) >= self.batch_size:
                processed_samples.append(self._collate_batch(current_batch))
                current_batch = []
        if current_batch:
            processed_samples.append(self._collate_batch(current_batch))
        return processed_samples


class ChatDatasetProcessor(BaseDatasetProcessor):
    def _encode_sample(self, sample: dict) -> torch.Tensor:
        chat_template_kwargs = {}
        messages = _normalize_messages_for_chat_template(sample[self.messages_field])
        if self.tools_field in sample:
            chat_template_kwargs = {"tools": _maybe_json_load(sample[self.tools_field])}
        chat_sample = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
            **chat_template_kwargs,
        )
        return self.tokenizer(
            chat_sample,
            truncation=self.truncate,
            max_length=self.max_input_len if self.truncate else None,
            return_tensors="pt",
        )["input_ids"]

    def get_llmcompressor_dataset(self) -> Dataset:
        """Get the mapped dataset without tokenization applied."""

        def chat_template_fn(sample: dict[str, any]) -> dict[str, any]:
            """Apply chat template to the sample."""
            chat_template_kwargs = {}
            messages = _normalize_messages_for_chat_template(
                sample[self.messages_field]
            )
            if self.tools_field in sample:
                chat_template_kwargs = {
                    "tools": _maybe_json_load(sample[self.tools_field])
                }
            chat_sample = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=False,
                **chat_template_kwargs,
            )
            return {"text": chat_sample}

        if self._mapped_dataset is None:
            self._mapped_dataset = self._map_dataset_rows()

        return self._mapped_dataset.map(chat_template_fn)


class LMDatasetProcessor(BaseDatasetProcessor):
    def _encode_sample(self, sample: str) -> torch.Tensor:
        return self.tokenizer(
            sample[self.text_field],
            truncation=self.truncate,
            max_length=self.max_input_len if self.truncate else None,
            return_tensors="pt",
        )["input_ids"]

    def get_llmcompressor_dataset(self) -> Dataset:
        """Get the mapped dataset without tokenization applied."""

        if self._mapped_dataset is None:
            self._mapped_dataset = self._map_dataset_rows()

        return self._mapped_dataset


class ConfiguredDatasetProcessor:
    """Factory for dataset processors built from config or auto-detection."""

    @classmethod
    def for_spec(cls, spec: DatasetProcessorSpec) -> type[BaseDatasetProcessor]:
        base_cls = (
            LMDatasetProcessor
            if processor_kind_to_base(spec) == "lm"
            else ChatDatasetProcessor
        )
        row_map_fn = make_map_fn(spec)

        class _ConfiguredProcessor(base_cls):
            def __init__(self, **kwargs):
                self.messages_field = spec.messages_field
                self.text_field = spec.text_field
                self.tools_field = spec.tools_field
                self.category_field = spec.category_field
                super().__init__(**kwargs)
                self._row_map_fn = row_map_fn

            @staticmethod
            def _map_fn(sample: dict[str, Any]) -> dict[str, Any]:
                return sample

        _ConfiguredProcessor.__name__ = f"Configured{spec.processor.title()}Processor"
        _ConfiguredProcessor.__qualname__ = _ConfiguredProcessor.__name__
        return _ConfiguredProcessor


### --- Concrete Implementations -----------------------------------------------


class CodeFeedbackChatDataset(ChatDatasetProcessor):
    category_field: str = "lang"
    messages_field: str = "text_fieldmessages"

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return sample


class TuluSFTMixtureChatDataset(ChatDatasetProcessor):
    category_field: str = "source"

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return sample


class PersonasMathChatDataset(ChatDatasetProcessor):
    """Dataset for Tulu-3 SFT Personas Math."""

    category_field: str = None

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return sample


class WildChatSFTMixtureChatDataset(ChatDatasetProcessor):
    category_field: str = "langauge"

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return sample


class MmluChatDataset(ChatDatasetProcessor):
    category_field: str = "subject"

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"{sample['question']} "
                        f"Choose from the following options: {sample['choices']}"
                    ),
                },
            ],
        }


class MagicoderEvolInstructChatDataset(ChatDatasetProcessor):
    """Dataset for Magicoder-Evol-Instruct-110K."""

    category_field: str = None

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return {
            "messages": [
                {"role": "user", "content": sample["instruction"]},
                {"role": "assistant", "content": sample["response"]},
            ],
        }


class C4LMDataset(LMDatasetProcessor):
    category_field: str = None

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return sample


class CodeAlpacaChatDataset(ChatDatasetProcessor):
    """Dataset for evol-codealpaca-v1."""

    category_field: str = None

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return {
            "messages": [
                {"role": "user", "content": sample["instruction"]},
                {"role": "assistant", "content": sample["output"]},
            ],
        }


class WritingPromptsChatDataset(ChatDatasetProcessor):
    """Dataset for WritingPrompts_curated."""

    category_field: str = None

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": f"Please write a creative story using the following writing prompt:\n\n {sample['prompt']}",
                },
                {"role": "assistant", "content": sample["body"]},
            ],
        }


class MixtureOfThoughtsDataset(ChatDatasetProcessor):
    category_field: str = None

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return sample


class XLamFunctionCallingDataset(ChatDatasetProcessor):
    category_field: str = None

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        tool_calls = []
        gt_tool_calls = _maybe_json_load(sample["answers"])

        for tool_call in gt_tool_calls:
            tool_calls.append(
                {
                    "function": {
                        "arguments": json.dumps(tool_call["arguments"]),
                        "name": tool_call["name"],
                    },
                    "id": f"chatcmpl-tool-{uuid.uuid4()}",
                    "type": "function",
                }
            )

        return {
            "messages": [
                {"role": "user", "content": sample["query"]},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls,
                },
            ],
            "tools": (
                sample["tools"]
                if isinstance(sample["tools"], str)
                else json.dumps(sample["tools"])
            ),
        }


class SWESmithTrajectoriesDataset(ChatDatasetProcessor):
    category_field: str = None

    tools = [
        {
            "type": "function",
            "function": {
                "name": "str_replace_editor",
                "description": (
                    "Custom editing tool for viewing, creating and editing files.\n"
                    "State is persistent across calls. If `path` is a file, `view` shows `cat -n`; "
                    "if a directory, `view` lists non-hidden entries up to 2 levels. "
                    "The `create` command fails if `path` already exists as a file. "
                    "Long outputs may be truncated with '<response clipped>'. "
                    "`undo_edit` reverts the last edit for `path`.\n\n"
                    "Notes for `str_replace`:\n"
                    "- `old_str` must match EXACTLY one or more consecutive lines (watch whitespace).\n"
                    "- If `old_str` is not unique in the file, no replacement happens—include enough context.\n"
                    "- `new_str` contains the edited lines replacing `old_str`."
                ),
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to run.",
                            "enum": [
                                "view",
                                "create",
                                "str_replace",
                                "insert",
                                "undo_edit",
                            ],
                        },
                        "path": {
                            "type": "string",
                            "description": "Absolute path to file or directory, e.g. `/testbed/file.py` or `/testbed`.",
                        },
                        "file_text": {
                            "type": "string",
                            "description": "Required for `create`: the full contents of the new file.",
                        },
                        "old_str": {
                            "type": "string",
                            "description": "Required for `str_replace`: the exact string in `path` to replace.",
                        },
                        "new_str": {
                            "type": "string",
                            "description": (
                                "Optional for `str_replace` (replacement text). "
                                "Required for `insert` (the string to insert)."
                            ),
                        },
                        "insert_line": {
                            "type": "integer",
                            "description": "Required for `insert`: insert `new_str` AFTER this 1-based line number.",
                        },
                        "view_range": {
                            "type": "array",
                            "description": (
                                "Optional for `view` when `path` is a file. If omitted, shows the full file. "
                                "Provide `[start, end]` (1-based). Use `[start, -1]` to show from start to EOF."
                            ),
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                    },
                    "required": ["command", "path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "_state_anthropic",
                "description": "Internal helper to manage persistent editor state across tool calls.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit",
                "description": (
                    "Submits the current file. "
                    "Note: implementation may use internal flags (e.g., a hidden '-f') not exposed here."
                ),
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Runs the given command directly in bash.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to execute.",
                        }
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        },
    ]

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        formatted_messages = []
        gt_messages = _maybe_json_load(sample["messages"])
        for message in gt_messages:
            formatted_message = {
                "role": message["role"],
                "content": _normalize_message_content(message.get("content")),
            }
            if "tool_calls" in message and message["tool_calls"] is not None:
                formatted_message["tool_calls"] = []
                for tool_call in message["tool_calls"]:
                    formatted_message["tool_calls"].append(
                        {
                            "function": {
                                "arguments": tool_call["function"]["arguments"],
                                "name": tool_call["function"]["name"],
                            },
                            "id": f"chatcmpl-tool-{uuid.uuid4()}",
                            "type": "function",
                        }
                    )
            formatted_messages.append(formatted_message)

        return {
            "messages": formatted_messages,
            "tools": SWESmithTrajectoriesDataset.tools,
        }


def resolve_dataset_processor(
    dataset_name: str,
    dataset: Dataset,
    dataset_config_path: str | None = None,
) -> type[BaseDatasetProcessor]:
    """Resolve a processor class for a dataset name."""
    registered = DATASET_REGISTRY.get(dataset_name)
    if registered is not None:
        return registered

    file_configs = _get_file_dataset_configs(dataset_config_path)
    spec = resolve_processor_spec(dataset_name, dataset, file_configs)
    logger.info(
        "Using configurable processor '%s' for dataset '%s'",
        spec.processor,
        dataset_name,
    )
    return ConfiguredDatasetProcessor.for_spec(spec)


def load_category_batches(
    dataset_name,
    split,
    subset,
    tokenizer,
    model_max_length,
    batch_size,
    split_by_category,
    return_vllm_tokens_prompt,
    truncate,
    batches_per_category,
    dataset_config_path=None,
    map_num_proc: int | None = None,
):
    """Load a dataset and build tokenized calibration batches per category.

    Parallelism is controlled by ``map_num_proc`` (row mapping, per-category batch
    building, and tokenization). Defaults to ``max(1, cpu_count - 1)`` workers.
    """
    raw_ds = _load_raw_dataset(dataset_name, split, subset=subset)

    proc_cls = resolve_dataset_processor(
        dataset_name,
        raw_ds,
        dataset_config_path=dataset_config_path,
    )

    processor = proc_cls(
        dataset=raw_ds,
        tokenizer=tokenizer,
        max_input_len=model_max_length,
        split=split,
        split_by_category=split_by_category,
        return_vllm_tokens_prompt=return_vllm_tokens_prompt,
        truncate=truncate,
        batch_size=batch_size,
        map_num_proc=map_num_proc,
    )
    return processor.get_processed_dataset(
        batches_per_category=batches_per_category,
    )


DATASET_REGISTRY: dict[str, type[BaseDatasetProcessor]] = {
    "m-a-p/CodeFeedback-Filtered-Instruction": CodeFeedbackChatDataset,
    "allenai/tulu-3-sft-mixture": TuluSFTMixtureChatDataset,
    "cais/mmlu": MmluChatDataset,
    "ise-uiuc/Magicoder-Evol-Instruct-110K": MagicoderEvolInstructChatDataset,
    "allenai/c4": C4LMDataset,
    "theblackcat102/evol-codealpaca-v1": CodeAlpacaChatDataset,
    "euclaise/WritingPrompts_curated": WritingPromptsChatDataset,
    "allenai/tulu-3-sft-personas-math": PersonasMathChatDataset,
    "open-r1/Mixture-of-Thoughts": MixtureOfThoughtsDataset,
    "Salesforce/xlam-function-calling-60k": XLamFunctionCallingDataset,
    "SWE-bench/SWE-smith-trajectories": SWESmithTrajectoriesDataset,
}
