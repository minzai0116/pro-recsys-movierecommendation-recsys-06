from __future__ import annotations

"""
S3Rec pretraining dataset.

입력:
- user_seqs: user별 item sequence
- long_sequence: 전체 아이템 나열(세그먼트 negative sampling용)
- max_len: 패딩/절단 길이
- item_size: vocab size
- mask_id: 마스크 토큰 id
- attribute_size: attribute vocab size(멀티핫)
- item2attribute: item_id(str) -> list[attribute_id]
- mask_p: masked item 비율

출력(1 sample, 7 tensors):
- attributes: float/int [max_len, attribute_size] (multi-hot)
- masked_item_sequence: long [max_len]
- pos_items: long [max_len]
- neg_items: long [max_len]
- masked_segment_sequence: long [max_len]
- pos_segment: long [max_len]
- neg_segment: long [max_len]
"""

import random
from typing import Dict, List, Sequence, Set

import torch
from torch.utils.data import Dataset


def _neg_sample(item_set: Set[int], item_size: int) -> int:
    if item_size <= 2:
        return 1
    while True:
        t = random.randint(1, item_size - 1)
        if t not in item_set:
            return t


class S3RecPretrainDataset(Dataset):
    """
    Pretraining dataset adapted from the attached baseline's PretrainDataset.

    Produces 7 tensors:
      attributes, masked_item_sequence, pos_items, neg_items,
      masked_segment_sequence, pos_segment, neg_segment
    """

    def __init__(
        self,
        *,
        user_seqs: Sequence[Sequence[int]],
        long_sequence: Sequence[int],
        max_len: int,
        item_size: int,
        mask_id: int,
        attribute_size: int,
        item2attribute: Dict[str, List[int]] | None,
        mask_p: float = 0.2,
    ):
        self.user_seqs = [list(map(int, s)) for s in user_seqs]
        self.long_sequence = list(map(int, long_sequence))
        self.max_len = int(max_len)
        self.item_size = int(item_size)
        self.mask_id = int(mask_id)
        self.attribute_size = int(attribute_size)
        self.item2attribute = item2attribute or {}
        self.mask_p = float(mask_p)

        self.part_sequence: List[List[int]] = []
        self._split_sequence()

    def _split_sequence(self) -> None:
        for seq in self.user_seqs:
            input_ids = seq[-(self.max_len + 2) : -2]  # align with baseline
            for i in range(len(input_ids)):
                self.part_sequence.append(input_ids[: i + 1])

    def __len__(self) -> int:
        return len(self.part_sequence)

    def __getitem__(self, index: int):
        sequence = self.part_sequence[index]  # pos_items

        # Masked Item Prediction
        masked_item_sequence: List[int] = []
        neg_items: List[int] = []
        item_set = set(sequence)
        for item in sequence[:-1]:
            prob = random.random()
            if prob < self.mask_p:
                masked_item_sequence.append(self.mask_id)
                neg_items.append(_neg_sample(item_set, self.item_size))
            else:
                masked_item_sequence.append(item)
                neg_items.append(item)

        # add mask at the last position
        masked_item_sequence.append(self.mask_id)
        neg_items.append(_neg_sample(item_set, self.item_size))

        # Segment Prediction
        if len(sequence) < 2 or len(self.long_sequence) < 2:
            masked_segment_sequence = sequence
            pos_segment = sequence
            neg_segment = sequence
        else:
            sample_length = random.randint(1, max(1, len(sequence) // 2))
            start_id = random.randint(0, len(sequence) - sample_length)
            neg_start_id = random.randint(0, len(self.long_sequence) - sample_length)

            pos_seg = sequence[start_id : start_id + sample_length]
            neg_seg = self.long_sequence[neg_start_id : neg_start_id + sample_length]

            masked_segment_sequence = (
                sequence[:start_id]
                + [self.mask_id] * sample_length
                + sequence[start_id + sample_length :]
            )
            pos_segment = (
                [self.mask_id] * start_id
                + pos_seg
                + [self.mask_id] * (len(sequence) - (start_id + sample_length))
            )
            neg_segment = (
                [self.mask_id] * start_id
                + neg_seg
                + [self.mask_id] * (len(sequence) - (start_id + sample_length))
            )

        # padding sequence
        pad_len = self.max_len - len(sequence)
        masked_item_sequence = [0] * pad_len + masked_item_sequence
        pos_items = [0] * pad_len + sequence
        neg_items = [0] * pad_len + neg_items
        masked_segment_sequence = [0] * pad_len + list(masked_segment_sequence)
        pos_segment = [0] * pad_len + list(pos_segment)
        neg_segment = [0] * pad_len + list(neg_segment)

        masked_item_sequence = masked_item_sequence[-self.max_len :]
        pos_items = pos_items[-self.max_len :]
        neg_items = neg_items[-self.max_len :]
        masked_segment_sequence = masked_segment_sequence[-self.max_len :]
        pos_segment = pos_segment[-self.max_len :]
        neg_segment = neg_segment[-self.max_len :]

        # Associated Attribute Prediction (multi-hot)
        attributes: List[List[int]] = []
        for item in pos_items:
            attr = [0] * self.attribute_size
            try:
                now_attr = self.item2attribute.get(str(int(item)), [])
                for a in now_attr:
                    ai = int(a)
                    if 0 <= ai < self.attribute_size:
                        attr[ai] = 1
            except Exception:
                pass
            attributes.append(attr)

        return (
            torch.tensor(attributes, dtype=torch.long),
            torch.tensor(masked_item_sequence, dtype=torch.long),
            torch.tensor(pos_items, dtype=torch.long),
            torch.tensor(neg_items, dtype=torch.long),
            torch.tensor(masked_segment_sequence, dtype=torch.long),
            torch.tensor(pos_segment, dtype=torch.long),
            torch.tensor(neg_segment, dtype=torch.long),
        )


