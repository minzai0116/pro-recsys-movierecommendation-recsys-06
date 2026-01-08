from __future__ import annotations

"""
SASRec-style dataset (seq_topn).

입력:
- user_seqs: Sequence[Sequence[int]] (user별 item sequence)
- max_len: int (sequence padding/truncation 길이)
- item_size: int (아이템 vocab size; 0은 padding)
- data_type: train|valid|test|submission

출력(1 sample):
- (user_id, input_ids, target_pos, target_neg, answer)
  - user_id: torch.long (== dataset index)
  - input_ids/target_pos/target_neg: torch.long [max_len]
  - answer: torch.long [1] (없으면 0으로 패딩)
"""

import random
from typing import List, Optional, Sequence, Set, Tuple

import torch
from torch.utils.data import Dataset


def _neg_sample(item_set: Set[int], item_size: int) -> int:
    """
    Sample a negative item id not in `item_set`.
    Assumes item ids are in [1, item_size-1] (0 is padding).
    """
    if item_size <= 2:
        return 1
    while True:
        t = random.randint(1, item_size - 1)
        if t not in item_set:
            return t


class SASRecDataset(Dataset):
    """
    Minimal SASRec-style dataset used by the attached baseline.

    data_type:
      - train: input_ids=items[:-3], target_pos=items[1:-2]
      - valid: input_ids=items[:-2], answer=[items[-2]]
      - test:  input_ids=items[:-1], answer=[items[-1]]
      - submission: input_ids=items[:], answer=[]
    """

    def __init__(
        self,
        *,
        user_seqs: Sequence[Sequence[int]],
        max_len: int,
        item_size: int,
        data_type: str = "train",
    ):
        self.user_seqs = [list(map(int, s)) for s in user_seqs]
        self.max_len = int(max_len)
        self.item_size = int(item_size)
        self.data_type = str(data_type)
        if self.data_type not in {"train", "valid", "test", "submission"}:
            raise ValueError(f"Unknown data_type={self.data_type}")

    def __len__(self) -> int:
        return len(self.user_seqs)

    def __getitem__(self, index: int):
        user_id = index
        items = self.user_seqs[index]

        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]
        elif self.data_type == "valid":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]] if len(items) >= 2 else []
        elif self.data_type == "test":
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]] if len(items) >= 1 else []
        else:
            input_ids = items[:]
            target_pos = items[:]
            answer = []

        # negative samples for each position in input_ids
        target_neg: List[int] = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(_neg_sample(seq_set, self.item_size))

        pad_len = self.max_len - len(input_ids)
        if pad_len < 0:
            # truncate from left
            input_ids = input_ids[-self.max_len :]
            target_pos = target_pos[-self.max_len :]
            target_neg = target_neg[-self.max_len :]
        else:
            input_ids = [0] * pad_len + input_ids
            target_pos = [0] * pad_len + target_pos
            target_neg = [0] * pad_len + target_neg

            input_ids = input_ids[-self.max_len :]
            target_pos = target_pos[-self.max_len :]
            target_neg = target_neg[-self.max_len :]

        # answer can be empty; keep fixed-shape tensor by padding with 0
        if not answer:
            answer = [0]

        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
        )


