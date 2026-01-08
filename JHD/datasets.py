import random

import torch
from torch.utils.data import Dataset

from utils import neg_sample


class PretrainDataset(Dataset):
    def __init__(self, args, user_seq, long_sequence):
        self.args = args
        self.user_seq = user_seq
        self.long_sequence = long_sequence
        self.max_len = args.max_seq_length # default = 50
        self.part_sequence = []
        self.split_sequence()

    # 학습 데이터의 길이를 max_len으로 통일하고, 학습용 prefix를 생성한다.
    # prefix 예시
    # [1, 3, 5, 7] => [1] [1,3] [1,3,5] 얘들이 prefix임
    def split_sequence(self):
        # seq = 개인의 시청 기록
        for seq in self.user_seq:
            # python_indexing
            # 유저의 시청 기록에서 마지막 2개를 제외한, 영화를 최대 50개까지 호출해옴
            input_ids = seq[-(self.max_len + 2) : -2]  # keeping same as train set
            
            # 유저 시청 기록을, 길이 1부터 최대 50까지 재생성(?)
            # prefix 생성 == 일종의 subset, 학습에 사용한다.
            for i in range(len(input_ids)):
                self.part_sequence.append(input_ids[: i + 1])

    def __len__(self):
        return len(self.part_sequence)

    def __getitem__(self, index):

        sequence = self.part_sequence[index]  # pos_items
        
        # Neg item sample for every masked item
        masked_item_sequence = []
        neg_items = []
        # Masked Item Prediction
        item_set = set(sequence)
        for item in sequence[:-1]:
            prob = random.random()
            # masking trigger 분기점
            # 발동시
            if prob < self.args.mask_p:
                # masked_item = masking 진행
                masked_item_sequence.append(self.args.mask_id)
                # neg_items = 등장한 item들 중에 없는 item을 채워넣음
                neg_items.append(neg_sample(item_set, self.args.item_size))
            # 비활성 = 아무일도 없음
            else:
                masked_item_sequence.append(item)
                neg_items.append(item)

        # add mask at the last position = 맞추라는건가?
        # neg_items에서 반드시 하나는 다르게 만들려고 하는걸수도.
        masked_item_sequence.append(self.args.mask_id)
        neg_items.append(neg_sample(item_set, self.args.item_size))

        # Segment Prediction
        if len(sequence) < 2:
            masked_segment_sequence = sequence
            pos_segment = sequence
            neg_segment = sequence
        else:
            sample_length = random.randint(1, len(sequence) // 2)
            start_id = random.randint(0, len(sequence) - sample_length)
            neg_start_id = random.randint(0, len(self.long_sequence) - sample_length)
            
            # pos_segment는 유저의 시청 기록에서 sample
            pos_segment = sequence[start_id : start_id + sample_length]
            
            # neg_segment는 전체 시청 기록에서 sample
            neg_segment = self.long_sequence[
                neg_start_id : neg_start_id + sample_length
            ]
            
            # masked_segment_sequence는 유저 시청 기록에서 sample length만큼 masking한다.
            masked_segment_sequence = (
                sequence[:start_id]
                + [self.args.mask_id] * sample_length
                + sequence[start_id + sample_length :]
            )
            # pos_segment는 masked_segment의 반전 술식
            # 둘의 masking 되지 않은 부분을 합치면 sequence가 된다.
            pos_segment = (
                [self.args.mask_id] * start_id
                + pos_segment
                + [self.args.mask_id] * (len(sequence) - (start_id + sample_length))
            )
            # neg_segment는 전체 시청 기록에서 뽑아왔다.
            # 그러니 masked_segment와 합쳐도 sequence가 나온다는 보장이 없음. (이걸로 pos와 neg의 loss를 계산하게 된다)
            neg_segment = (
                [self.args.mask_id] * start_id
                + neg_segment
                + [self.args.mask_id] * (len(sequence) - (start_id + sample_length))
            )

        assert len(masked_segment_sequence) == len(sequence)
        assert len(pos_segment) == len(sequence)
        assert len(neg_segment) == len(sequence)

        # padding sequence
        # sequence들의 길이가 max_len에 도달하지 못하면, 남은 자리를 padding으로 채워준다.
        # 근데 앞에서부터 채우네? -> 항상 마지막 자리 (ex) hidden_state[-1], logit[-1]이 출력의 자리가 되도록 고정하는 역할
        pad_len = self.max_len - len(sequence)
        masked_item_sequence = [0] * pad_len + masked_item_sequence
        pos_items = [0] * pad_len + sequence
        neg_items = [0] * pad_len + neg_items
        masked_segment_sequence = [0] * pad_len + masked_segment_sequence
        pos_segment = [0] * pad_len + pos_segment
        neg_segment = [0] * pad_len + neg_segment

        masked_item_sequence = masked_item_sequence[-self.max_len :]
        pos_items = pos_items[-self.max_len :]
        neg_items = neg_items[-self.max_len :]

        masked_segment_sequence = masked_segment_sequence[-self.max_len :]
        pos_segment = pos_segment[-self.max_len :]
        neg_segment = neg_segment[-self.max_len :]

        # Associated Attribute Prediction
        # Masked Attribute Prediction
        attributes = []
        for item in pos_items:
            # 장르의 갯수만큼 0으로 초기화
            attribute = [0] * self.args.attribute_size
            try:
                # item_index를 str을 통해 key로 변환
                # iten2attribute는 dict임
                now_attribute = self.args.item2attribute[str(item)]
                for a in now_attribute:
                    attribute[a] = 1
            except:
                pass
            attributes.append(attribute)

        assert len(attributes) == self.max_len
        assert len(masked_item_sequence) == self.max_len
        assert len(pos_items) == self.max_len
        assert len(neg_items) == self.max_len
        assert len(masked_segment_sequence) == self.max_len
        assert len(pos_segment) == self.max_len
        assert len(neg_segment) == self.max_len

        cur_tensors = (
            torch.tensor(attributes, dtype=torch.long),                 # 장르 one-hot-encoding
            torch.tensor(masked_item_sequence, dtype=torch.long),       # mask_p에 따라 masking된 시청기록
            torch.tensor(pos_items, dtype=torch.long),                  # 시청기록 + padding
            torch.tensor(neg_items, dtype=torch.long),                  # mask_p에 따라 생성된 본 적없는거 (아주 적은 확률로 시청기록과 딱 하나만 다를 수 있음)
            torch.tensor(masked_segment_sequence, dtype=torch.long),    # 시청기록에서 일정 구간이 가려짐.
            torch.tensor(pos_segment, dtype=torch.long),                # 가려진 구간. masked_segment와 pos_segment는 반전술식 관계
            torch.tensor(neg_segment, dtype=torch.long),                # 전체 시청 기록에서 가져옴. 아주 높은 확률로 pos_segment와 다름.
        )
        return cur_tensors


class SASRecDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type="train"):
        self.args = args
        self.user_seq = user_seq                      
        self.test_neg_items = test_neg_items # 무슨 말인지 모르겠네
        self.data_type = data_type
        self.max_len = args.max_seq_length


    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test", "submission"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]

        # submission [0, 1, 2, 3, 4, 5, 6]
        # answer None

        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]  # no use

        elif self.data_type == "valid":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        elif self.data_type == "test":
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]
        else:
            input_ids = items[:]
            target_pos = items[:]  # will not be used
            answer = []

        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        # seq의 길이를 max_len으로 맞춰준다.
        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len :]
        target_pos = target_pos[-self.max_len :]
        target_neg = target_neg[-self.max_len :]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )

        return cur_tensors

    def __len__(self):
        return len(self.user_seq)
    
    
class myBERTRecDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, submission=False):
        self.args = args                
        self.user_seq = user_seq                      # [num_users, watched_list] : list
        self.test_neg_items = test_neg_items    # test_neg 요구 여부
        self.max_len = args.max_seq_length      
        self.drop_max = 10
        self.submission = submission

    def __getitem__(self, index):
        
        user_id = index                         
        items = self.user_seq[user_id]
        
        # 시청 기록
        input_ids = items[:]
        
        # 시청 기록에서 최대 10개까지 무작의 drop ->> 나중에 함수로 짜서 utils로 옮기기

        # drop할 갯수
        # 시청 기록이 너무 짧으면 30%까지만 (+ 최소 1개, 반드시 하나는 치워짐)
        num_Drop = min(self.drop_max, max(1, int(len(input_ids) * 0.3))) 
        
        # 마지막 시청 기록은 반드시in drop
        Drop_positions = [self.max_len - 1]
        
        # 마지막을 제외한 모든 자리에 대해
        available_positions = list(range(len(input_ids) - 1))
        # 섞어서, (num_Drop - 1)개 만큼 고른다 
        random.shuffle(available_positions)
        Drop_positions.extend(available_positions[:num_Drop-1])
        
        # padding + drop
        masked_input_ids = [0]*self.max_len + [v for i, v in enumerate(input_ids) if i not in Drop_positions]
        masked_input_ids = masked_input_ids[-self.max_len:]

        # list : 정답 item_ids
        target_pos = [0]*self.drop_max + [v for i, v in enumerate(input_ids) if i in Drop_positions]
        target_pos = target_pos[-self.drop_max:]
                      
        # list : 오답 item_ids
        seq_set = set(items)
        target_neg = [0]*10*self.drop_max + [neg_sample(seq_set, self.args.item_size) for i in range(10*len(Drop_positions))]
        target_neg = target_neg[-10*self.drop_max:]

        if self.submission:
            masked_input_ids = [0]*self.max_len + items
            masked_input_ids = masked_input_ids[-self.max_len:]

        # neg_samples
        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),
                torch.tensor(masked_input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
                1
            )
        else:
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),
                torch.tensor(masked_input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                1
            )

        return cur_tensors

    def __len__(self):
        return len(self.user_seq)
    
class BERT4RecDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, submission=False):
        self.args = args                
        self.user_seq = user_seq                      # [num_users, watched_list] : list
        self.test_neg_items = test_neg_items    # test_neg 요구 여부
        self.max_len = args.max_seq_length      
        self.drop_max = 10
        self.submission = submission

    def __getitem__(self, index):
        
        user_id = index                         
        items = self.user_seq[user_id]
        
        # 시청 기록
        input_ids = items[:]
        
        # 모든 자리에서 args.mask_p에 따라 masking 적용
        masked_input_ids = []
        mask = []
        target_pos_list = []
        target_neg_list = []
        seq_set = set(items)
        
        for item in input_ids:
            prob = random.random()
            if prob < self.args.mask_p:
                masked_input_ids.append(self.args.mask_id)
                mask.append(1)
                target_pos_list.append(item)
                target_neg_list.append(neg_sample(seq_set, self.args.item_size))
            else:
                masked_input_ids.append(item)
                mask.append(0)
        
        # target_pos와 target_neg를 drop_max로 제한
        num_masked = len(target_pos_list)
        if num_masked > self.drop_max:
            # 랜덤하게 drop_max개 선택
            indices = random.sample(range(num_masked), self.drop_max)
            target_pos_list = [target_pos_list[i] for i in indices]
            target_neg_list = [target_neg_list[i] for i in indices]
            num_masked = self.drop_max
        
        target_pos = target_pos_list + [0] * (self.drop_max - num_masked)
        target_neg = [neg_sample(seq_set, self.args.item_size) for _ in range(10 * num_masked)]
        target_neg += [0] * (10 * self.drop_max - len(target_neg))
        target_neg = target_neg[:10 * self.drop_max]

        # padding
        pad_len = self.max_len - len(masked_input_ids)
        masked_input_ids = [0] * pad_len + masked_input_ids
        mask = [0] * pad_len + mask
        masked_input_ids = masked_input_ids[-self.max_len:]
        mask = mask[-self.max_len:]

        if self.submission:
            masked_input_ids = [0] * self.max_len + items
            masked_input_ids = masked_input_ids[-self.max_len:]
            mask = [0] * self.max_len  # no mask for submission

        # neg_samples
        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),
                torch.tensor(masked_input_ids, dtype=torch.long),
                torch.tensor(mask, dtype=torch.long),  # mask 추가
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
                1
            )
        else:
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),
                torch.tensor(masked_input_ids, dtype=torch.long),
                torch.tensor(mask, dtype=torch.long),  # mask 추가
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                1
            )

        return cur_tensors

    def __len__(self):
        return len(self.user_seq)
