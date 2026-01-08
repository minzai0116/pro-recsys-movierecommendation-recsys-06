import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} created")


def neg_sample(item_set, item_size):
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):

            if score[i] > self.best_score[i] + self.delta:
                return False
        return True

    def __call__(self, score, model):

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0] * len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """Saves model when the performance is better."""
        if self.verbose:
            print(f"Better performance. Saving model to {self.checkpoint_path} ...")
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index).squeeze(dim)


def avg_pooling(x, dim):
    return x.sum(dim=dim) / x.size(dim)


def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]:  # 마지막 2개는 학습에 쓰겠다는건가?
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    # interaction matrix 생성
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_rating_matrix_submission(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:]:  # 다 씀. (얘는 학습용인가?)
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_submission_file(data_file, preds):

    rating_df = pd.read_csv(data_file)
    users = rating_df["user"].unique()

    result = []

    for index, items in enumerate(preds):
        for item in items:
            result.append((users[index], item))

    save_path = "output/submission.csv"
    print(f"Saved to {save_path}")
    pd.DataFrame(result, columns=["user", "item"]).to_csv(
        save_path, index=False
    )


def get_user_seqs(data_file):
    rating_df = pd.read_csv(data_file)
    lines = rating_df.groupby("user")["item"].apply(list)
    user_seq = []
    item_set = set()
    
    # line : 각 유저의 시청 기록
    for line in lines:

        items = line
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    num_users = len(lines)
    num_items = max_item + 2

    # 유저 시청 기록의 마지막 2개는 제외한, interaction matrix 생성
    valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
    test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    
    # 얘는 시청 기록 다 써서 interaction matrix 생성
    submission_rating_matrix = generate_rating_matrix_submission(
        user_seq, num_users, num_items
    )
    return (
        user_seq,
        max_item,
        valid_rating_matrix,
        test_rating_matrix,
        submission_rating_matrix,
    )


def get_user_seqs_long(data_file):
    '''
    ###########################
    input: 
        train_ratings.csv
            - columns: user, item
    
    output:
        user_seq = 유저 개인의 seq를 저장하고 있음
        long_seq = 모든 유저의 seq를 하나의 list로 저장 
    ###########################
    '''
    rating_df = pd.read_csv(data_file)
    
    # user의 item들을 list로 변환
    # 순서는 변하지 않음
    lines = rating_df.groupby("user")["item"].apply(list)
    
    user_seq = []  # 유저 개인의 시청 기록
    long_sequence = []  # 유저들의 시청 기록을 하나의 list로 통합 저장
    item_set = set()    # 등장한 모든 영화 기록
    
    # line = 유저의 시청 영화
    for line in lines:
        items = line
        long_sequence.extend(items)
        user_seq.append(items)
        item_set = item_set | set(items)
        
    # 데이터에 등장한 item id 중 최댓값
    # embedding layer의 item_size를 정하기 위해 사용하는듯
    max_item = max(item_set)

    return user_seq, max_item, long_sequence


def get_item2attribute_json(data_file):
    item2attribute = json.loads(open(data_file).readline())
    attribute_set = set()
    for item, attributes in item2attribute.items():
        attribute_set = attribute_set | set(attributes)
    attribute_size = max(attribute_set)
    return item2attribute, attribute_size


def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT / len(pred_list), NDCG / len(pred_list), MRR / len(pred_list)


def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)


def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users


def my_recall_at_k(actual, predicted, topk):
    # actual (torch.tensor.long) [Batch, 10] : 유저의 실제 시청 영화의 ids, 순서는 순위가 아님. 모두 동등
    # predicted (torch.tensor.long) : 모델의 마지막 출력 logit에 topk를 적용한 값.
    # topk (int) : topk 순위까지 recall 계산
    if torch.is_tensor(actual):
        actual = actual.detach().cpu().numpy()
    if torch.is_tensor(predicted):
        predicted = predicted.detach().cpu().numpy()

    sum_recall = 0.0
    true_users = 0
    num_users = len(predicted)
    for i in range(num_users):
        # 유저 target을 가져와서 set으로 만든다.
        act_set = set(actual[i])
        act_set.discard(0)
        pred_set = set(predicted[i][:topk])
        pred_set.discard(0)
        if len(act_set) != 0:
            # 정답 중에 몇 개 맞췄는지
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
            
    return sum_recall / true_users if true_users > 0 else 0.0


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        
        pred_set.discard(0)
        act_set.discard(0)
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum(
            [
                int(predicted[user_id][j] in set(actual[user_id])) / math.log(j + 2, 2)
                for j in range(topk)
            ]
        )
        res += dcg_k / idcg
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res

def split_user_level(
    user_seq,
    ratios=(0.8, 0.15, 0.05),
    seed=42
):
    assert abs(sum(ratios) - 1.0) < 1e-6

    num_users = len(user_seq)
    user_indices = list(range(num_users))

    random.seed(seed)
    random.shuffle(user_indices)

    n_train = int(num_users * ratios[0])
    n_valid = int(num_users * ratios[1])

    train_users = user_indices[:n_train]
    valid_users = user_indices[n_train:n_train + n_valid]
    test_users  = user_indices[n_train + n_valid:]

    print(len(train_users))
    print(len(valid_users))
    print(len(test_users))

    train_seq = [user_seq[u] for u in train_users]
    valid_seq = [user_seq[u] for u in valid_users]
    test_seq  = [user_seq[u] for u in test_users]

    return (
        train_seq, valid_seq, test_seq
    )

def windowed_seq(args, user_seq):
    windowed_seq = []
    for seq in user_seq:
        for index in range(max(1, min(args.max_slide,len(seq)+1-args.max_seq_length-args.max_drop))):
            windowed_seq.append(seq[index:min(len(seq), index+args.max_seq_length)])
    
    split_set = {}
    split_set["train"], split_set["valid"], split_set["test"] = split_user_level(windowed_seq)
    
    return split_set


def truncated_seq(args, user_seq):
    windowed_seq = []
    for seq in user_seq:
        windowed_seq.append(seq[:max(len(seq), args.max_seq_length+args.max_drop)])
    
    split_set = {}
    split_set["train"], split_set["valid"], split_set["test"] = split_user_level(windowed_seq)
    
    return split_set

def truncated_seq2(args, user_seq):
    windowed_seq = []
    for seq in user_seq:
        total_trunc = len(seq)//args.max_seq_length
        for i in range(total_trunc):
            start = -(1+(args.max_seq_length+args.max_drop)*(i+1))
            end = -(1+(args.max_seq_length+args.max_drop)*i)
            windowed_seq.append(seq[start:end])
    
    split_set = {}
    split_set["train"], split_set["valid"], split_set["test"] = split_user_level(windowed_seq)
    
    return split_set