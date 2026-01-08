import argparse
import yaml
import os
from tqdm import tqdm
import pandas as pd

import torch

from src.myBERT import TrainMIPDataset, EvalLastMaskDataset, Bert4RecModel
from src.myUtils import split_user_level_last2, get_user_logits_dict, ensemble_topN_topM_make_submission, verify_state_dict_equal
from src.utils import set_seed
from src.data_utils import (
    load_data, 
    create_user_item_matrix, 
    user_sequence_split,
    create_ground_truth
)
from src.ease import EASE
from src.metrics import recall_at_k, ndcg_at_k
from src.hyperparameter_tuning import optimize_lambda

from utils import (
    EarlyStopping,
    check_path,
    get_item2attribute_json,
    get_user_seqs,
    set_seed,
    windowed_seq,
    truncated_seq
)
    
def main():
    
    # ====== args ========
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--data_dir", default="./data/train/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Ml", type=str)

    # model args
    parser.add_argument("--model_name", default="Finetune_full", type=str)
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="hidden size of transformer model"
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=1, help="number of layers"
    )
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0.1,
        help="attention dropout p",
    )
    parser.add_argument(
        "--hidden_dropout_prob", type=float, default=0.3, help="hidden dropout p"
    )
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_seq_length", default=40, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="number of batch_size"
    )
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight_decay of adam"
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="adam first beta value"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="adam second beta value"
    )
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    parser.add_argument("--using_pretrain", action="store_true")
    
    parser.add_argument("--max_drop", type=int, default="10")
    parser.add_argument("--max_slide", type=int, default="5")
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + "train_ratings.csv"
    item2attribute_file = args.data_dir + args.data_name + "_item2attributes.json"

    user_seq, max_item, valid_rating_matrix, test_rating_matrix, _ = get_user_seqs(
        args.data_file
    )
    
    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.attribute_size = attribute_size + 1

    # save model args
    args_str = f"{args.model_name}-{args.data_name}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")
    print(str(args))

    args.item2attribute = item2attribute
    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)
    

    # Config 로드
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # ====== Params =======
     
    # ckpt = torch.load("./output/Stage_best.pt", map_location="cpu")
    ckpt = torch.load("./checkpoints/best.pt", map_location="cpu")
    sd = ckpt["model_state"]
    
    batch_size = 256
    item_num = args.item_size-2
    max_len = args.max_seq_length

    hidden_size=128
    num_layers=2
    num_heads=4
    dropout=0.1
        
    # ====== Model ======
    
    # ---- BERT ----
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Bert4RecModel(
        item_num=item_num,
        max_len=max_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        pad_id=0,
        mask_id=item_num + 1,
    )
    
    ckpt = torch.load("./output/Stage_best.pt", map_location="cpu")   # ✅ CPU에서 로드 (안 터짐)
    model.load_state_dict(ckpt["model_state"], strict=True)
    
    ok = verify_state_dict_equal(
        model,
        ckpt["model_state"],
        atol=0.0,
        rtol=0.0
    )
    if ok:
        print("Model verified")
    else:
        print("Error : Model collapse")
    
    print(" ========= Bert Load done ==========")
    # ---- EASE ----
    
    # -- data --
    data_path = cfg['data']['train_path']
    df = load_data(data_path)
 
    # -- load model --
    lambda_reg = cfg['model']['lambda_reg']
    MJ_EASE = EASE(lambda_reg=lambda_reg)
    
    print(" ========= Training EASE ==========")
    full_matrix, full_user_id_to_idx, full_item_id_to_idx = create_user_item_matrix(df)
    print(f"  전체 행렬 생성 완료: {full_matrix.shape}")
    
    # 아이템 ID 매핑 업데이트 (전체 데이터 기준)
    MJ_EASE.item_id_to_idx = full_item_id_to_idx
    MJ_EASE.idx_to_item_id = {idx: item_id for item_id, idx in full_item_id_to_idx.items()}
    
    MJ_EASE.fit(full_matrix, full_item_id_to_idx, verbose=True)
 
    print(" ====== EASE done ====== ")
    
    print(" ====== predict ====== ")

    sub_df = ensemble_topN_topM_make_submission(
        bert_model=model,
        ease_model=MJ_EASE,
        df=df,
        user_item_matrix=full_matrix,   
        user_id_to_idx=full_user_id_to_idx,   
        max_len=max_len,
        item_num=item_num,
        k=10,
        topN_ease=7,
        topM_bert=5,
        out_csv_path="topN_M2.csv",
    )
    
if __name__ == "__main__":
    main()