import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam

from utils import ndcg_k, recall_at_k


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        submission_dataloader,
        args,
    ):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.submission_dataloader = submission_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(
            self.model.parameters(),
            lr=self.args.lr,
            betas=betas,
            weight_decay=self.args.weight_decay,
        )

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        self.criterion = nn.BCELoss()

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch):
        return self.iteration(epoch, self.eval_dataloader, mode="valid")

    def test(self, epoch):
        return self.iteration(epoch, self.test_dataloader, mode="test")

    def submission(self, epoch):
        return self.iteration(epoch, self.submission_dataloader, mode="submission")

    def iteration(self, epoch, dataloader, mode="train"):
        raise NotImplementedError

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "RECALL@5": "{:.4f}".format(recall[0]),
            "NDCG@5": "{:.4f}".format(ndcg[0]),
            "RECALL@10": "{:.4f}".format(recall[1]),
            "NDCG@10": "{:.4f}".format(ndcg[1]),
        }
        print(post_fix)

        return [recall[0], ndcg[0], recall[1], ndcg[1]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (
            (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()
        )  # [batch*seq_len]
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def pairwise_cross_entropy(self, seq_out, pos_ids, neg_ids, tau=1.0):
        """_summary_

        Args:
            seq_out : 
                - 모델의 출력에 item_embedding을 내적한 tensor == logit
                - (_type_): torch.tensor.float 
                -  [Batch, item_sizes]
            
            input_ids : 
                - 유저의 시청 기록
                - (_type_): torch.tensor.long
                - [Batch, max_len]
            
            pos_ids : 
                - 시청 기록에서 랜덤으로 탈락된 영화 id
                - (_type_): torch.tentor.long 
                - [Batch, index_droped]
            
            neg_ids : 
                - negative sampling된 영화 ids
                - (_type_): torch.tentor.long 
                - [Batch, num_droped]
        """
        
        seq_out = seq_out / tau
        
        # 로짓 모아오기
        pos_s = seq_out.gather(1, pos_ids)      # [B, D]
        neg_s = seq_out.gather(1, neg_ids)      # [B, 10D]

        pos_mask = (pos_ids != 0).float()
        neg_mask = (neg_ids != 0).float()

        neg_s = neg_s.masked_fill(neg_mask == 0, float("-inf"))
        
        # Hard neg == topK neg 적용
        hard_k = min(self.args.max_seq_length, neg_s.size(1))
        neg_s, _ = torch.topk(neg_s, k=hard_k, dim=1)
        
        hard_neg_mask = torch.isfinite(neg_s).float()
        loss_neg = (F.softplus(neg_s) * hard_neg_mask).sum(1) / (hard_neg_mask.sum(1) + 1e-8)

        loss_pos = (F.softplus(-pos_s) * pos_mask).sum(1) / (pos_mask.sum(1) + 1e-8)
        
        loss = (loss_pos + loss_neg).mean()

        return loss

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class PretrainTrainer(Trainer):
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        submission_dataloader,
        args,
    ):
        super(PretrainTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            submission_dataloader,
            args,
        )

    def pretrain(self, epoch, pretrain_dataloader):

        desc = (
            f"AAP-{self.args.aap_weight}-"
            f"MIP-{self.args.mip_weight}-"
            f"MAP-{self.args.map_weight}-"
            f"SP-{self.args.sp_weight}"
        )

        pretrain_data_iter = tqdm(
            enumerate(pretrain_dataloader),
            desc=f"{self.args.model_name}-{self.args.data_name} Epoch:{epoch}",
            total=len(pretrain_dataloader),
            bar_format="{l_bar}{r_bar}",
        )

        self.model.train()
        aap_loss_avg = 0.0
        mip_loss_avg = 0.0
        map_loss_avg = 0.0
        sp_loss_avg = 0.0

        for i, batch in pretrain_data_iter:
            # 0. batch_data will be sent into the device(GPU or CPU)
            batch = tuple(t.to(self.device) for t in batch)
            (
                attributes,
                masked_item_sequence,
                pos_items,
                neg_items,
                masked_segment_sequence,
                pos_segment,
                neg_segment,
            ) = batch

            aap_loss, mip_loss, map_loss, sp_loss = self.model.pretrain(
                attributes,
                masked_item_sequence,
                pos_items,
                neg_items,
                masked_segment_sequence,
                pos_segment,
                neg_segment,
            )

            joint_loss = (
                self.args.aap_weight * aap_loss
                + self.args.mip_weight * mip_loss
                + self.args.map_weight * map_loss
                + self.args.sp_weight * sp_loss
            )

            self.optim.zero_grad()
            joint_loss.backward()
            self.optim.step()

            aap_loss_avg += aap_loss.item()
            mip_loss_avg += mip_loss.item()
            map_loss_avg += map_loss.item()
            sp_loss_avg += sp_loss.item()

        num = len(pretrain_data_iter) * self.args.pre_batch_size
        losses = {
            "epoch": epoch,
            "aap_loss_avg": aap_loss_avg / num,
            "mip_loss_avg": mip_loss_avg / num,
            "map_loss_avg": map_loss_avg / num,
            "sp_loss_avg": sp_loss_avg / num,
        }
        print(desc)
        print(str(losses))
        return losses


class FinetuneTrainer(Trainer):
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        submission_dataloader,
        args,
    ):
        super(FinetuneTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            submission_dataloader,
            args,
        )

    def iteration(self, epoch, dataloader, mode="train"):

        # Setting the tqdm progress bar

        rec_data_iter = tqdm(
            enumerate(dataloader),
            desc="Recommendation EP_%s:%d" % (mode, epoch),
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}",
        )
        
        # Self-supervised train : Train mode에서 학습은 일반적인 transformer와 같다.
        # input = X[:-2], label = X[1:-1]
        if mode == "train":
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                # 
                _, input_ids, target_pos, target_neg, _= batch
                # Binary cross_entropy
                sequence_output = self.model.finetune(input_ids)
                
                # Bert
                logit = self.predict_full(sequence_output[:,-1,:])
                loss = self.pairwise_cross_entropy(logit, target_pos, target_neg)
                
                # SASRec 
                # loss = self.cross_entropy(sequence_output, target_pos, target_neg)
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": "{:.4f}".format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": "{:.4f}".format(rec_cur_loss),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

        else:
            self.model.eval()

            pred_list = None
            answer_list = None
            for i, batch in rec_data_iter:
                
                '''Bert'''
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, target_pos, target_neg, _ = batch
                recommend_output = self.model.finetune(input_ids)

                logit = self.predict_full(recommend_output[:, -1, :])
                
                # 정답은 추천 안 하게 logit을 아주 작게 설정
                logit.scatter_(1, input_ids, float("-inf"))

                # torch.topk를 사용하여 top 10 index 추출
                _, batch_pred_list = torch.topk(logit, k=10, dim=1, largest=True)
                batch_pred_list = batch_pred_list.cpu().numpy()

                if i == 0:
                    pred_list = batch_pred_list
                    answer_list = target_pos.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(
                        answer_list, target_pos.cpu().data.numpy(), axis=0
                    )
                    
                '''SASRec'''
                # batch = tuple(t.to(self.device) for t in batch)
                # user_ids, input_ids, _, target_neg, answers = batch
                # recommend_output = self.model.finetune(input_ids)

                # recommend_output = recommend_output[:, -1, :]

                # rating_pred = self.predict_full(recommend_output)

                # rating_pred.scatter_(1, input_ids, float("-inf"))
                # _, batch_pred_list = torch.topk(rating_pred, k=10, dim=1, largest=True)
                # batch_pred_list = batch_pred_list.cpu().numpy()

                # if i == 0:
                #     pred_list = batch_pred_list
                #     answer_list = answers.cpu().data.numpy()
                # else:
                #     pred_list = np.append(pred_list, batch_pred_list, axis=0)
                #     answer_list = np.append(
                #         answer_list, answers.cpu().data.numpy(), axis=0
                #     )

            if mode == "submission":
                return pred_list
            else:
                return self.get_full_sort_score(epoch, answer_list, pred_list)
            
class BERT4RecTrainer(Trainer):
    def __init__(
            self,
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            submission_dataloader,
            args
        ):
        super(BERT4RecTrainer, self).__init__(
            model, 
            train_dataloader, 
            eval_dataloader, 
            test_dataloader, 
            submission_dataloader, 
            args
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def iteration(self, epoch, dataloader, mode="train"):
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_samples = 0

        for batch in tqdm(dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            user_ids, input_ids, target_pos, target_neg = batch[:4]

            seq_out = self.model(input_ids)  # [B, L, H]

            # For BERT4Rec, predict the next item
            logits = seq_out[:, :-1, :] @ self.model.item_embeddings.weight.T  # [B, L-1, item_size]
            labels = input_ids[:, 1:]  # [B, L-1]

            loss = self.criterion(logits.view(-1, self.args.item_size), labels.view(-1))

            if mode == "train":
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)

        return total_loss / total_samples
