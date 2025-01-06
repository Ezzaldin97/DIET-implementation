import torch
import pandas as pd
from torch.utils.data import RandomSampler, DataLoader
from transformers import get_linear_schedule_with_warmup
from torchmetrics.classification import Accuracy, F1Score
from tqdm import tqdm
from src.data_utils import custom_collate
from src.dataset import DIETDataset
from src.model import DIET
from src.train_utils import generate_class_weights_tensor
from typing import List, Dict, Any
from timeit import default_timer as timer 

class Trainer:
    def __init__(
        self, 
        train_df: pd.DataFrame, 
        valid_df: pd.DataFrame,
        device: str,
        sentence_embed_model: str,
        word_embed_model: str,
        max_seq_len: int,
        batch_size: int,
        intents: List[str],
        tags: List[str],
        sentence_emb_input_mapper_conf: Dict[str, Any],
        word_emb_input_mapper_conf: Dict[str, Any],
        sentence_emb_output_mapper_conf: Dict[str, Any],
        word_emb_output_mapper_conf: Dict[str, Any],
        transformer_conf: Dict[str, Any],
        epochs: int,
        lr: float,
        weight_decay: float,
        intent_weights: torch.Tensor,
        seed: float,
        w_e: float,
        w_i: float,
        logger,
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.train_ds = DIETDataset(
            train_df,
            sentence_embed_model,
            word_embed_model,
            device,
            max_seq_len,
        )
        self.valid_ds = DIETDataset(
            valid_df,
            sentence_embed_model,
            word_embed_model,
            device,
            max_seq_len,
        )
        self.train_dataloader = DataLoader(
            self.train_ds,
            batch_size = batch_size, 
            collate_fn = custom_collate,
            sampler = RandomSampler(self.train_ds)
        )
        self.valid_dataloader = DataLoader(
            self.valid_ds,
            batch_size = batch_size, 
            collate_fn = custom_collate,
            sampler = RandomSampler(self.valid_ds)
        )
        self.intents = intents
        self.tags = tags
        self.model = DIET(
            sentence_emb_input_mapper_conf,
            word_emb_input_mapper_conf,
            transformer_conf,
            sentence_emb_output_mapper_conf,
            word_emb_output_mapper_conf,
            len(tags),
        )
        self.epochs = epochs
        self.logger = logger
        self._total_steps = self.epochs * len(self.train_dataloader)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = lr,
            weight_decay=weight_decay
        )
        self.schedular = get_linear_schedule_with_warmup(
            self.optimizer,
            num_training_steps=0,
            num_warmup_steps=self._total_steps
        )
        self.intent_loss_fn = torch.nn.CrossEntropyLoss(
            weight=intent_weights,
            reduction="mean"
        )
        self.seed = seed
        self.w_e = w_e
        self.w_i = w_i

    @staticmethod
    def print_train_time(
        start: float, 
        end: float, 
        logger,
        device: str
    ):
        """Prints difference between start and end time.

        Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        logger: Logger instance
        device ([type], optional): Device that compute is running on. Defaults to None.

        Returns:
        float: time between start and end in seconds (higher is longer).
        """
        total_time = end - start
        logger.info(f"Train time on {device}: {total_time:.3f} seconds")
        return total_time

    def _intent_loss(self, intent_logits, true_intents):
        probs = torch.softmax(intent_logits, dim=-1)
        intent_loss = self.intent_loss_fn(probs, true_intents.squeeze())
        return intent_loss

    def _calculate_entities_weights(self, entities_words):
        accumulated_gt_entities = torch.tensor([], dtype = torch.long).to(self.device)
        not_padded_idx = torch.where(entities_words!=-100)
        entities_yb = entities_words[not_padded_idx]
        accumulated_gt_entities = torch.cat([accumulated_gt_entities, entities_yb]).to(self.device)
        updated_entities_weights = generate_class_weights_tensor(
            accumulated_gt_entities.cpu(), len(self.tags), self.device
        )
        return updated_entities_weights.to(self.device)

    def _ner_loss(
        self, 
        entities_logits, 
        crf_model, 
        crf_mask, 
        entities_weights, 
        entities_words
    ):
        weighted_logits = entities_logits*entities_weights
        entities_loss = -crf_model(
            weighted_logits, 
            entities_words, 
            mask=crf_mask, 
            reduction="mean"
        )
        return entities_loss
        
    def train(self) -> None:
        torch.manual_seed(self.seed)
        train_time_start_on_gpu = timer()
        intents_acc = F1Score(
            task="multiclass", num_classes=len(self.intents)
        ).to(self.device)
        entities_acc = F1Score(
            task="multiclass", num_classes=len(self.tags)
        ).to(self.device)
        for eidx, epoch in enumerate(tqdm(range(self.epochs), desc = "Epoch Processing")):
            train_loss, train_i_loss, train_e_loss = 0, 0, 0
            train_i_acc = 0
            train_e_acc = 0
            batch_counter = 0
            for idx, batch in enumerate(tqdm(self.train_dataloader, desc = "Batch Processing")):
                batch_counter+=self.batch_size
                self.model.train()
                intent_logits, entities_logits, entities_preds, crf_model = self.model(
                    batch["attention_mask"], 
                    batch["word_embeddings"], 
                    batch["sentence_embeddings"]
                )
                intent_preds = torch.argmax(intent_logits, -1)
                batch_i_loss=self._intent_loss(intent_logits, batch["intent"]).item()
                train_i_loss+=batch_i_loss 
                batch_i_acc = intents_acc(intent_preds, batch["intent"].squeeze(-1)).item()
                train_i_acc+=batch_i_acc
                e_weights = self._calculate_entities_weights(batch["tags"])
                batch_e_loss=self._ner_loss(
                    entities_logits, 
                    crf_model, 
                    batch["attention_mask"].squeeze()[:, 1:].type(torch.uint8), 
                    e_weights, 
                    batch["tags"][:, 1:]
                )
                train_e_loss+=batch_e_loss
                batch_train_loss=(self.w_e*batch_e_loss)+(self.w_i*batch_i_loss)
                train_loss+=batch_train_loss
                if idx%20 == 0:
                    self.logger.info(
                        f"BATCH: {idx}| INTENT loss: {batch_i_loss:.5f}, INTENT F1: {batch_i_acc:.5f}\n" 
                        f"BATCH: {idx}| NER loss: {batch_e_loss:.5f}, ACC NER F1: Not Implemented Yet\n"
                    )
                self.optimizer.zero_grad()
                batch_train_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.schedular.step()
            train_loss /= len(self.train_dataloader)
            train_i_acc /= len(self.train_dataloader)
            test_loss, test_i_loss, test_e_loss = 0, 0, 0
            test_i_acc = 0
            test_e_acc = 0
            self.model.eval()
            with torch.no_grad():
                for batch in tqdm(self.valid_dataloader, desc="Evaluation"):
                    intent_logits, entities_logits, entities_preds, crf_model = self.model(batch["attention_mask"], batch["word_embeddings"], batch["sentence_embeddings"])
                    intent_preds = torch.argmax(intent_logits, -1)
                    test_i_loss+=self._intent_loss(intent_logits, batch["intent"]).item()
                    test_i_acc+=intents_acc(intent_preds, batch["intent"].squeeze(-1)).item()
                    train_e_loss+=self._ner_loss(
                        entities_logits, 
                        crf_model, 
                        batch["attention_mask"].squeeze()[:, 1:].type(torch.uint8), 
                        e_weights, 
                        batch["tags"][:, 1:]
                    ).item()
                    test_loss=(self.w_e*batch_e_loss)+(self.w_i*batch_i_loss)
                test_loss /= len(self.valid_dataloader)
                test_i_acc /= len(self.valid_dataloader)
            self.logger.info(f"EPOCH: {eidx}| Train loss: {train_loss:.5f}, Train ACC: {train_i_acc:.5f} | Test loss: {test_loss:.5f}, Test ACC: {test_i_acc:.5f}\n")
            # Calculate training time        
        train_time_end_on_gpu = timer()
        total_train_time_model_0 = Trainer.print_train_time(
            start=train_time_start_on_gpu, 
            end=train_time_end_on_gpu,
            logger=self.logger,
            device=self.device
        )