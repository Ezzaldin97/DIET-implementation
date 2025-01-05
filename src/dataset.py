import pandas as pd
import torch
from src.data_utils import align_labels_with_tokens, word_embeddings_provider
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

### create a dataset for the processed dataframe...
class DIETDataset(Dataset):
    def __init__(
        self, 
        df: pd.DataFrame,
        embed_model: str,
        encoder_model: str,
        device: str,
        max_length: int,
    ) -> None:
        super().__init__()
        self.df = df
        self.sentence_model = SentenceTransformer(embed_model, device = device)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)
        self.encoder_model = AutoModel.from_pretrained(
            encoder_model, 
            device_map = device, 
            output_hidden_states = True
        )
        self.device = device
        self.max_length = max_length
    def __len__(self) -> int:
        return self.df.shape[0]
    def __getitem__(self, idx: int):
        row = self.df.loc[idx, :]
        query = row["cleaned_text"]
        current_tags = row["tags_idx"]
        encoded = self.tokenizer(
            query,
            return_tensors="pt",
            return_special_tokens_mask=True,
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        ).to(self.device)
        input_id = encoded["input_ids"]
        atten_mask = encoded["attention_mask"]
        ### align tags with tokens after tokenization process..
        # 1- Get word IDs and special tokens mask
        word_ids = encoded.word_ids(batch_index=0)
        # 2- Align labels with tokens
        aligned_labels = align_labels_with_tokens(current_tags, word_ids, label_all_tokens=True)
        # 3- Pad the labels to the maximum length
        padded_labels = [-100] * self.max_length
        for i, label in enumerate(aligned_labels):
            padded_labels[i] = label
        ### keep CLS token from token embeddings of bert encoder model....
        ### make sure to remove it 
        tokens = {k: v for k, v in encoded.items() if k!="special_tokens_mask"}
        ### instead of using last hidden layer, use the last 4 hidden layers and sum...
        tokens_embs = word_embeddings_provider(
            tokens, self.encoder_model
        ).to(self.device)
        ### get the sentence embeddings of the query...
        sent_embs = self.sentence_model.encode(
            row["cleaned_text"], 
            convert_to_tensor=True,
            show_progress_bar=False,
            device=self.device
        )
        ### the intent label as tensor
        intent = torch.tensor([row["intent_idx"]]).to(self.device)
        ### the tags labels as tensor
        tags = torch.tensor(padded_labels).to(self.device)
        return input_id, atten_mask, tokens_embs, sent_embs, intent, tags