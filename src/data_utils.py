import torch
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple
import ast
import re
import os

INTENT_PATTEN = r"##\s*intent:(\w+)"
INTENT_PATTERN2 = r"\s*intent:(\w+)"
ENTITY_PATTERN = r"\(([^)]+)\)"

def extract_intents(path: str) -> List[str]:
    intent_lst = []
    if os.path.exists(path):
        with open(path, "r") as f:
            txt_data = f.read()
        sentences = txt_data.split("\n")
        for sentence in sentences:
            match = re.search(INTENT_PATTEN, sentence)
            if match:
                intent_name = match.group(1)
                intent_lst.append(intent_name)
        return list(set(intent_lst))
    else:
        raise FileNotFoundError(f"{path} not exists!!!!")

def extract_entities(path: str) -> List[str]:
    intent_lst = []
    if os.path.exists(path):
        with open(path, "r") as f:
            txt_data = f.read()
        sentences = txt_data.split("\n")
        for sentence in sentences:
            for token in sentence.split():
                match = re.search(ENTITY_PATTERN, token)
                if match:
                    intent_name = match.group(1)
                    intent_lst.append(intent_name)
        intent_lst = list(set(intent_lst))
        intent_lst.append("O")
        return intent_lst
    else:
        raise FileNotFoundError(f"{path} not exists!!!!")

def remove_tags(text: str) -> str:
    text = re.sub(ENTITY_PATTERN, "", text)
    text = text.replace("[", "").replace("]", "")
    return text

def remove_tags_from_tokens(lst: List[str]) -> List[str]:
    new_lst = []
    for token in lst:
        token = re.sub(ENTITY_PATTERN, "", token)
        token = token.replace("[", "").replace("]", "")
        new_lst.append(token)
    return new_lst

def create_model_df(
    path: str, 
    intent2idx: Dict[str, int], 
    entities2idx: Dict[str, int]
) -> pd.DataFrame:
    data = {"intent": [], "text": [], "tokens": [], "tags": []}
    if os.path.exists(path):
        with open(path, "r") as f:
            txt_data = f.read()
        intent_records = txt_data.split("##")
        intent_records = [rec for rec in intent_records if len(rec)>1]
        for rec in tqdm(intent_records, desc = "Convert Text Data to DataFrame"):
            sentences = txt_data.split("\n")
            for sentence in sentences:
                match = re.search(INTENT_PATTERN2, rec)
                if sentence.startswith("  -"):
                    data["intent"].append(match.group(1))
                    tokens = sentence.replace("  - ", "").split()
                    data["text"].append(sentence.replace("  - ", ""))
                    data["tokens"].append(tokens)
                    lst = []
                    for token in tokens:
                        match = re.search(ENTITY_PATTERN, token)
                        if match:
                            lst.append(match.group(1))
                        else:
                            lst.append("O")
                    data["tags"].append(lst)
        df = pd.DataFrame(data)
        df["cleaned_text"] = df["text"].apply(remove_tags)
        df["cleaned_tokens"] = df["tokens"].apply(remove_tags_from_tokens)
        df["intent_idx"] = df["intent"].apply(lambda x: intent2idx[x])
        df["tags_idx"] = df["tags"].apply(lambda x: [entities2idx[val] for val in x])
        return df
    else:
        raise FileNotFoundError(f"{path} not exists!!!!")

def align_labels_with_tokens(labels, word_ids, label_all_tokens):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        try:
            if word_id is None:
                # Special tokens like [CLS], [SEP], etc.
                new_labels.append(-100)  # Use -100 to ignore these tokens in loss computation
            elif word_id != current_word:
                # Start of a new word
                current_word = word_id
                new_labels.append(labels[word_id])
            else:
                # Same word as previous token
                label = labels[word_id]
                if label_all_tokens:
                    new_labels.append(label)
                else:
                    new_labels.append(-100)  # Use -100 to ignore these tokens in loss computation
        except IndexError:
            continue
    return new_labels

def word_embeddings_provider(encoded_input, encoder, layers=[-1, -2, -3, -4]) -> torch.Tensor:
    output = encoder(**encoded_input)
    return torch.stack([output.hidden_states[i] for i in layers]).sum(0).squeeze()

### create a custom collate for the data loader..
def custom_collate(data_batch: Tuple):
    input_ids = []
    mask = []
    tensors_wd = []
    tensors_st = []
    intents = []
    tags = []
    for sample in data_batch:
        ids, atten_mask, tk_embed, st_embed, intent_label, tags_labels = sample
        mask.append(atten_mask)
        input_ids.append(ids)
        tensors_st.append(st_embed)
        tensors_wd.append(tk_embed)
        intents.append(intent_label)
        tags.append(tags_labels)
    wd_embed = torch.stack(tensors_wd)
    input_ids_tensor = torch.stack(input_ids)
    mask_tensor = torch.stack(mask)
    st_embed = torch.stack(tensors_st)
    intents_tesnor = torch.stack(intents)
    tags_tesnor = torch.stack(tags)
    return {
        'input_ids': input_ids_tensor,
        'attention_mask': mask_tensor,
        'sentence_embeddings': st_embed,
        'word_embeddings': wd_embed,
        'intent': intents_tesnor,
        'tags': tags_tesnor,
    }

def str_tolist(val):
    return ast.literal_eval(val)