import torch
import pandas as pd
import argparse
import pickle
from src.data_utils import (
    extract_entities, 
    extract_intents, 
    create_model_df,
    str_tolist
)
from src import Config
from src.train_utils import generate_class_weights_tensor
from src.trainer import Trainer
from dotenv import load_dotenv
from src.logger import ExecutorLogger
load_dotenv(".env")
import os

if not os.path.exists("assets/data/bin"):
    os.makedirs("assets/data/bin")
if not os.path.exists("assets/data/prepared"):
    os.makedirs("assets/data/prepared")

if __name__ == "__main__":
    logger = ExecutorLogger(os.getenv('LOGS_PATH'))
    device = "cuda:0" if torch.cuda.is_available() else "cpu" 
    conf = Config(config_path=os.getenv("CONFIG_PATH"))
    parser = argparse.ArgumentParser(description="DEIT model parameters")
    parser.add_argument(
        "--process",
        type=str, 
        choices=["prepare_data", "train", "inference"],
        required=True,
    )
    args = parser.parse_args()
    if args.process == "prepare_data":
        ### ATIS
        atis_intents = extract_intents(conf.config["data"]["raw"]["atis_train_data_path"])
        atis_entities = extract_entities(conf.config["data"]["raw"]["atis_train_data_path"])
        ### SNIPS
        snips_intents = extract_intents(conf.config["data"]["raw"]["snips_train_data_path"])
        snips_entities = extract_entities(conf.config["data"]["raw"]["snips_train_data_path"])
        ## intent encoding/decoding
        intents = atis_intents + snips_intents
        intent2idx = {intent: idx for idx, intent in enumerate(intents)}
        idx2intent = {idx: intent for intent, idx in intent2idx.items()}
        ## save intent encoder
        with open(conf.config["data"]["bin"]["intent2idx_path"], "wb") as file:
            pickle.dump(intent2idx, file)
        ### save intent decoder
        with open(conf.config["data"]["bin"]["idx2intent_path"], "wb") as file:
            pickle.dump(idx2intent, file)
        ### entities encoding/decoding
        tags = list(set(atis_entities + snips_entities))
        tag2idx = {tag: idx for idx, tag in enumerate(tags)}
        idx2tag = {idx: tag for tag, idx in tag2idx.items()}
        ## save entities encoder
        with open(conf.config["data"]["bin"]["tag2idx_path"], "wb") as file:
            pickle.dump(tag2idx, file)
        ### save entities decoder
        with open(conf.config["data"]["bin"]["idx2tag_path"], "wb") as file:
            pickle.dump(idx2tag, file)
        ### create dataframes from provided text data.....
        ## train data
        atis_train_df = create_model_df(
            conf.config["data"]["raw"]["atis_train_data_path"],
            intent2idx,
            tag2idx
        )
        snips_train_df = create_model_df(
            conf.config["data"]["raw"]["snips_train_data_path"],
            intent2idx,
            tag2idx
        )
        train_df = pd.concat([atis_train_df, snips_train_df]).reset_index().drop("index", axis = 1)
        train_df.to_csv(conf.config["data"]["prepared"]["train_data_path"], index=False)
        ### validation data
        atis_valid_df = create_model_df(
            conf.config["data"]["raw"]["atis_valid_data_path"],
            intent2idx,
            tag2idx
        )
        snips_valid_df = create_model_df(
            conf.config["data"]["raw"]["snips_valid_data_path"],
            intent2idx,
            tag2idx
        )
        valid_df = pd.concat([atis_valid_df, snips_valid_df]).reset_index().drop("index", axis = 1)
        valid_df.to_csv(conf.config["data"]["prepared"]["valid_data_path"], index=False)
    elif args.process == "train":
        logger.info(f"Current Device: {device}")
        train_df = pd.read_csv(conf.config["data"]["prepared"]["train_data_path"])
        train_df["tokens"] = train_df["tokens"].apply(str_tolist)
        train_df["cleaned_tokens"] = train_df["cleaned_tokens"].apply(str_tolist)
        train_df["tags"] = train_df["tags"].apply(str_tolist)
        train_df["tags_idx"] = train_df["tags_idx"].apply(str_tolist)
        valid_df = pd.read_csv(conf.config["data"]["prepared"]["valid_data_path"])
        valid_df["tokens"] = valid_df["tokens"].apply(str_tolist)
        valid_df["cleaned_tokens"] = valid_df["cleaned_tokens"].apply(str_tolist)
        valid_df["tags"] = valid_df["tags"].apply(str_tolist)
        valid_df["tags_idx"] = valid_df["tags_idx"].apply(str_tolist)
        with open(conf.config["data"]["bin"]["intent2idx_path"], "rb") as file:
            intents = list(pickle.load(file).keys())
        with open(conf.config["data"]["bin"]["tag2idx_path"], "rb") as file:
            tags = list(pickle.load(file).keys())
        intent_weights = generate_class_weights_tensor(train_df["intent_idx"], len(intents), device)
        trainer = Trainer(
            train_df=train_df,
            valid_df=valid_df,
            device=device,
            sentence_embed_model=conf.config["train"]["models"]["sentence_embedding_model"],
            word_embed_model=conf.config["train"]["models"]["word_embedding_model"],
            max_seq_len=conf.config["data"]["training_data_params"]["dataset"]["max_seq_len"],
            batch_size=conf.config["data"]["training_data_params"]["dataloader"]["batch_size"],
            intents=intents,
            tags=tags,
            sentence_emb_input_mapper_conf=conf.config["train"]["diet_model_layers_params"]["sentence_embedding_input_mapper"],
            word_emb_input_mapper_conf=conf.config["train"]["diet_model_layers_params"]["word_embedding_input_mapper"],
            sentence_emb_output_mapper_conf=conf.config["train"]["diet_model_layers_params"]["sentence_embedding_output_mapper"],
            word_emb_output_mapper_conf=conf.config["train"]["diet_model_layers_params"]["word_embedding_output_mapper"],
            transformer_conf=conf.config["train"]["diet_model_layers_params"]["transformer_layer"],
            epochs=conf.config["train"]["train_params"]["epochs"],
            lr=conf.config["train"]["train_params"]["lr"],
            weight_decay=conf.config["train"]["train_params"]["weight_decay"],
            intent_weights=intent_weights,
            seed=conf.config["train"]["random_seed"],
            w_e=conf.config["train"]["train_params"]["w_e"],
            w_i=conf.config["train"]["train_params"]["w_i"],
            logger=logger
        )
        trainer.train()
    else:
        raise NotImplementedError("Process Not Implemented")