import torch
import numpy as np
import pandas as pd

def calculate_class_weight(n_samplesj, n_classes, n_samples):
    wj = n_samples / (n_classes * n_samplesj)
    return wj

def generate_class_weights_tensor(gt_tensor, num_of_classes, device):
    train_ground_truth = pd.DataFrame(gt_tensor)
    if 0 in train_ground_truth.columns:
        train_ground_truth.rename(columns = {0: 'class'}, inplace=True)
    else:
        train_ground_truth.rename(columns = {"intent_idx": 'class'}, inplace=True)
    all_entities_df = pd.DataFrame(list(range(num_of_classes)))
    all_entities_df = all_entities_df.rename(columns = {0: 'class'})
    train_ground_truth['count'] = 0
    train_ground_truth = pd.merge(all_entities_df, train_ground_truth, on = 'class', how = 'left').fillna(0)
    n_samples = len(train_ground_truth)
    grouped_count = train_ground_truth.groupby(['class']).count()
    n_classes = len(grouped_count)
    grouped_count['class_weight'] = grouped_count['count'].apply(lambda x: calculate_class_weight(x, n_classes, n_samples))
    weights = torch.tensor(grouped_count['class_weight'].tolist(), dtype =  torch.float32).to(device)
    return weights