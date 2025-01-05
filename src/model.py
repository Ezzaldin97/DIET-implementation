import torch
from torch import nn
import torch.nn.functional as F
from torchcrf import CRF
from typing import Dict, Any
import math

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len, dropout):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    def forward(self, X):
        output = X + self.pe[:, : X.size(1)].requires_grad_(False)
        return self.dropout(output)

class MapperNN(nn.Module):
    def __init__(self, config_dict: Dict[str, Any]) -> None:
        super(MapperNN, self).__init__()
        self.input_size = config_dict["input_size"]
        self.output_size = config_dict["output_size"]
        self.list_hidden_layers_sizes = config_dict["list_hidden_layers_sizes"]
        self.dropout_prob = config_dict["dropout_prob"]
        self.last_layer_activation_func = config_dict["last_layer_activation_func"]
        self.batch_norm_flg = config_dict["batch_norm_flg"]
        if self.batch_norm_flg:
            batch_norm_layers = [nn.BatchNorm1d(self.input_size, track_running_stats=False)]
            for layer_size in self.list_hidden_layers_sizes:
                batch_norm_layers.append(nn.BatchNorm1d(layer_size, track_running_stats=False))
            batch_norm_layers.append(nn.BatchNorm1d(self.output_size, track_running_stats=False))
            self.batch_norm_layers = nn.ModuleList(batch_norm_layers)
        if len(self.list_hidden_layers_sizes)>0:
            fc_layers = [nn.Linear(self.input_size, self.list_hidden_layers_sizes[0])]
            for layer_size in self.list_hidden_layers_sizes[1:]:
                fc_layers.append(nn.Linear(self.list_hidden_layers_sizes[0], layer_size))
            fc_layers.append(nn.Linear(self.list_hidden_layers_sizes[-1], self.output_size))
        else:
            fc_layers = [nn.Linear(self.input_size, self.output_size)]
        self.fc_layers = nn.ModuleList(fc_layers)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.leaky_relu = nn.LeakyReLU(0.1)
    def forward(self, X):
        if self.batch_norm_flg == False:
            output = self.fc_layers[0](X)
            if len(self.fc_layers)>1:
                for idx, layer in enumerate(self.fc_layers[1:]):
                    output = self.leaky_relu(output)
                    output = self.dropout(output)
                    output = layer(output)
        else:
            output = self.batch_norm_layers[0](X)
            output = self.fc_layers[0](output)
            if len(self.fc_layers)>1:
                for idx, layer in enumerate(self.fc_layers[1:]):
                    output = self.batch_norm_layers[idx](output)
                    output = self.leaky_relu(output)
                    output = self.dropout(output)
                    output = layer(output)
        if self.last_layer_activation_func == "softmax":
            output = F.softmax(output, dim=1)
        else:
            output = F.leaky_relu(output, 0.1)
        return output

class DIET(nn.Module):
    def __init__(
        self, 
        input_sentence_config, 
        input_word_config, 
        trans_config,
        output_sentence_config,
        output_word_config,
        n_tags,
    ) -> None:
        super(DIET, self).__init__()
        self.sentence_input_nn = MapperNN(input_sentence_config)
        self.word_input_nn = MapperNN(input_word_config)
        self.sentence_output_nn = MapperNN(output_sentence_config)
        self.word_output_nn = MapperNN(output_word_config)
        self.pe = PositionalEncoder(
            trans_config['d_model'], 
            trans_config['max_seq_len_positional_enc'],
            trans_config['position_enc_dropout']
        )
        self.transformer = nn.Transformer(
            d_model = trans_config['d_model'],
            nhead = trans_config['n_heads'],
            num_encoder_layers = trans_config['num_encoder_layers'],
            num_decoder_layers = trans_config['num_decoder_layers'],
            dim_feedforward = trans_config['dim_feedforward'],
            dropout = trans_config['dropout'],
            batch_first = True
        )
        self.crf = CRF(n_tags, batch_first = True)
    def forward(self, batch_mask, word_embed, sentence_embed):
        words_input = self.word_input_nn(word_embed)
        sentence_input = self.sentence_input_nn(sentence_embed)
        ### now concat mapping inputs before deliver to transformer...
        merged_inputs = torch.concat(
            (sentence_input.unsqueeze(1), words_input[:, 1:, :]), 
            dim=1
        )
        ### pass it to the positional encoder....
        merged_inputs = self.pe(merged_inputs)
        ### pass to the transformer layer...
        transformer_output = self.transformer.encoder(
            merged_inputs,
            src_key_padding_mask = batch_mask.squeeze(1).type(torch.bool)
        )
        entities_transformer_output = transformer_output[:, 1:, :]
        intent_transformer_output = transformer_output[:, 0, :]##[CLS]
        intent_logits = self.sentence_output_nn(intent_transformer_output)
        entities_mapping_output = self.word_output_nn(entities_transformer_output)
        entities_predictions = self.crf.decode(
            entities_mapping_output, 
            mask = batch_mask.squeeze(1)[:, 1:].type(torch.uint8)
        )
        return intent_logits, entities_mapping_output, entities_predictions, self.crf