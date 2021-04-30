from tensorflow import convert_to_tensor as tft
from tensorflow import stack as tfstack
from transformers import AutoModel, AutoTokenizer, TFAutoModel
import tensorflow as tf
import torch
import numpy as np


class BCBEmbedder:
    def __init__(self, model_type):
        self.model_type = model_type
        if self.model_type == 'bioclinicalbert':  # easier to write and no slashes
            self.model_type = 'emilyalsentzer/Bio_ClinicalBERT'
            self.model = AutoModel.from_pretrained(self.model_type,
                                                   output_hidden_states=True, )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

        if self.model_type == 'roberta':  # easier to write and no slashes
            self.model_type = 'roberta-base'
            self.model = TFAutoModel.from_pretrained(self.model_type,
                                                     output_hidden_states=True, )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        print('init done')

    def get_cls(self, y):
        assert self.model_type == 'emilyalsentzer/Bio_ClinicalBERT'
        # y = "Why is this broken?"
        y = "[CLS] " + y + " [SEP]"
        tok_y = self.tokenizer.tokenize(y)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tok_y)
        segments_ids = [1] * len(tok_y)
        tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
        segments_tensors = torch.tensor([segments_ids]).to(self.device)
        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            out = outputs[2][-1][:, 0, :].cpu()
        cls = np.array(out).squeeze()
        del out
        torch.cuda.empty_cache()
        return cls

    # def get_avg_embedding(self, y):
    #     assert self.model_type == 'roberta-base'
    #     y = "<s>" + y + "</s>"
    #     tok_y = self.tokenizer.tokenize(y)
    #     indexed_tokens = self.tokenizer.convert_tokens_to_ids(tok_y)
    #     segments_ids = [1] * len(tok_y)
    #     tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
    #     segments_tensors = torch.tensor([segments_ids]).to(self.device)
    #     with torch.no_grad():
    #         outputs = self.model(tokens_tensor, segments_tensors)
    #         out = outputs[2][-1][:, 1:, :].cpu()
    #     out = np.mean(np.array(out), axis = 1)
    #     avg = np.array(out).squeeze()
    #     del out
    #     torch.cuda.empty_cache()
    #     return avg

    def get_avg_embedding(self, y):
        assert self.model_type == 'roberta-base'
        y = "<s>" + y + "</s>"
        tok_y = self.tokenizer.tokenize(y)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tok_y)
        segments_ids = [1] * len(tok_y)
        tokens_tensor = tf.expand_dims(tf.convert_to_tensor(indexed_tokens), 0)
        segments_tensor = tf.expand_dims(tf.convert_to_tensor(segments_ids), 0)
        outputs = self.model([tokens_tensor, segments_tensor])
        out = outputs[2][-1][:, 1:, :]
        out = np.mean(np.array(out), axis=1)
        avg = np.array(out).squeeze()
        return avg

    def make_array(self):
        if self.model_type == 'emilyalsentzer/Bio_ClinicalBERT':
            out = tfstack([tft(self.get_cls(y)) for y in self.sentence_list])
        if self.model_type == 'roberta-base':
            out = tfstack([tft(self.get_avg_embedding(y)) for y in self.sentence_list])
        return out

    def __call__(self, sentence_list):
        self.sentence_list = sentence_list
        out = self.make_array()
        return out


if __name__ == "__main__":
    pass
    # self = BCBEmbedder('bioclinicalbert')
    # sentence_list = train_sent[:10]
    # y = sentence_list[0]
