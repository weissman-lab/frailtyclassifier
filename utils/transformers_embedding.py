from tensorflow import convert_to_tensor as tft
from tensorflow import stack as tfstack
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


class BCBEmbedder:
    def __init__(self, model_type):
        self.model_type = model_type
        if self.model_type == 'bioclinicalbert': #easier to write and no slashes
            self.model_type = 'emilyalsentzer/Bio_ClinicalBERT'
        self.model = AutoModel.from_pretrained(self.model_type,
                                               output_hidden_states=True, )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if self.device == "cpu":
            print("no CUDA!")
        print('init done')

    def get_cls(self, y):
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
        return cls

    def make_cls_array(self):
        out = tfstack([tft(self.get_cls(y)) for y in self.sentence_list])
        return out

    def __call__(self, sentence_list):
        self.sentence_list = sentence_list
        if self.model_type == 'emilyalsentzer/Bio_ClinicalBERT':
            out = self.make_cls_array()
        else:
            raise Exception("Need to implement HS averaging for roberta")
        return out

if __name__ == "__main__":
    pass
    # self = BCBEmbedder()
    # sentence_list = train_sent[:10]
    # y = sentence_list[0]