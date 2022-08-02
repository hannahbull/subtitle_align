import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel, MPNetTokenizer, MPNetModel
import numpy as np

__all__ = ['VidTextModel']

class BertTextModel(torch.nn.Module):

    def __init__(self, multi_queries=0):
        super().__init__()

        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, text_inp):

        device = self.bert_model.pooler.dense.weight.device  # hack

        # text_emb_list = []
        # for i in range(self.multi_queries):
        #text_inp[i]

        tokenizer_out = self.bert_tokenizer(text_inp,
                                            padding=True,
                                            truncation=True,
                                            return_tensors="pt")

        txt_mask = tokenizer_out['attention_mask'].to(device)  # for padding

        bert_out = self.bert_model(tokenizer_out['input_ids'].to(device),
                                attention_mask=txt_mask)


        txt_feat = bert_out['last_hidden_state']
        txt_feat = txt_feat.permute([0, 2, 1])[..., None]  # b c t 1

        return txt_feat


class Word2VecTextModel(torch.nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()

        self.itos = np.load('word2vec/dict.npy')
        self.itos = np.concatenate([np.array(['<pad>']), self.itos], 0)

        self.stoi = dict([(ee, ii) for ii, ee in enumerate(self.itos)])

        if pretrained:
            word2vec_path = 'word2vec/word2vec.pth'
            self.word_embd = nn.Embedding.from_pretrained(
                torch.load(word2vec_path))
            self.word_embd.weight.requires_grad = False
        else:
            self.word_embd = nn.Embedding(len(self.stoi), 300)

    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = torch.zeros(size - len(tensor)).long()
            return torch.cat((tensor, zero), dim=0)

    def tokenize_str(self, st):
        words = st.split()
        toks = np.array(
            [self.stoi[ww] if ww in self.stoi else 0 for ww in words],
            dtype=np.int64)
        return torch.from_numpy(toks)

    def forward(self, text_inp):

        device = self.word_embd.weight.device  # hack

        tokens = [self.tokenize_str(tt) for tt in text_inp]

        maxdim = max([len(tens) for tens in tokens])
        tokens_padded = torch.stack(
            [self._zero_pad_tensor_token(tens, maxdim) for tens in tokens])

        txt_feat = self.word_embd(tokens_padded.to(device))

        txt_feat = txt_feat.permute([0, 2, 1])[..., None]  # b c t 1

        return txt_feat

class MPNetTextModel(torch.nn.Module):

    def __init__(self, multi_queries=0):
        super().__init__()

        self.mpnet_tokenizer = MPNetTokenizer.from_pretrained('microsoft/mpnet-base')
        self.mpnet_model = MPNetModel.from_pretrained('microsoft/mpnet-base')
        self.multi_queries = multi_queries

    def forward(self, text_inp):

        device = self.mpnet_model.pooler.dense.weight.device  # hack

        # text_emb_list = []
        # for i in range(self.multi_queries):
        #text_inp[i]

        tokenizer_out = self.mpnet_tokenizer(text_inp,
                                            padding=True,
                                            truncation=True,
                                            return_tensors="pt")

        txt_mask = tokenizer_out['attention_mask'].to(device)  # for padding

        bert_out = self.mpnet_model(tokenizer_out['input_ids'].to(device),
                                attention_mask=txt_mask)


        txt_feat = bert_out['last_hidden_state']
        txt_feat = txt_feat.permute([0, 2, 1])[..., None]  # b c t 1

        return txt_feat
