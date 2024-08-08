import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, lang_src, lang_tgt, seq_len) -> None:
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt   
        self.sos_token = torch.Tensor([self.tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.Tensor([self.tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.Tensor([self.tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        item = self.ds[idx]
        src = self.tokenizer_src.encode(item['translation'][self.lang_src]).ids
        tgt = self.tokenizer_tgt.encode(item['translation'][self.lang_tgt]).ids
        
        enc_num_pad = self.seq_len - len(src) - 2 # -2 because of the [SOS] and [EOS] tokens. 
        dec_num_pad = self.seq_len - len(tgt) - 2 # -2 because of the [SOS] and [EOS] tokens.
        
        if enc_num_pad < 0 or dec_num_pad < 0:
            raise ValueError("Sequence length is too long.")
        
        encoder_input = torch.cat([self.sos_token, torch.tensor(src,dtype=torch.int64), self.eos_token, torch.tensor([self.pad_token]*enc_num_pad, dtype=torch.int64)])
        decoder_input = torch.cat([self.sos_token, torch.tensor(tgt,dtype=torch.int64), torch.tensor([self.pad_token]*dec_num_pad, dtype=torch.int64)])
        label = torch.cat([torch.tensor(tgt,dtype=torch.int64), self.eos_token, torch.tensor([self.pad_token]*dec_num_pad, dtype=torch.int64)])
        
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, 1, seq_len) & (1, seq_len, seq_len)
            "label": label,
            "src_text": item['translation'][self.lang_src],
            "tgt_text": item['translation'][self.lang_tgt]
        }
        
        
def causal_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).type(torch.int)
    return mask.unsqueeze(0) # (1, size, size)