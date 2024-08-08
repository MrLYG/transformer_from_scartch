import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from dataset import BilingualDataset
from model import build_transformer
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from config import get_config,get_weights_file_path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang] # yields the translation of the language specified by lang.

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_path"].format(lang)) # lang is "en" or "de".  
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer
        
def get_dataloaders(config):
    ds = load_dataset("opus_books", f'{config["lang_src"]}-{config["lang_tgt"]}', split="train")
    tokenizer_src = get_or_build_tokenizer(config, ds, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds, config["lang_tgt"])
    
    train_ds_size = int(0.9 * len(ds))
    val_ds_size = len(ds) - train_ds_size
    train_ds, val_ds = random_split(ds, [train_ds_size, val_ds_size])
    
    train_ds = BilingualDataset(train_ds, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    val_ds = BilingualDataset(val_ds, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    
    max_len_src  = 0
    max_len_tgt = 0
    
    for item in ds:
        src_ids = tokenizer_src.encode(item['translation'][config["lang_src"]]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config["lang_tgt"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    print(f"Max length of source language: {max_len_src}")
    print(f"Max length of target language: {max_len_tgt}")
    
    train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False)
    return train_dl, val_dl, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['d_model'])
    return model

def train_model(config):
    
    # cuda, mps, cpu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("mps") if torch.backends.mps.is_available() else device
    print(f"Using device: {device}")
    
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    
    train_dl, val_dl, tokenizer_src, tokenizer_tgt = get_dataloaders(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    writer = SummaryWriter(config["experiment_name"])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    
    initial_epoch = 0
    global_step = 0
    
    if config["preload"] is not None:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model weights from {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch']
        global_step = state['global_step']
        optimizer.load_state_dict(state['optimizer'])
        global_step = state['global_step']
    
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1).to(device)
    
    for epoch in range(initial_epoch, config["epochs"]):
        model.train()
        batch_iter = tqdm(train_dl, desc=f"Epoch {epoch}")        
        for batch in batch_iter:
            encoder_input = batch['encoder_input'].to(device) # (batch_size, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch_size, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (batch_size, 1, seq_len, seq_len)
            label = batch['label'].to(device) # (batch_size, seq_len)
            
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch_size, seq_len, d_model)
            projection = model.project(decoder_output) # (batch_size, seq_len, vocab_tgt_len)
            label = batch['label'].to(device) # (batch_size, seq_len)
            
            loss = loss_fn(projection.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iter.set_postfix(loss=loss.item())
            
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()
            
    
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            global_step += 1
        model_filename = get_weights_file_path(config, f'{epoch:0.2d}')
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict()
        }, model_filename)
        
    
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)