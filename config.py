from pathlib import Path
import torch



def get_config():
    config = {
        "lang_src": "en",
        "lang_tgt": "it",
        "seq_len": 350,
        "d_model": 512,
        "batch_size": 8,
        "lr": 1e-4,
        "epochs": 20,
        "model_folder": "weights",
        "model_filename": "tmodel_",
        "preload": None,
        "tokenizer_path": "tokenizer_{0}.json",
        "experiment_name": "run/tmodel",
    }
    return config

def get_weights_file_path(config, epoch):
    model_folder = config["model_folder"]
    model_basename = config["model_filename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(model_folder) / model_filename)

