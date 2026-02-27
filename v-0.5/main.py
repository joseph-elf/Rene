import sys
import os

sys.path.append(os.path.abspath(".."))
from package import *

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from IPython.display import clear_output, display



sliding_windows = 0.

context_window = 256
batch_size = 64

d_emb = 512
nb_heads = 8
d_k = d_emb // nb_heads

mlp_multiplication=4

nb_layers = 8


name = input()


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)



tokenizer = SingleCharTokenizer()
tokens = torch.tensor(tokenizer.load_tokens("../tokens_sc.tok"))
print(tokens.shape)


vocab_size = tokenizer.vocab_size
print("vocab_size : " , vocab_size)

dataset = TextDataset(tokens,context_window=context_window,sliding_windows=sliding_windows)
print("dataset_size : " ,len(dataset))


loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers = 0
)







gpt = GPT(  vocab_size=vocab_size,
            context_window=context_window,
            d_emb=d_emb,
            nb_layers=nb_layers,
            nb_heads=nb_heads, 
            mlp_multiplication=mlp_multiplication
            ).to(device)
gpt.architecture()





engine = Engine(gpt,tokenizer,device)
print(len(iter(loader)))


engine.train(loader,1,5000,100,print_frequency=10)
engine.save_model("training_historic/final.w")

engine.save("training_historic/final.e")