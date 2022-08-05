import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import math, random, sys
import numpy as np
import argparse
from tqdm import tqdm

from hgraph import *
import rdkit

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--vocab', required=True)
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--model', required=True)
parser.add_argument('--mols_to_sample', default=None)

parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--nsample', type=int, default=10000)

parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=250)
parser.add_argument('--embed_size', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--latent_size', type=int, default=32)
parser.add_argument('--depthT', type=int, default=15)
parser.add_argument('--depthG', type=int, default=15)
parser.add_argument('--diterT', type=int, default=1)
parser.add_argument('--diterG', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.0)

args = parser.parse_args()

vocab = [x.strip("\r\n ").split() for x in open(args.vocab)] 
args.vocab = PairVocab(vocab)

model = HierVAE(args).cuda()

model.load_state_dict(torch.load(args.model)[0])
model.eval()

torch.manual_seed(args.seed)
random.seed(args.seed)

with torch.no_grad():
    for _ in tqdm(range(args.nsample // args.batch_size)):
        if args.mols_to_sample:
            smiles_list = model.sample(args.batch_size, greedy=True)
        else:
            with open(args.mols_to_sample) as f:
                specific_mols = [smi[:-2] for smi in f.readlines()]
                print(specific_mols[:5])
            smiles_list = model.specific_sample(args.batch_size, specific_mols, greedy=True, )
        for _,smiles in enumerate(smiles_list):
                print(smiles)

