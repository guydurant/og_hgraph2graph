from email.utils import parsedate_tz
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import math
import random
import sys
import numpy as np
import argparse
from tqdm import tqdm
import itertools

from hgraph import *
import rdkit
from preprocess import tensorize

def remove_mols_not_in_vocab(smiles, vocab):
    vocab_atoms = []
    for i in vocab:
        try:
            atoms = rdkit.Chem.MolFromSmiles(i[0]).GetAtoms()
            for a in atoms:
                vocab_atoms.append(a.GetSymbol())
        except:
            print(i)
    vocab_atom_set = set(vocab_atoms)
    mols_to_remove = []
    for i in smiles[:]:
        mol_atoms = rdkit.Chem.MolFromSmiles(i).GetAtoms()
        for m in mol_atoms:
            if m.GetSymbol() not in vocab_atom_set:
                if i in smiles:
                    smiles.remove(i)
    for i in smiles[:]:
        smiles_vocab = get_vocab([i])
        for s in smiles_vocab:
            if s[1] not in list(zip(*vocab))[1]:
                if i in smiles:
                    smiles.remove(i)
    return smiles

def get_vocab(smiles):
    vocab = set()
    for s in smiles:
        s = s.strip("\r\n ")
        hmol = MolGraph(s)
        for node,attr in hmol.mol_tree.nodes(data=True):
            smiles = attr['smiles']
            vocab.add( attr['label'] )
            for i,s in attr['inter_label']:
                vocab.add( (smiles, s) )
    return vocab

def generate_latent_space_for_mol(model, smiles):
    molecule_tensor = tensorize(smiles, model.vocab)
    tree_tensors, graph_tensors = hgnn.make_cuda(molecule_tensor[1])
    root_vecs, tree_vecs, _, graph_vecs = model.encoder(
        tree_tensors, graph_tensors)
    vectors, _ = model.rsample(
        root_vecs, model.R_mean, model.R_var, perturb=False)
    return vectors


lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--vocab', required=True)
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--model', required=True)
parser.add_argument('--mols_to_sample', default=None)
parser.add_argument('--mode', default='noise')
parser.add_argument('--save_file', default='test.txt')
parser.add_argument('--noise_level', type=float, default=0.1)

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
    if not args.mols_to_sample:
        for _ in tqdm(range(args.nsample // args.batch_size)):
            smiles_list = model.sample(args.batch_size, greedy=True).cuda()
    else:
        with open(args.mols_to_sample) as f:
            given_smiles_list = [smi for smi in f.readlines()]
        given_smiles_list = remove_mols_not_in_vocab(given_smiles_list, vocab)
        selected_mol_vectors = generate_latent_space_for_mol(
            model, given_smiles_list).cuda()
        if args.mode == 'same':
            smiles_list = model.specific_sample(
                args.batch_size, selected_mol_vectors, args.mode, greedy=True)
            with open(args.save_file, 'a') as f:
                for _, smiles in enumerate(smiles_list):
                    f.write(smiles+'\n')
        else:
            for _ in tqdm(range(args.nsample // args.batch_size)):
                smiles_list = model.specific_sample(
                    args.batch_size, selected_mol_vectors, args.mode, greedy=True, noise=args.noise_level)
                with open(args.save_file, 'a') as f:
                    for _, smiles in enumerate(smiles_list):
                        f.write(smiles+'\n')
