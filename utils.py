import numpy as np
import scipy.sparse as sp
import torch
import json
import pickle


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data():
    print('Loading dataset...') 
    wikirelations_file = './Data/raw_data/wiki_relations/comps_wikirelations.pkl'
    with open(wikirelations_file, 'rb') as f:
        adj = pickle.load(f)
    return adj
    
    
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

if __name__ == '__main__':
    load_data()