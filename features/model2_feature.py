import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Fragments, MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import logging

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def get_maccs_only(mol):
    try:
        arr = np.array(MACCSkeys.GenMACCSKeys(mol))
        names = [f"MACCS_bit{i}" for i in range(len(arr))]
        return arr, names
    except:
        return None, []

def get_morgan_only(mol, radius=2, nBits=2048):
    try:
        mg = GetMorganGenerator(radius=radius, fpSize=nBits)
        arr = np.array(mg.GetFingerprint(mol))
        names = [f"Morgan{radius}_bit{i}" for i in range(nBits)]
        return arr, names
    except:
        return None, []

def get_all_features(mol):
    feats_list, names_list = [], []

    maccs_feats, _ = get_maccs_only(mol)
    if maccs_feats is not None:
        feats_list.append(maccs_feats)
        names_list += [f"MACCS_bit{i}" for i in range(len(maccs_feats))]

    morgan_feats, morgan_names = get_morgan_only(mol)
    if morgan_feats is not None:
        feats_list.append(morgan_feats)
        names_list += morgan_names

    if len(feats_list) == 0:
        return None, []

    all_feats = np.concatenate(feats_list)
    return all_feats, names_list