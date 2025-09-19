import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Fragments
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import logging

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

physchem_desc = [
    Descriptors.MolWt, Descriptors.ExactMolWt, Descriptors.MolLogP,
    Descriptors.NumHAcceptors, Descriptors.NumHDonors, Descriptors.TPSA,
    Descriptors.NumRotatableBonds, Descriptors.NumAromaticRings,
    Descriptors.NumAliphaticRings, Descriptors.NumSaturatedRings,
    Descriptors.HeavyAtomCount, Descriptors.NumValenceElectrons,
    Descriptors.NumHeteroatoms, Descriptors.RingCount, Lipinski.FractionCSP3
]
def get_physchem_only(mol):
    try:
        feats = [desc(mol) for desc in physchem_desc]
        names = [f"PhysChem_{i}" for i in range(len(feats))]
        return np.array(feats, dtype=np.float32), names
    except:
        return None, []

frag_funcs = [
    ("fr_Al_OH_noTert", Fragments.fr_Al_OH_noTert), ("fr_COO", Fragments.fr_COO),
    ("fr_C_S", Fragments.fr_C_S), ("fr_SH", Fragments.fr_SH),
    ("fr_alkyl_carbamate", Fragments.fr_alkyl_carbamate), ("fr_azo", Fragments.fr_azo),
    ("fr_nitro", Fragments.fr_nitro), ("fr_halogen", Fragments.fr_halogen),
    ("fr_amide", Fragments.fr_amide), ("fr_ester", Fragments.fr_ester), ("fr_ketone", Fragments.fr_ketone)
]

def get_fragment_only(mol):
    try:
        feats = [func(mol) for _, func in frag_funcs]
        names = [f"Frag_{name}" for name, _ in frag_funcs]
        return np.array(feats, dtype=np.float32), names
    except:
        return None, []
    
def get_morgan_only(mol, radius=2, nBits=2048):
    try:
        generator = GetMorganGenerator(radius=radius, fpSize=nBits)
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((1,), dtype=np.int32)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        names = [f"Morgan{radius}_bit{i}" for i in range(nBits)]
        return arr, names
    except Exception as e:
        logging.error(f"Morgan指纹生成失败: {e}")
        return None, []

def get_morgan2_only(mol): return get_morgan_only(mol, radius=2)
def get_morgan3_only(mol): return get_morgan_only(mol, radius=3)
def get_morgan4_only(mol): return get_morgan_only(mol, radius=4)

def get_morgan_combined(mol):
    try:
        m2, n2 = get_morgan2_only(mol)
        m3, n3 = get_morgan3_only(mol)
        m4, n4 = get_morgan4_only(mol)
        feats = np.concatenate([m2, m3, m4])
        names = n2 + n3 + n4
        return feats, names
    except Exception as e:
        logging.error(f"Morgan指纹组合失败: {e}")
        return None, []

def get_all_features(mol):
    feats_list, names_list = [], []

    physchem_feats, _ = get_physchem_only(mol)
    if physchem_feats is not None:
        feats_list.append(physchem_feats)
        names_list += [f"PhysChem_{i}" for i in range(len(physchem_feats))]

    frag_feats, _ = get_fragment_only(mol)
    if frag_feats is not None:
        feats_list.append(frag_feats)
        names_list += [f"Frag_{name}" for name, _ in frag_funcs]

    morgan_feats, morgan_names = get_morgan_combined(mol)
    if morgan_feats is not None:
        feats_list.append(morgan_feats)
        names_list += morgan_names

    if len(feats_list) == 0:
        return None, []

    all_feats = np.concatenate(feats_list)
    return all_feats, names_list
