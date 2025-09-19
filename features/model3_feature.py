import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Fragments, MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import logging

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def one_hot_encoding(x, permitted_list):

    if x not in permitted_list:
        x = permitted_list[-1]
    return [int(x == s) for s in permitted_list]

physchem_desc = [
    Descriptors.MolWt, Descriptors.ExactMolWt, Descriptors.MolLogP,
    Descriptors.NumHAcceptors, Descriptors.NumHDonors, Descriptors.TPSA,
    Descriptors.NumRotatableBonds, Descriptors.NumAromaticRings,
    Descriptors.NumAliphaticRings, Descriptors.NumSaturatedRings,
    Descriptors.HeavyAtomCount, Descriptors.NumValenceElectrons,
    Descriptors.NumHeteroatoms, Descriptors.RingCount, Lipinski.FractionCSP3
]
def get_physchem_only(mol):
    """提取物理化学描述符。"""
    try:
        feats = [desc(mol) for desc in physchem_desc]
        names = [f"PhysChem_{i}" for i in range(len(feats))]
        return np.array(feats, dtype=np.float32), names
    except:
        return None, []

def get_atom_features(atom):
    features = []
    features += one_hot_encoding(atom.GetAtomicNum(), list(range(1,31)) + ["Other"])
    features.extend([atom.GetAtomicNum(), atom.GetMass()])
    features.extend([
        atom.GetDegree(),
        atom.GetValence(getExplicit=False),
        atom.GetImplicitValence(),
        atom.GetTotalValence()
    ])
    features.extend([atom.GetFormalCharge(), atom.GetTotalNumHs()])
    features.extend([int(atom.IsInRing()), int(atom.GetIsAromatic())])
    features += one_hot_encoding(str(atom.GetHybridization()), ["S","SP","SP2","SP3","SP3D","SP3D2","OTHER"])
    features += one_hot_encoding(str(atom.GetChiralTag()),
                                 ['CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW','CHI_OTHER'])
    neighbor_Z = [n.GetAtomicNum() for n in atom.GetNeighbors()]
    features += [neighbor_Z[i] if i < len(neighbor_Z) else 0 for i in range(4)]
    return np.array(features, dtype=np.float32)

def get_atom_only(mol):
    try:
        atom_feats = np.mean([get_atom_features(atom) for atom in mol.GetAtoms()], axis=0)
        names = [f"AtomFeat_{i}" for i in range(len(atom_feats))]
        return atom_feats, names
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

def get_morgan1_only(mol): return get_morgan_only(mol, radius=1)
def get_morgan2_only(mol): return get_morgan_only(mol, radius=2)
def get_morgan3_only(mol): return get_morgan_only(mol, radius=3)
def get_morgan4_only(mol): return get_morgan_only(mol, radius=4)

def get_morgan_combined(mol):

    m1, n1 = get_morgan1_only(mol)
    m2, n2 = get_morgan2_only(mol)
    m3, n3 = get_morgan3_only(mol)
    m4, n4 = get_morgan4_only(mol)
    feats_list = [f for f in [m1, m2, m3, m4] if f is not None]
    names_list = n1 + n2 + n3 + n4
    if len(feats_list) == 0:
        return None, []
    return np.concatenate(feats_list), names_list

def get_all_features(mol):

    feats_list, names_list = [], []

    atom_feats, _ = get_atom_only(mol)
    if atom_feats is not None:
        feats_list.append(atom_feats)
        names_list += [f"AtomFeat_{i}" for i in range(len(atom_feats))]

    physchem_feats, _ = get_physchem_only(mol)
    if physchem_feats is not None:
        feats_list.append(physchem_feats)
        names_list += [f"PhysChem_{i}" for i in range(len(physchem_feats))]

    frag_feats, _ = get_fragment_only(mol)
    if frag_feats is not None:
        feats_list.append(frag_feats)
        names_list += [f"Frag_{name}" for name, _ in frag_funcs]

    maccs_feats, _ = get_maccs_only(mol)
    if maccs_feats is not None:
        feats_list.append(maccs_feats)
        names_list += [f"MACCS_bit{i}" for i in range(len(maccs_feats))]

    morgan_feats, morgan_names = get_morgan_combined(mol)
    if morgan_feats is not None:
        feats_list.append(morgan_feats)
        names_list += morgan_names

    if len(feats_list) == 0:
        return None, []

    all_feats = np.concatenate(feats_list)
    return all_feats, names_list
