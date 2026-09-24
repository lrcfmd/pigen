import pandas as pd
import numpy as np
from pigen.eval.bond_table_ionic import element_charges
from pymatgen.core import Composition, Element
from itertools import product
from pymatgen.core import Structure
from tqdm import tqdm

def charges_range(step):
    ranges = {}
    for el, ch in element_charges.items():
        if len(ch) > 1 and Element(el).is_transition_metal:
            ranges[el] = [round(i,1) for i in np.arange(min(ch), max(ch) + step, step)]
        else:
            ranges[el] = [i for i in ch]
    return ranges

def check_charge_reported(df: pd.DataFrame,
                          ref_df: pd.DataFrame = None):
    """
    Utility function that takes as input a dataframe of STRcutures
    and filters it based on charge balance and if the compound is reported
    in a reference dataset
    """
    el_charges = charges_range(step=0.5)

    df['composition'] = df['cif'].apply(get_composition)
    df['formula']     = df['cif'].apply(get_formula)
    df['is_balanced'] = df['composition'].apply(is_balanced_multi, e_charges=el_charges)
    df = df[df['is_balanced']]

    if len(df) == 0:
        return df
    
    if ref_df is not None:
        assert "formula" in ref_df.columns, 'Reference dataframe must have a `formula` column'
        #filtering df based on reported formulas
        ref_form = set(ref_df['formula'])
        df       = df[~df['formula'].isin(ref_form)]
    
    df = df.drop(columns=['composition', 'formula', 'is_balanced', 'label'], axis=1)
    
    return df

def check_reported(df: pd.DataFrame):
    """
    Utility function that takes as input a dataframe of STRcutures
    and filters it based on if the compound is reported
    in a reference dataset
    """
    df['formula'] = df['cif'].apply(get_formula)
    if len(df) == 0:
        return df
    df = df.drop(columns=['formula', 'label'], axis=1)
    return df
    
def is_balanced(compound):
    composition = Composition(compound.strip()).as_dict()
    try:
        total_charge = sum([amount * element_charges[el] for el, amount in composition.items()])
    except:
        total_charge = 1
    if not total_charge:
        return True
    else:
        return False
    
def get_composition(cif: str):
    """
    Function that takes as input a CIF file and returns the composition
    """
    structure   =  Structure.from_str(cif, fmt='cif')
    composition = structure.composition
    return composition

def get_formula(cif: str):
    """
    Function that takes as input a CIF file and returns the formula
    """
    structure = Structure.from_str(cif, fmt='cif')
    formula   = structure.formula.replace(' ','')
    return formula

def is_balanced_multi(composition, e_charges):
    # composition = Composition(compound.strip()).as_dict()
    composition = composition.as_dict()
    nelem = len(composition)
    charges =[[] for i in range(nelem)]
    for i, it in enumerate(composition.items()):
        el, amnt = it
        if el not in e_charges:
            e_charges[el] = {1234: []}
        for ch in e_charges[el]:
                charges[i].append(amnt * ch)
    charge = [sum(comb)==0 for comb in product(*charges)]
    return any(charge)

# def is_balanced_multi(composition):
#     # composition = Composition(compound.strip()).as_dict()
#     comp_dict = composition.as_dict()
#     nelem     = len(composition)
#     charges   = [[] for i in range(nelem)]
#     for i, it in enumerate(comp_dict.items()):
#         el, amnt = it

#         if el not in element_charges:
#             element_charges[el] = {1234: []}

#         for ch in element_charges[el]:
#             charges[i].append(amnt * ch)

#     charge = [sum(comb)==0 for comb in product(*charges)]
    
#     return any(charge)

def is_balanced_multi_structure(strcuture: str, e_charges):
    structure = Structure.from_str(strcuture, fmt='cif')
    return is_balanced_multi(structure.composition, e_charges)

if __name__ == '__main__':
    df = pd.read_csv('data/full_data/all.csv')
    strcuture = df.iloc[0]['cif']

    el_charges = charges_range(step=0.5)
    is_balanced_multi_structure(strcuture, el_charges)

    # print('--- Computing formulas ---')
    # tqdm.pandas()
    # df['formula'] = df['cif'].progress_apply(get_formula)
    # df.to_csv('data/full_data/all_with_formula.csv', index=False)
