import argparse
import pandas as pd
from pymatgen.core import Structure


def read_formulas(df):
    col = 'cif' if 'cif' in df.columns else 'structure'
    formulas = []
    for entry in df[col]:
        try:
            formulas.append(Structure.from_str(entry, fmt='cif').composition.reduced_formula)
        except:
            formulas.append(None)
    return formulas


def main(args):
    df     = pd.read_csv(args.inputfile)
    ref_df = pd.read_csv(args.ref)

    df['reduced_formula'] = read_formulas(df)
    gen_formulas = set(f for f in df['reduced_formula'] if f)
    if 'reduced_formula' in ref_df.columns:
        ref_formulas = set(ref_df['reduced_formula'].tolist())
    else:
        print('COULDNT READ ref["reduced_formula"]')
        ref_formulas = set(f for f in read_formulas(ref_df) if f)

    known = gen_formulas & ref_formulas
    new   = gen_formulas - ref_formulas

    print(f"Parsed:  {len(gen_formulas)} valid structures")
    print(f"Known:   {len(known)}")
    print(f"New:     {len(new)}")

    bf = df[df['reduced_formula'].isin(new)]
    print(f"New:     {bf.shape}")
    fname = args.inputfile.replace('.csv', 'newcomposition.csv')
    bf.to_csv(fname, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile')
    parser.add_argument('--ref', required=True)
    main(parser.parse_args())
