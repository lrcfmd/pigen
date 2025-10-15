#!/usr/bin/env python3
"""
Filter crystal structures using Statistical Proxy Potential (SPP) feasibility score.

SPP score quantifies structural feasibility based on interatomic distance distributions
derived from the Inorganic Crystal Structure Database (ICSD).

Reference:
    Schwalbe-Koda et al. (2019). A priori crystal structure prediction via 
    statistical proxy potentials. Nature Communications.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
from io import StringIO
import warnings

import numpy as np
import pandas as pd
from ase.io import read
from ase import Atoms
from pymatgen.core import Structure


# Constants
DEFAULT_SPP_FILE = 'SPP_collected.json'
DEFAULT_SPP_THRESHOLD = 0.362  # Standard threshold from literature
STRUCTURE_COLUMN_NAMES = ['structure', 'cif']  # Possible column names


class SPPCalculationError(Exception):
    """Raised when SPP calculation fails for a structure."""
    pass


def parse_cif_string(cif_string: str) -> Atoms:
    """
    Parse a CIF string into an ASE Atoms object.
    
    Args:
        cif_string: CIF format string
        
    Returns:
        ASE Atoms object
        
    Raises:
        ValueError: If CIF parsing fails
    """
    try:
        cif_file = StringIO(cif_string)
        atoms = read(cif_file, format='cif')
        return atoms
    except Exception as e:
        raise ValueError(f"Failed to parse CIF string: {e}") from e


def get_bin_value(x: float, bins: np.ndarray) -> float:
    """
    Find the bin value for a given distance.
    
    Args:
        x: Distance value
        bins: Array of bin edges (sorted)
        
    Returns:
        Bin value (upper edge of the bin containing x)
    """
    idx = np.digitize(x, bins)
    # Clip to valid range
    idx = min(idx, len(bins) - 1)
    return bins[idx]


def get_spp_error(distance: float, spp_dict: Dict[str, float]) -> float:
    """
    Get SPP error for a given interatomic distance.
    
    Args:
        distance: Interatomic distance in Angstroms
        spp_dict: Dictionary mapping distance bins to error values
        
    Returns:
        SPP error value for the given distance
    """
    bin_edges = np.array(sorted([float(k) for k in spp_dict.keys()]))
    
    # Handle out-of-range distances
    if distance > bin_edges[-1]:
        return spp_dict[str(bin_edges[-1])]
    elif distance < bin_edges[0]:
        return spp_dict[str(bin_edges[0])]
    else:
        bin_value = get_bin_value(distance, bin_edges)
        return spp_dict[str(bin_value)]


def calculate_spp_score(cif_string: str, spp_data: Dict[str, Dict[str, float]]) -> float:
    """
    Calculate SPP feasibility score for a crystal structure.
    
    The SPP score is the average error across all unique atom pairs,
    where error is derived from ICSD distance statistics.
    
    Args:
        cif_string: CIF format string of the structure
        spp_data: SPP database dictionary (element_pair -> distance_bin -> error)
        
    Returns:
        Average SPP error (lower is better, typically 0.0-1.0)
        Returns np.nan if calculation fails or data is missing
        
    Raises:
        SPPCalculationError: If calculation fails critically
    """
    try:
        atoms = parse_cif_string(cif_string)
        symbols = atoms.get_chemical_symbols()
        distances = atoms.get_all_distances(mic=True)  # Use minimum image convention
        
        total_error = 0.0
        pair_count = 0
        missing_pairs = set()
        
        # Calculate error for all unique pairs
        for i, symbol_i in enumerate(symbols):
            for j, symbol_j in enumerate(symbols):
                if i >= j:  # Skip self and double counting
                    continue
                
                # Create sorted pair key (e.g., "Fe-O")
                pair_key = '-'.join(sorted([symbol_i, symbol_j]))
                
                # Check if pair exists in SPP database
                if pair_key not in spp_data:
                    missing_pairs.add(pair_key)
                    continue
                
                distance = distances[i, j]
                error = get_spp_error(distance, spp_data[pair_key])
                
                total_error += error
                pair_count += 1
        
        # Handle missing data
        if missing_pairs:
            warnings.warn(
                f"Missing SPP data for pairs: {', '.join(sorted(missing_pairs))}"
            )
            return np.nan
        
        if pair_count == 0:
            warnings.warn("No valid atom pairs found in structure")
            return np.nan
        
        return round(total_error / pair_count, 4)
        
    except ValueError as e:
        warnings.warn(f"Structure parsing failed: {e}")
        return np.nan
    except Exception as e:
        raise SPPCalculationError(f"SPP calculation failed: {e}") from e


def extract_composition(cif_string: str) -> str:
    """
    Extract chemical composition from CIF string.
    
    Args:
        cif_string: CIF format string
        
    Returns:
        Chemical formula string
    """
    try:
        structure = Structure.from_str(cif_string, fmt='cif')
        return structure.formula
    except Exception as e:
        warnings.warn(f"Failed to extract composition: {e}")
        return "Unknown"


def find_structure_column(df: pd.DataFrame) -> str:
    """
    Find the column containing structure/CIF data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Name of the structure column
        
    Raises:
        ValueError: If no structure column is found
    """
    for col_name in STRUCTURE_COLUMN_NAMES:
        if col_name in df.columns:
            return col_name
    
    raise ValueError(
        f"No structure column found. Expected one of: {STRUCTURE_COLUMN_NAMES}"
    )


def process_structures(
    df: pd.DataFrame,
    spp_data: Dict[str, Dict[str, float]],
    structure_col: Optional[str] = None,
    spp_threshold: float = DEFAULT_SPP_THRESHOLD
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Calculate SPP scores and filter structures.
    
    Args:
        df: Input DataFrame with structure data
        spp_data: SPP database dictionary
        structure_col: Name of column containing CIF strings (auto-detect if None)
        spp_threshold: Maximum SPP error threshold for filtering
        
    Returns:
        Tuple of (filtered_dataframe, spp_scores_series)
        
    Raises:
        ValueError: If input validation fails
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    # Auto-detect structure column
    if structure_col is None:
        structure_col = find_structure_column(df)
        print(f"Using structure column: '{structure_col}'")
    elif structure_col not in df.columns:
        raise ValueError(f"Column '{structure_col}' not found in DataFrame")
    
    df_result = df.copy()
    
    # Add composition if not present
    if 'composition' not in df_result.columns:
        print("Extracting compositions...")
        df_result['composition'] = df_result[structure_col].apply(extract_composition)
    
    # Calculate SPP scores
    print(f"Calculating SPP scores for {len(df_result)} structures...")
    spp_scores = []
    
    for idx, cif_string in enumerate(df_result[structure_col]):
        if idx % 100 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(df_result)} structures...")
        
        try:
            score = calculate_spp_score(cif_string, spp_data)
            spp_scores.append(score)
        except Exception as e:
            warnings.warn(f"Failed at index {idx}: {e}")
            spp_scores.append(np.nan)
    
    df_result['spp_error'] = spp_scores
    
    # Drop structures with missing scores
    initial_count = len(df_result)
    df_result = df_result.dropna(subset=['spp_error'])
    dropped_na = initial_count - len(df_result)
    
    if dropped_na > 0:
        print(f"Dropped {dropped_na} structures with missing SPP scores")
    
    # Statistics before filtering
    if len(df_result) > 0:
        valid_scores = df_result['spp_error']
        print(f"\nSPP score statistics (n={len(valid_scores)}):")
        print(f"  Mean:   {valid_scores.mean():.4f}")
        print(f"  Median: {valid_scores.median():.4f}")
        print(f"  Std:    {valid_scores.std():.4f}")
        print(f"  Min:    {valid_scores.min():.4f}")
        print(f"  Max:    {valid_scores.max():.4f}")
        
        # Calculate validity percentage
        n_valid = (df_result['spp_error'] <= spp_threshold).sum()
        validity_pct = 100 * n_valid / len(df_result)
        print(f"\nStructures with SPP ≤ {spp_threshold}: {n_valid}/{len(df_result)} ({validity_pct:.1f}%)")
    
    # Filter by threshold
    df_filtered = df_result[df_result['spp_error'] <= spp_threshold]
    
    n_removed = len(df_result) - len(df_filtered)
    print(f"\nFiltering results:")
    print(f"  Retained: {len(df_filtered)} structures")
    print(f"  Removed:  {n_removed} structures ({100*n_removed/len(df_result):.1f}%)")
    
    return df_filtered, df_result['spp_error']


def load_spp_database(filepath: Path) -> Dict[str, Dict[str, float]]:
    """
    Load SPP database from JSON file.
    
    Args:
        filepath: Path to SPP JSON file
        
    Returns:
        SPP database dictionary
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not filepath.exists():
        raise FileNotFoundError(f"SPP database not found: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            spp_data = json.load(f)
        
        if not isinstance(spp_data, dict):
            raise ValueError("SPP database must be a dictionary")
        
        print(f"Loaded SPP database with {len(spp_data)} element pairs")
        return spp_data
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}") from e


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Filter crystal structures by SPP feasibility score",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Input CSV file containing structures'
    )
    parser.add_argument(
        '-s', '--spp-database',
        type=str,
        default=DEFAULT_SPP_FILE,
        help='Path to SPP database JSON file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output CSV file (default: <input>_spp.csv)'
    )
    parser.add_argument(
        '--structure-column',
        type=str,
        default=None,
        help='Name of column containing CIF strings (auto-detect if not specified)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=DEFAULT_SPP_THRESHOLD,
        help='Maximum SPP error threshold for filtering'
    )
    parser.add_argument(
        '--no-filter',
        action='store_true',
        help='Calculate SPP scores but do not filter'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Load SPP database
    try:
        spp_path = Path(args.spp_database)
        spp_data = load_spp_database(spp_path)
    except Exception as e:
        print(f"Error loading SPP database: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Load input data
    try:
        print(f"\nLoading data from {input_path}...")
        df = pd.read_csv(input_path, comment='#')
        print(f"Loaded {len(df)} rows")
    except Exception as e:
        print(f"Error loading CSV: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Process structures
    try:
        df_filtered, spp_scores = process_structures(
            df,
            spp_data,
            structure_col=args.structure_column,
            spp_threshold=args.threshold if not args.no_filter else float('inf')
        )
    except Exception as e:
        print(f"Error processing structures: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output filename
    if args.output:
        output_path = Path(args.output)
    else:
        stem = input_path.stem
        output_path = input_path.parent / f"{stem}_spp.csv"
    
    # Save results
    try:
        df_filtered.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    except Exception as e:
        print(f"Error saving output: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
