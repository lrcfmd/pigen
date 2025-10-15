#!/usr/bin/env python3
"""
Filter crystal structures by atomic packing compactness.

This script calculates the compactness (packing fraction) of crystal structures
and filters them based on physically reasonable thresholds.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple
import warnings

import pandas as pd
import numpy as np
from pymatgen.core import Structure, Composition, Element


# Constants
DEFAULT_COMPACTNESS_MIN = 0.55
DEFAULT_COMPACTNESS_MAX = 0.85
SPHERE_VOLUME_FACTOR = 4.0 / 3.0 * np.pi


class CompactnessCalculationError(Exception):
    """Raised when compactness calculation fails for a structure."""
    pass


def calculate_atomic_volume(element: str) -> float:
    """
    Calculate the volume of a single atom based on its atomic radius.
    
    Args:
        element: Element symbol (e.g., 'Fe', 'O')
        
    Returns:
        Atomic volume in Angstrom^3. Returns 0.0 if radius is unavailable.
        
    Note:
        Uses atomic radius from pymatgen. Missing radii return 0.0 volume.
    """
    try:
        radius = Element(element).atomic_radius
        if radius is None:
            warnings.warn(f"Atomic radius not available for {element}, using 0.0")
            return 0.0
        return SPHERE_VOLUME_FACTOR * (radius ** 3)
    except Exception as e:
        warnings.warn(f"Error getting atomic volume for {element}: {e}")
        return 0.0


def calculate_compactness(structure: Structure) -> float:
    """
    Calculate the atomic packing compactness of a crystal structure.
    
    Compactness is defined as the ratio of total atomic volumes to unit cell volume:
        compactness = (sum of atomic volumes) / (unit cell volume)
    
    Args:
        structure: Pymatgen Structure object
        
    Returns:
        Compactness value (typically 0.0 to 1.0)
        
    Raises:
        CompactnessCalculationError: If calculation fails
    """
    try:
        composition = Composition(structure.formula).as_dict()
        total_atomic_volume = sum(
            calculate_atomic_volume(element) * count
            for element, count in composition.items()
        )
        
        if structure.volume <= 0:
            raise CompactnessCalculationError(
                f"Invalid unit cell volume: {structure.volume}"
            )
        
        return total_atomic_volume / structure.volume
        
    except Exception as e:
        raise CompactnessCalculationError(
            f"Failed to calculate compactness: {e}"
        ) from e


def process_structures(
    df: pd.DataFrame,
    cif_column: str = 'cif',
    min_compactness: float = DEFAULT_COMPACTNESS_MIN,
    max_compactness: float = DEFAULT_COMPACTNESS_MAX
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Process structures and filter by compactness.
    
    Args:
        df: DataFrame containing structure data
        cif_column: Name of column containing CIF strings
        min_compactness: Minimum compactness threshold
        max_compactness: Maximum compactness threshold
        
    Returns:
        Tuple of (filtered_dataframe, compactness_series)
        
    Raises:
        ValueError: If input validation fails
    """
    if cif_column not in df.columns:
        raise ValueError(f"Column '{cif_column}' not found in DataFrame")
    
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    if not 0 <= min_compactness <= 1:
        raise ValueError(f"min_compactness must be in [0, 1], got {min_compactness}")
    
    if not 0 <= max_compactness <= 1:
        raise ValueError(f"max_compactness must be in [0, 1], got {max_compactness}")
    
    if min_compactness >= max_compactness:
        raise ValueError(
            f"min_compactness ({min_compactness}) must be < max_compactness ({max_compactness})"
        )
    
    print(f"Processing {len(df)} structures...")
    
    compactness_values = []
    failed_indices = []
    
    for idx, cif_string in enumerate(df[cif_column]):
        try:
            structure = Structure.from_str(cif_string, fmt='cif')
            compactness = calculate_compactness(structure)
            compactness_values.append(compactness)
        except Exception as e:
            warnings.warn(f"Failed to process structure at index {idx}: {e}")
            compactness_values.append(np.nan)
            failed_indices.append(idx)
    
    if failed_indices:
        print(f"Warning: Failed to process {len(failed_indices)} structures")
    
    # Add compactness column
    df_result = df.copy()
    df_result['compactness'] = compactness_values
    
    # Remove failed structures
    df_result = df_result.dropna(subset=['compactness'])
    
    # Statistics before filtering
    valid_compactness = df_result['compactness']
    print(f"\nCompactness statistics (n={len(valid_compactness)}):")
    print(f"  Mean: {valid_compactness.mean():.4f}")
    print(f"  Std:  {valid_compactness.std():.4f}")
    print(f"  Min:  {valid_compactness.min():.4f}")
    print(f"  Max:  {valid_compactness.max():.4f}")
    
    # Filter by compactness
    df_filtered = df_result[
        (df_result['compactness'] >= min_compactness) &
        (df_result['compactness'] <= max_compactness)
    ]
    
    n_removed = len(df_result) - len(df_filtered)
    print(f"\nFiltering with range [{min_compactness}, {max_compactness}]:")
    print(f"  Retained: {len(df_filtered)} structures")
    print(f"  Removed:  {n_removed} structures ({100*n_removed/len(df_result):.1f}%)")
    
    return df_filtered, df_result['compactness']


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Filter crystal structures by atomic packing compactness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Input CSV file containing structures'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output CSV file (default: <input>_compact<N>.csv)'
    )
    parser.add_argument(
        '--cif-column',
        type=str,
        default='cif',
        help='Name of column containing CIF strings'
    )
    parser.add_argument(
        '--min-compactness',
        type=float,
        default=DEFAULT_COMPACTNESS_MIN,
        help='Minimum compactness threshold'
    )
    parser.add_argument(
        '--max-compactness',
        type=float,
        default=DEFAULT_COMPACTNESS_MAX,
        help='Maximum compactness threshold'
    )
    parser.add_argument(
        '--no-filter',
        action='store_true',
        help='Calculate compactness but do not filter'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Load data
    try:
        print(f"Loading data from {input_path}...")
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} rows")
    except Exception as e:
        print(f"Error loading CSV: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Process structures
    try:
        if args.no_filter:
            df_result = df.copy()
            compactness_values = []
            for cif_string in df[args.cif_column]:
                try:
                    structure = Structure.from_str(cif_string, fmt='cif')
                    compactness_values.append(calculate_compactness(structure))
                except Exception:
                    compactness_values.append(np.nan)
            df_result['compactness'] = compactness_values
            df_filtered = df_result
        else:
            df_filtered, _ = process_structures(
                df,
                cif_column=args.cif_column,
                min_compactness=args.min_compactness,
                max_compactness=args.max_compactness
            )
    except Exception as e:
        print(f"Error processing structures: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output filename
    if args.output:
        output_path = Path(args.output)
    else:
        stem = input_path.stem
        n_structures = len(df_filtered)
        output_path = input_path.parent / f"{stem}_compact{n_structures}.csv"
    
    # Save results
    try:
        df_filtered.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    except Exception as e:
        print(f"Error saving output: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
