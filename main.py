"""
Main entry point for BuildingConvex - Three-Level LOD Simplification and Convexification System.

This script provides a user-friendly interface for processing building geometry files
with different levels of detail (LOD).

Features:
- Three LOD levels: precise, medium, low
- Automatic geometry simplification
- Convexification processing
- Batch processing support
- Detailed logging and progress reporting

"""

import os
import sys
import argparse
import time
from pathlib import Path

from cuger.__transform import process as ps


def validate_directory(directory):
    """Check if directory exists."""
    if not os.path.exists(directory):
        print(f"Error: Directory not found: {directory}")
        return False
    return True


def count_geo_files(directory):
    """Count GEO files in directory."""
    count = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.geo'):
                count += 1
    return count


def process_single_file(input_path, output_dir, lod="medium", verbose=True):
    """
    Process a single GEO file.
    
    Parameters:
    -----------
    input_path : str
        Path to input GEO file
    output_dir : str
        Output directory
    lod : str
        Level of detail: 'precise', 'medium', 'low'
    verbose : bool
        Print detailed messages
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        modelname = Path(input_path).stem
        paths = ps.get_output_paths(modelname, output_dir, lod=lod)
        
        if verbose:
            print(f"\n[Processing] {modelname}")
            print(f"  Input: {input_path}")
        
        # Step 1: Simplify
        if verbose:
            print(f"  [1/2] Simplifying (LOD={lod})...", end=" ", flush=True)
        ps.simplify_process(input_path, paths["simplified_geo_path"], lod=lod)
        if verbose:
            print("[OK]")
        
        # Step 2: Convexify
        if verbose:
            print(f"  [2/2] Convexifying...", end=" ", flush=True)
        ps.convex_process(paths["simplified_geo_path"], paths["convex_geo_path"])
        if verbose:
            print("[OK]")
        
        if verbose:
            print(f"  Output: {paths['convex_geo_path']}")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"[FAILED] Error: {str(e)}")
        return False


def process_directory(input_dir, output_dir, lod="medium", verbose=True):
    """
    Process all GEO files in a directory.
    
    Parameters:
    -----------
    input_dir : str
        Input directory path
    output_dir : str
        Output directory path
    lod : str
        Level of detail
    verbose : bool
        Print detailed messages
        
    Returns:
    --------
    dict
        Statistics of processing
    """
    stats = {"total": 0, "success": 0, "failed": 0}
    
    # Validate input directory
    if not validate_directory(input_dir):
        return stats
    
    # Collect GEO files
    geo_files = []
    for dirpath, dirnames, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('.geo'):
                geo_files.append(os.path.join(dirpath, filename).replace('\\', '/'))
    
    if not geo_files:
        if verbose:
            print(f"No GEO files found in: {input_dir}")
        return stats
    
    stats["total"] = len(geo_files)
    
    # Process files
    if verbose:
        print(f"\n{'='*70}")
        print(f"Processing {len(geo_files)} file(s) with LOD='{lod}'")
        print(f"{'='*70}")
    
    for idx, geo_file in enumerate(geo_files, 1):
        if verbose:
            print(f"\n[{idx}/{len(geo_files)}]", end=" ")
        
        if process_single_file(geo_file, output_dir, lod=lod, verbose=verbose):
            stats["success"] += 1
        else:
            stats["failed"] += 1
    
    return stats


def interactive_mode():
    """Interactive mode for user input."""
    print("\n" + "="*70)
    print("BuildingConvex - Interactive Mode")
    print("="*70)
    
    # Get input directory
    input_dir = input("\nEnter input directory (default: tests/examples): ").strip()
    if not input_dir:
        input_dir = "tests/examples"
    
    if not validate_directory(input_dir):
        return
    
    geo_count = count_geo_files(input_dir)
    print(f"Found {geo_count} GEO file(s)")
    
    if geo_count == 0:
        print("No GEO files to process.")
        return
    
    # Get output directory
    output_dir = input("Enter output directory (default: tests/example_results): ").strip()
    if not output_dir:
        output_dir = "tests/example_results"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Select LOD level
    print("\nAvailable LOD levels:")
    print("  1. precise - Full geometry without simplification")
    print("  2. medium  - Multi-layer OBB (12 faces) - RECOMMENDED")
    print("  3. low     - Single OBB (6 faces)")
    
    lod_choice = input("\nSelect LOD (1-3, default: 2): ").strip()
    lod_map = {"1": "precise", "2": "medium", "3": "low"}
    lod = lod_map.get(lod_choice, "medium")
    
    # Process
    start_time = time.time()
    stats = process_directory(input_dir, output_dir, lod=lod)
    elapsed = time.time() - start_time
    
    # Print results
    print("\n" + "="*70)
    print(f"Processing Complete - LOD: {lod}")
    print("="*70)
    print(f"Total files:   {stats['total']}")
    print(f"Successful:    {stats['success']}")
    print(f"Failed:        {stats['failed']}")
    print(f"Time elapsed:  {elapsed:.2f}s")
    print(f"Output dir:    {output_dir}")
    print("="*70)


def command_line_mode(args):
    """Command line mode with arguments."""
    input_dir = args.input or "tests/examples"
    output_dir = args.output or "tests/example_results"
    lod = args.lod or "medium"
    
    # Validate
    if not validate_directory(input_dir):
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check LOD validity
    if lod not in ["precise", "medium", "low"]:
        print(f"Error: Invalid LOD '{lod}'. Choose from: precise, medium, low")
        sys.exit(1)
    
    # Process
    if args.single:
        # Process single file
        if not os.path.isfile(args.single):
            print(f"Error: File not found: {args.single}")
            sys.exit(1)
        
        success = process_single_file(args.single, output_dir, lod=lod)
        sys.exit(0 if success else 1)
    else:
        # Process directory
        start_time = time.time()
        stats = process_directory(input_dir, output_dir, lod=lod, verbose=True)
        elapsed = time.time() - start_time
        
        print("\n" + "="*70)
        print(f"Results - LOD: {lod}")
        print("="*70)
        print(f"Total:     {stats['total']}")
        print(f"Success:   {stats['success']}")
        print(f"Failed:    {stats['failed']}")
        print(f"Time:      {elapsed:.2f}s")
        print("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="BuildingConvex - Geometry Simplification and Convexification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Interactive mode
  python main.py -i input_dir -o output_dir       # Batch process directory
  python main.py -i input_dir -o output_dir -l medium  # With LOD selection
  python main.py -s input_file.geo -o output_dir  # Process single file
        """
    )
    
    parser.add_argument("-i", "--input", default=None,
                       help="Input directory (default: tests/examples)")
    parser.add_argument("-o", "--output", default=None,
                       help="Output directory (default: tests/example_results)")
    parser.add_argument("-l", "--lod", choices=["precise", "medium", "low"],
                       default=None,
                       help="Level of detail (default: medium)")
    parser.add_argument("-s", "--single", default=None,
                       help="Process single file (geo path)")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Determine mode
    if args.input or args.output or args.lod or args.single:
        # Command line mode
        command_line_mode(args)
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        sys.exit(1)

