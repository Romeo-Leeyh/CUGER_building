"""
simulate.py
-----------
EnergyPlus simulation utilities.

This module provides:
- EnergyPlus installation location utilities
- IDF file preparation and modification
- Single and batch simulation runners
- Output variable configuration
"""

import zipfile
import tempfile
from typing import Optional, List, Tuple
from eppy.modeleditor import IDF
from pathlib import Path

from .idfedit import IDFEdit 



# Global variables for EnergyPlus paths
EPLUS_IDD_PATH = None
EPLUS_EXE_PATH = None


def locate_eplus(eplus_root: Optional[str] = None) -> Tuple[str, str]:
    """
    Locate EnergyPlus executable and IDD file paths and assign to global variables.
    
    Args:
        eplus_root: Optional root directory to search for EnergyPlus installation.
                   If None, searches common installation locations.
    
    Returns:
        Tuple of (exe_path, idd_path)
    
    Raises:
        FileNotFoundError: If EnergyPlus installation cannot be found
    """
    global EPLUS_EXE_PATH, EPLUS_IDD_PATH
    
    def search_in_root(root: str | Path) -> Optional[Tuple[str, str]]:
        """Search for energyplus.exe and Energy+.idd in root and its subfolders."""
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists() or not root_path.is_dir():
            return None

        exe = root_path / "energyplus.exe"
        idd = root_path / "Energy+.idd"
        if exe.is_file() and idd.is_file():
            return str(exe), str(idd)

        for subdir in root_path.iterdir():
            if not subdir.is_dir():
                continue
            exe = subdir / "energyplus.exe"
            idd = subdir / "Energy+.idd"
            if exe.is_file() and idd.is_file():
                return str(exe), str(idd)
        
        return None
    
    # Search in provided directory
    if eplus_root is not None:
        result = search_in_root(eplus_root)
        if result is None:
            raise FileNotFoundError(
                f"EnergyPlus not found in provided directory or its subdirectories: {eplus_root}"
            )
        EPLUS_EXE_PATH, EPLUS_IDD_PATH = result
        return EPLUS_EXE_PATH, EPLUS_IDD_PATH
    
    # Search common installation locations
    common_roots = [
        Path("C:/"),
        Path("D:/"),
        Path("E:/"),
        Path("C:/Program Files"),
        Path("C:/Program Files (x86)"),
    ]

    candidates: list[Path] = []
    for root in common_roots:
        if root.exists() and root.is_dir():
            candidates.extend(root.glob("EnergyPlus*"))

    candidates = sorted(candidates, reverse=True)
    
    for c in candidates:
        result = search_in_root(c)
        if result is not None:
            EPLUS_EXE_PATH, EPLUS_IDD_PATH = result
            return EPLUS_EXE_PATH, EPLUS_IDD_PATH
    
    raise FileNotFoundError(
        "EnergyPlus installation could not be located. "
        "Please provide the installation directory explicitly."
    )


# Default output variables for energy analysis
DEFAULT_OUTPUT_VARIABLES = [
    # Zone total loads
    ("*", "Zone Ideal Loads Supply Air Total Cooling Energy", "Hourly"),
    ("*", "Zone Ideal Loads Supply Air Total Heating Energy", "Hourly"),
    
    # Opaque and fenestration heat transfer
    ("*", "Zone Windows Total Heat Gain Energy", "Hourly"),
    ("*", "Zone Windows Total Heat Loss Energy", "Hourly"),
    ("*", "Zone Opaque Surface Inside Faces Total Conduction Heat Gain Energy", "Hourly"),
    ("*", "Zone Opaque Surface Inside Faces Total Conduction Heat Loss Energy", "Hourly"),
    
    # Infiltration and ventilation loads
    ("*", "Zone Infiltration Total Heat Gain Energy", "Hourly"),
    ("*", "Zone Infiltration Total Heat Loss Energy", "Hourly"),
    ("*", "Zone Ventilation Latent Heat Gain Energy", "Hourly"),
    ("*", "Zone Ventilation Latent Heat Loss Energy", "Hourly"),
    ("*", "Zone Ventilation Sensible Heat Gain Energy", "Hourly"),
    ("*", "Zone Ventilation Sensible Heat Loss Energy", "Hourly"),
    
    # Internal gains
    ("*", "Zone Total Internal Total Heating Energy", "Hourly"),
]


def get_idf_files(input_dir: str) -> List[str]:
    """
    Get all IDF files from input directory.
    
    Args:
        input_dir: Directory to search for IDF files
        
    Returns:
        List of sorted IDF file paths
    """
    input_root = Path(input_dir)
    if not input_root.exists() or not input_root.is_dir():
        return []

    return sorted(str(p) for p in input_root.glob("*.idf") if p.is_file())


def prepare_idf(
    idf_path: str,
    output_path: str,
    epw_path: str,
    modify_outputs: bool = True,
    output_variables: Optional[List[Tuple[str, str, str]]] = None,
    diagnostics: Optional[List[str]] = None,
    verbose: bool = False
) -> None:
    """
    Prepare IDF file, optionally modifying output variables.
    
    This function combines loading, optional modification, and saving of IDF files.
    
    Args:
        idf_path: Input IDF file path
        output_path: Output IDF file path
        epw_path: EPW file path
        modify_outputs: Whether to modify output variables (default: True)
        output_variables: List of (key, variable_name, frequency) tuples.
                         If None and modify_outputs=True, uses DEFAULT_OUTPUT_VARIABLES.
        diagnostics: List of diagnostic keys (e.g., ["DisplayAdvancedReportVariables"])
        verbose: Whether to print detailed information
    """
    if EPLUS_IDD_PATH is None:
        raise RuntimeError(
            "EnergyPlus IDD path not set. Call locate_eplus() first."
        )
    
    # Set IDD path
    IDF.setiddname(EPLUS_IDD_PATH)
    
    # Load IDF
    idf = IDF(idf_path, epw_path)
    
    if modify_outputs:
        # Create editor
        editor = IDFEdit(idf)
        
        # Remove existing output variables
        removed_count = editor.remove_output_variable()
        if verbose:
            print(f"    Removed {removed_count} existing output variables")
        
        # Add diagnostics
        if diagnostics:
            editor.add_diagnostics(*diagnostics)
            if verbose:
                print(f"    Added diagnostics: {', '.join(diagnostics)}")
        
        # Add output variables
        if output_variables is None:
            output_variables = DEFAULT_OUTPUT_VARIABLES
        
        if output_variables:
            for key, var, freq in output_variables:
                editor.add_output_variable(key, var, freq)
            if verbose:
                print(f"    Added {len(output_variables)} output variables")
        
        # Save modified IDF
        editor.saveas(output_path)
    else:
        # Save IDF without modification
        idf.saveas(output_path)
        if verbose:
            print(f"    IDF copied without modification")


def run_eplus(
    idf_path: str,
    epw_path: str,
    output_dir: str,
    verbose: str = "v"
) -> bool:
    """
    Run EnergyPlus simulation.
    
    Args:
        idf_path: Path to IDF file
        epw_path: Path to EPW or ZIP file
        output_dir: Directory for simulation outputs
        expandobjects: Whether to expand objects before simulation
        verbose: Verbosity level ("v" for verbose, "q" for quiet)
    
    Returns:
        True if simulation succeeded, False otherwise
    
    Raises:
        RuntimeError: If EnergyPlus paths are not set (call locate_eplus first)
    """
    if EPLUS_IDD_PATH is None or EPLUS_EXE_PATH is None:
        raise RuntimeError(
            "EnergyPlus paths not set. Call locate_eplus() first."
        )
    
    idf_input = Path(idf_path)
    epw_input = Path(epw_path)
    output_root = Path(output_dir)

    idf_name = idf_input.stem
    epw_name = epw_input.stem
    
    # Handle EPW or ZIP file
    temp_epw_dir = None
    if epw_input.suffix.lower() == ".epw":
        final_epw_path = str(epw_input)
    else:
        temp_epw_dir = tempfile.TemporaryDirectory()
        temp_epw_root = Path(temp_epw_dir.name)
        with zipfile.ZipFile(epw_input, "r") as zf:
            zf.extractall(temp_epw_dir.name)
        epw_files = sorted(p for p in temp_epw_root.glob("*.epw") if p.is_file())
        if not epw_files:
            if temp_epw_dir is not None:
                temp_epw_dir.cleanup()
            print(f"No EPW file found in ZIP: {epw_path}")
            return False
        final_epw_path = str(epw_files[0])
    
    # Prepare run directory
    run_dir = output_root / f"{idf_name}__{epw_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Load IDF with EPW
    idf = IDF(str(idf_input), final_epw_path)
    
    # Save IDF to run directory
    temp_idf_path = run_dir / f"{idf_name}.idf"
    
    # Update IDF name to point to saved file
    idf.idfname = str(temp_idf_path)
    
    # Run EnergyPlus
    try:
        idf.run(
            output_directory=str(run_dir),
            verbose=verbose
        )
        return True
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        return False
    finally:
        if temp_epw_dir is not None:
            temp_epw_dir.cleanup()


def run_single_simulation(
    idf_path: str,
    epw_path: str,
    output_dir: str,
    modify_outputs: bool = True,
    output_variables: Optional[List[Tuple[str, str, str]]] = None,
    diagnostics: Optional[List[str]] = None,
    verbose: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Run a single EnergyPlus simulation with optional IDF modification.
    
    This is a high-level function that combines IDF preparation and simulation.
    
    Args:
        idf_path: Path to IDF file
        epw_path: Path to EPW file
        output_dir: Output directory
        modify_outputs: Whether to modify output variables
        output_variables: Output variables to add (if modify_outputs=True)
        diagnostics: Diagnostics to add (if modify_outputs=True)
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (success: bool, sql_path: Optional[str])
    """
    idf_name = Path(idf_path).stem
    output_root = Path(output_dir)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {idf_name}")
        print(f"{'='*60}")
    
    # Prepare IDF
    temp_idf_path = output_root / f"{idf_name}.idf"
    output_root.mkdir(parents=True, exist_ok=True)
    
    try:
        if verbose:
            print(f"Preparing IDF...")
        
        prepare_idf(
            idf_path=idf_path,
            output_path=str(temp_idf_path),
            epw_path=epw_path,
            modify_outputs=modify_outputs,
            output_variables=output_variables,
            diagnostics=diagnostics,
            verbose=verbose
        )
        
        if verbose:
            print(f"✓ IDF prepared")
    except Exception as e:
        print(f"✗ Error preparing IDF: {e}")
        return False, None
    
    # Run simulation
    if verbose:
        print(f"Running simulation...")
    
    success = run_eplus(
        idf_path=temp_idf_path,
        epw_path=epw_path,
        output_dir=str(output_root),
        verbose="q"  # Quiet mode for cleaner output
    )
    
    if success:
        # Find SQL output
        epw_name = Path(epw_path).stem
        sql_path = output_root / f"{temp_idf_path.stem}__{epw_name}" / "eplusout.sql"

        if sql_path.exists():
            if verbose:
                print(f"✓ Simulation completed: {idf_name}")
            return True, str(sql_path)
        else:
            if verbose:
                print(f"⚠ Simulation completed but SQL file not found")
            return True, None
    else:
        if verbose:
            print(f"✗ Simulation failed: {idf_name}")
        return False, None
