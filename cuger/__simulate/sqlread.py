"""
sqlread.py
----------
EnergyPlus SQL output file reader and analyzer.

This module provides:
- SQLReader class for reading EnergyPlus SQL outputs
- Zone-level energy data extraction
- Load analysis and visualization
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from ladybug.sql import SQLiteResult  
from pathlib import Path

class SQLReader:
    """
    EnergyPlus SQL output reader.
    
    This class provides methods to:
    - Read zone-level hourly data
    - Calculate external and ideal loads
    - Export data to CSV
    - Visualize load comparisons
    """
    
    # Default external gain variables
    DEFAULT_EXTERNAL_GAIN_VARS = [
        "Zone Windows Total Heat Gain Energy",
        "Zone Opaque Surface Inside Faces Total Conduction Heat Gain Energy",
        "Zone Infiltration Total Heat Gain Energy",
    ]
    
    # Default external loss variables
    DEFAULT_EXTERNAL_LOSS_VARS = [
        "Zone Windows Total Heat Loss Energy",
        "Zone Opaque Surface Inside Faces Total Conduction Heat Loss Energy",
        "Zone Infiltration Total Heat Loss Energy",
    ]
    
    # Ideal loads variables
    IDEAL_HEATING_VAR = "Zone Ideal Loads Supply Air Total Heating Energy"
    IDEAL_COOLING_VAR = "Zone Ideal Loads Supply Air Total Cooling Energy"
    
    def __init__(self, sql_path: str):
        """
        Initialize SQLReader.
        
        Args:
            sql_path: Path to eplusout.sql file
        """
        if SQLiteResult is None:
            raise ImportError(
                "ladybug is required for SQLReader. Install with: pip install ladybug-core"
            )

        self.sql_path = Path(sql_path)
        self.sql = SQLiteResult(str(self.sql_path))
        self.available_outputs = self._get_available_outputs()
        self.zone_hourly_data = None
    
    def _get_available_outputs(self):
        """Get available outputs from SQL file."""
        if callable(self.sql.available_outputs_info):
            return self.sql.available_outputs_info()
        else:
            return self.sql.available_outputs_info
    
    def read_zone_data(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Read all zone-level hourly data from SQL file.
        
        Returns:
            Dictionary mapping zone_id to output_name to hourly values
        """
        zone_hourly_data = defaultdict(lambda: defaultdict(lambda: [0.0] * 8760))
        
        for available_output in self.available_outputs:
            output_name = available_output['output_name']
            #print(f"Processing output: {output_name}")
            
            data = self.sql.data_collections_by_output_name(output_name)
            
            if "Surface" in output_name:
                # Surface-level data: aggregate by zone
                for data_item in data:
                    surface_name = data_item.header.metadata['Surface']
                    pattern = r"^([A-Z0-9]+(?:_[A-Z0-9]+)*)"
                    match = re.match(pattern, surface_name)
                    if match:
                        zone_id = match.group(1)
                        for i, value in enumerate(data_item.values):
                            zone_hourly_data[zone_id][output_name][i] += value
            
            elif "Zone" in output_name:
                # Zone-level data
                for data_item in data:
                    if any(key in output_name for key in ["Ventilation", "Infiltration", "Humidity", "Heat Balance"]):
                        zone_id = data_item.header.metadata['System']
                    elif "Ideal Loads Supply Air" in output_name:
                        pattern = r"^(.*?)(?:[_\-\s]?IDEAL LOADS AIR.*)?$"
                        zone_id = data_item.header.metadata['System']
                        match = re.match(pattern, zone_id)
                        if match:
                            zone_id = match.group(1)
                    else:
                        zone_id = data_item.header.metadata['Zone'].removesuffix("_SPACE")
                    
                    for i, value in enumerate(data_item.values):
                        zone_hourly_data[zone_id][output_name][i] += value
        
        self.zone_hourly_data = zone_hourly_data
        return zone_hourly_data
    
    def get_zone_ids(self) -> List[str]:
        """
        Get list of zone IDs.
        
        Returns:
            List of zone IDs
        """
        if self.zone_hourly_data is None:
            self.read_zone_data()
        return list(self.zone_hourly_data.keys())
    
    def get_zone_areas(self, zone_ids):
        """
        Return the floor area of a given zone using Zone Summary table.
        Automatically converts to float and skips non-numeric values.
        """
        table = self.sql.tabular_data_by_name("Zone Summary")  # OrderedDict[row_name] = row_values
        
        zone_areas = {}

        for zone_id in zone_ids:
            raw_area = table[zone_id][0]
            try:
                zone_areas[zone_id] = float(raw_area)
            except (TypeError, ValueError):
                zone_areas[zone_id] = None
    
        return zone_areas


    
    def export_zone_to_csv(
        self,
        zone_id: str,
        output_path: str
    ) -> None:
        """
        Export zone hourly data to CSV.
        
        Args:
            zone_id: Zone ID to export
            output_path: Output CSV file path
        """
        if self.zone_hourly_data is None:
            self.read_zone_data()
        
        if zone_id not in self.zone_hourly_data:
            raise ValueError(f"Zone {zone_id} not found in data")
        
        data = self.zone_hourly_data[zone_id]
        df = pd.DataFrame({k: pd.Series(v) for k, v in data.items()})
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Exported {zone_id} data to {output_file}")
    
    def export_all_zones_to_csv(
        self,
        output_dir: str,
        prefix: str = ""
    ) -> None:
        """
        Export all zones' hourly data to CSV files.
        
        Args:
            output_dir: Output directory
            prefix: Optional prefix for filenames
        """
        if self.zone_hourly_data is None:
            self.read_zone_data()

        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        
        for zone_id in self.zone_hourly_data.keys():
            filename = f"{prefix}{zone_id}_hourly_outputs.csv" if prefix else f"{zone_id}_hourly_outputs.csv"
            output_path = output_root / filename
            self.export_zone_to_csv(zone_id, str(output_path))
    
    def export_external_loads_csv(
        self,
        output_path: str,
        external_gain_vars: Optional[List[str]] = None,
        external_loss_vars: Optional[List[str]] = None
    ) -> None:
        """
        Export a single CSV containing each zone's external load time series (via calculate_load).
        
        Args:
            output_path: CSV file path to write (one CSV per SQL run)
            aggregate_to_daily: If True, export daily aggregated values (365 rows); else hourly (8760 rows)
            external_gain_vars: Optional list to pass to calculate_load
            external_loss_vars: Optional list to pass to calculate_load
        """
        if self.zone_hourly_data is None:
            self.read_zone_data()
        
        zone_ids = self.get_zone_ids()
        zone_areas = self.get_zone_areas(zone_ids)
        
        if not zone_ids:
            raise ValueError("No zones available to export.")
        
        # Determine length from first zone's outputs
        sample_zone = zone_ids[0]
        sample_outputs = self.zone_hourly_data[sample_zone]
        n = len(list(sample_outputs.values())[0])
        
        # Prepare DataFrame
        data = {}
        for zone_id in zone_ids:
            loads = self.calculate_loads(
                zone_id,
                external_gain_vars=external_gain_vars,
                external_loss_vars=external_loss_vars,
            )

            area = zone_areas.get(zone_id, None)

            if area is None or area <= 0:
                raise ValueError(f"Invalid area for zone {zone_id}")
            data[zone_id] = loads["external"] / area
        
        df = pd.DataFrame(data)

        df.index = pd.RangeIndex(start=1, stop=1 + df.shape[0], name="Hour")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=True)
        #print(f"Exported external loads for {len(zone_ids)} zones to {output_file}")
    
    @staticmethod
    def analyze_results(sql_path: str, data_dir: str) -> None:
        """
        Analyze simulation results.
        
        Args:
            sql_path: Path to SQL file
            data_dir: Directory for analysis outputs
        """
        try:
            print(f"\n  Analyzing results...")
            reader = SQLReader(sql_path)
            reader.read_zone_data()
            
            zone_ids = reader.get_zone_ids()
            print(f"    Found {len(zone_ids)} zones: {zone_ids}")
            
            # Create output directory
            sql_name = Path(sql_path).parent.name

            csv_path = Path(data_dir) / f"{sql_name}.csv"

            reader.export_external_loads_csv(
                output_path=str(csv_path),
            )
            print(f"    ✓ Exported CSV to: {csv_path}")
            
            
        except Exception as e:
            print(f"    ✗ Analysis failed: {e}")
    
    def calculate_loads(
        self,
        zone_id: str,
        external_gain_vars: Optional[List[str]] = None,
        external_loss_vars: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Calculate external and ideal loads for a zone.
        
        Args:
            zone_id: Zone ID to analyze
            external_gain_vars: List of external gain variable names
            external_loss_vars: List of external loss variable names
            aggregate_to_daily: If True, aggregate hourly to daily averages
            
        Returns:
            Dictionary with "external" and "ideal" load arrays
        """
        if self.zone_hourly_data is None:
            self.read_zone_data()
        
        if zone_id not in self.zone_hourly_data:
            raise ValueError(f"Zone {zone_id} not found in data")
        
        outputs = self.zone_hourly_data[zone_id]
        n = len(list(outputs.values())[0])
        
        # Use default variables if not provided
        if external_gain_vars is None:
            external_gain_vars = self.DEFAULT_EXTERNAL_GAIN_VARS
        if external_loss_vars is None:
            external_loss_vars = self.DEFAULT_EXTERNAL_LOSS_VARS
        
        # Calculate external load
        ext_load = np.zeros(n)
        
        for var in external_gain_vars:
            if var in outputs:
                ext_load += np.array(outputs[var])
        
        for var in external_loss_vars:
            if var in outputs:
                ext_load -= np.array(outputs[var])
        
        # Calculate ideal load (Cooling - Heating)
        heating = np.array(outputs.get(self.IDEAL_HEATING_VAR, np.zeros(n)))
        cooling = np.array(outputs.get(self.IDEAL_COOLING_VAR, np.zeros(n)))
        ideal_load = cooling - heating
        
        return {
            "external": ext_load,
            "ideal": ideal_load
        }
    

