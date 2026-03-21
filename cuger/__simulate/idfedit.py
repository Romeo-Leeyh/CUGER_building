"""
idfedit.py
----------
Utility functions for editing Output:Variable objects in EnergyPlus IDF files,
implemented using the eppy library.

This module provides:
- IDFEdit class for editing IDF files
- Querying existing Output:Variable entries
- Adding new Output:Variable entries
- Removing Output:Variable entries
- Updating reporting frequencies
- Converting IDF to NetworkX graph
- Visualizing IDF graph structure
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from eppy.modeleditor import IDF


class IDFEdit:
    """
    IDF editor for modifying EnergyPlus IDF files.
    
    This class provides methods to:
    - Edit Output:Variable objects
    - Add diagnostics
    - Convert IDF to graph structure
    - Visualize building geometry
    """
    
    def __init__(self, idf: IDF):
        """
        Initialize IDFEdit with an IDF object.
        
        Args:
            idf: eppy IDF object
        """
        self.idf = idf
    
    def saveas(self, path: str) -> None:
        """
        Save IDF to file.
        
        Args:
            path: Output file path
        """
        return self.idf.saveas(path)
    
    # ========== Internal utilities ==========
    
    @staticmethod
    def _get_field_value(eppy_obj, field_label: str):
        """
        Get field value from eppy object, handling various naming conventions.
        
        Args:
            eppy_obj: eppy IDF object
            field_label: Field label to retrieve
            
        Returns:
            Field value or None if not found
        """
        candidates = [
            field_label,
            field_label.replace(" ", "_"),
            field_label.replace(" ", "_").lower(),
            field_label.replace(" ", "_").title(),
            field_label.replace(" ", ""),
        ]
        for c in candidates:
            if hasattr(eppy_obj, c):
                return getattr(eppy_obj, c)
        return None
    
    @staticmethod
    def _set_field_value(eppy_obj, field_label: str, value) -> bool:
        """
        Set field value in eppy object, handling various naming conventions.
        
        Args:
            eppy_obj: eppy IDF object
            field_label: Field label to set
            value: Value to set
            
        Returns:
            True if successful, False otherwise
        """
        candidates = [
            field_label,
            field_label.replace(" ", "_"),
            field_label.replace(" ", "_").lower(),
            field_label.replace(" ", "_").title(),
            field_label.replace(" ", ""),
        ]
        for c in candidates:
            try:
                setattr(eppy_obj, c, value)
                return True
            except Exception:
                continue
        
        try:
            setattr(eppy_obj, field_label.replace(" ", "_"), value)
            return True
        except Exception:
            return False
    
    # ========== Output:Variable operations ==========
    
    def list_output_variables(self) -> List[Dict]:
        """
        List all Output:Variable objects in IDF.
        
        Returns:
            List of dictionaries containing output variable information
        """
        objs = self.idf.idfobjects.get("OUTPUT:VARIABLE", [])
        results = []
        
        for o in objs:
            key_value = self._get_field_value(o, "Key Value")
            variable_name = self._get_field_value(o, "Variable Name")
            frequency = self._get_field_value(o, "Reporting Frequency")
            
            results.append({
                "obj": o,
                "key_value": key_value,
                "variable_name": variable_name,
                "reporting_frequency": frequency,
            })
        
        return results
    
    def add_output_variable(
        self,
        key_value: str,
        variable_name: str,
        reporting_frequency: str = "Hourly",
        allow_duplicate: bool = False
    ):
        """
        Add Output:Variable to IDF.
        
        Args:
            key_value: Key value for output variable (e.g., "*" for all)
            variable_name: Variable name to output
            reporting_frequency: Reporting frequency (e.g., "Hourly", "Daily")
            allow_duplicate: If True, allow duplicate entries
            
        Returns:
            Created IDF object
            
        Raises:
            ValueError: If duplicate exists and allow_duplicate is False
        """
        existing = self.list_output_variables()
        
        for item in existing:
            if (item["key_value"] == key_value and
                item["variable_name"] == variable_name and
                item["reporting_frequency"] == reporting_frequency):
                
                if not allow_duplicate:
                    raise ValueError(
                        "Output:Variable entry already exists with the same "
                        "key, variable name, and reporting frequency."
                    )
        
        obj = self.idf.newidfobject(
            "Output:Variable",
            Key_Value=key_value,
            Variable_Name=variable_name,
            Reporting_Frequency=reporting_frequency
        )
        return obj
    
    def remove_output_variable(
        self,
        key_value: Optional[str] = None,
        variable_name: Optional[str] = None,
        reporting_frequency: Optional[str] = None
    ) -> int:
        """
        Remove Output:Variable objects matching criteria.
        
        Args:
            key_value: Key value to match (None = match all)
            variable_name: Variable name to match (None = match all)
            reporting_frequency: Frequency to match (None = match all)
            
        Returns:
            Number of objects removed
        """
        objs = self.idf.idfobjects.get("OUTPUT:VARIABLE", [])
        targets = []
        
        for o in objs:
            kv = self._get_field_value(o, "Key Value")
            var = self._get_field_value(o, "Variable Name")
            freq = self._get_field_value(o, "Reporting Frequency")
            
            match = True
            if key_value is not None and kv != key_value:
                match = False
            if variable_name is not None and var != variable_name:
                match = False
            if reporting_frequency is not None and freq != reporting_frequency:
                match = False
            
            if match:
                targets.append(o)
        
        for o in targets:
            self.idf.idfobjects["Output:Variable"].remove(o)
        
        return len(targets)
    
    def add_diagnostics(self, *keys: str):
        """
        Add Output:Diagnostics object to IDF.
        
        Args:
            *keys: Diagnostic keys to add (e.g., "DisplayAdvancedReportVariables")
            
        Returns:
            Created IDF object
        """
        # Clear existing diagnostics (unique object)
        self.idf.idfobjects["Output:Diagnostics"].clear()
        
        obj = self.idf.newidfobject("Output:Diagnostics")
        
        # Add extensible fields
        for i, key in enumerate(keys):
            obj["Key_" + str(i + 1)] = key
        
        return obj
    
    # ========== Graph conversion and visualization ==========
    
    def to_networkx_graph(self) -> nx.MultiDiGraph:
        """
        Convert IDF to NetworkX graph.
        
        Nodes:
            - Face nodes: BuildingSurface:Detailed
            - Space nodes: Zone
        
        Edges:
            - face --adjacent_to--> face
            - face --belongs_to--> space
        
        Returns:
            NetworkX MultiDiGraph
        """
        G = nx.MultiDiGraph()
        
        # Add SPACE nodes (Zone)
        zones = self.idf.idfobjects["ZONE"]
        
        for z in zones:
            G.add_node(
                f"space::{z.Name}",
                node_type="space",
                name=z.Name,
                volume=getattr(z, "Volume", None),
                floor_area=getattr(z, "Floor_Area", None),
            )
        
        # Add FACE nodes
        surfaces = self.idf.idfobjects["BUILDINGSURFACE:DETAILED"]
        
        for s in surfaces:
            # Calculate centroid
            try:
                verts = np.array(s.coords)
                centroid = verts.mean(axis=0).tolist()
            except Exception:
                centroid = [None, None, None]
            
            G.add_node(
                f"face::{s.Name}",
                node_type="face",
                name=s.Name,
                surface_type=s.Surface_Type,
                zone=s.Zone_Name,
                area=s.area,
                tilt=s.tilt,
                azimuth=s.azimuth,
                outside_bc=s.Outside_Boundary_Condition,
                centroid=centroid,
            )
        
        # Face → Space (belongs_to)
        for s in surfaces:
            face_id = f"face::{s.Name}"
            space_id = f"space::{s.Zone_Name}"
            
            if space_id in G:
                G.add_edge(
                    face_id,
                    space_id,
                    relation="belongs_to",
                )
        
        # Face ↔ Face (adjacent_to)
        for s in surfaces:
            if s.Outside_Boundary_Condition == "Surface":
                other = s.Outside_Boundary_Condition_Object
                if other:
                    f1 = f"face::{s.Name}"
                    f2 = f"face::{other}"
                    
                    if f1 in G and f2 in G:
                        G.add_edge(
                            f1,
                            f2,
                            relation="adjacent_to",
                            internal=True,
                        )
                        # Bidirectional edge
                        G.add_edge(
                            f2,
                            f1,
                            relation="adjacent_to",
                            internal=True,
                        )
        
        return G
    