# CUGER_building

## Introduction

**CUGER** is a Python-based open-source algorithm designed to decompose complex building shapes/plans into convex geometry and convert various building model formats into a unified graph representation. The method is conceptually inspired by **Arkin, Ronald C. (1987)**, *“Path Planning for a Vision-Based Autonomous Robot.”*

CUGER is built on top of **Moosas**. To access the full set of functions—such as building model conversion, recognition, and structured data generation—you need to install both CUGER and MOOSAS.  
This algorithm is also integrated into **Moosas+**, and the corresponding implementation can be found in [`moosas/MoosasPy/encoding`](https://github.com/UmikoXiao/moosas).

## Installation

### Install from git

```bash
git clone https://github.com/Romeo_Leeyh/CUGER_building.git
cd cuger
pip install -e

# or install requirements directly
pip install -r reqirements.txt 
```

### Install Moosas+

If you just want to try the algorithm to split the buiding models, this step can be ignore. For more instructions for installation, please move to Moosas+ [repository](https://github.com/UmikoXiao/moosas).

## Usage

### Example Test

You can run the example test script in `cuger/test.py`.  
This will process the sample building model located at `tests/examples/example0.geo`  
and generate outputs in `tests/examples_results/` as shown below:

```bash
tests
├── examples
│   └── example0.geo        # Sample test case
└── examples_results
    ├── figure_convex       # Convex decomposition figures
    ├── figure_graph        # Graph visualization figures
    ├── geo_c               # Converted geometric files
    ├── graph/example0
    │ ├── edges.json        # Generated graph edges
    │ └── nodes.json        # Generated graph nodes
    ├── new_geo             # Exported .geo files
    ├── new_idf             # Exported .idf files, this function is developing
    ├── new_rdf             # Exported .rdf files
    ├── new_xml             # Exported .xml files
    └── log                 # Processing logs
```

To run the test:

```bash
python tests/test.py
```

### Module I/O Explanation

CUGER operates through a series of processing modules. Each module consumes specific input files and produces standardized outputs that together form the unified graph-based representation of the building model.

#### **Input**

- **`.geo` file**  
  The primary geometric description of the building.  
  This file contains surfaces, edges, vertex positions, and semantic tags used to reconstruct the building’s spatial structure.

#### **Intermediate Outputs**

- **`figure_convex/`**  
  Visualization of the convex decomposition process, showing how non-convex polygons are split.
- **`figure_graph/`**  
  Graph-level visualizations illustrating nodes, edges, and spatial relationships.
- **`geo_c/`**  
  Converted or cleaned geometric files after preprocessing and convex decomposition.

#### **Graph Outputs**

- **`graph/<case_name>/nodes.json`**  
  Encodes all nodes (faces, spaces, openings, etc.) with geometric, semantic, and topological attributes.
- **`graph/<case_name>/edges.json`**  
  Encodes adjacency relations, directional edges, and multi-scale topology for downstream GNN tasks.

#### **Exported Model Formats**

- **`new_geo/`**  
  Reconstructed geometry exported back into `.geo` format.
- **`new_idf/`** *(in development)*  
  Prototype EnergyPlus IDF export based on the generated graph structure.
- **`new_rdf/`**  
  Semantic web representation exported as RDF, suitable for linked-data workflows, which is dumped in `.owl` format.
- **`new_xml/`**  
  XML-based representation for interoperability with external BIM or simulation environments.

#### **Logs**

- **`log/`**  
  Records the processing pipeline, including geometric checks, convexity reports, and conversion summaries.


### Main use

Developing...

## Citation

If you used this project in your research, please cite the paper below:

```
@inproceedings{Li2024GraphConvex,
    author    = {Li, Yihui and Xiao, Jun and Zhou, Hao and Lin, Borong.},
    title     = {A Cross-Scale Normative Encoding Representation Method for 3D Building Models Suitable for Graph Neural Networks},
    booktitle = {Proceedings of the Building Simulation Conference 2025},
    publisher = {IBPSA},
    address   = {Brisbane, Australia},
    month     = {August},
    year      = {2025},
    pages     = {},
    doi       = {10.26868/25222708.2025.1305},
}
```