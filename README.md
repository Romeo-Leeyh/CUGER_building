# CUGER_building

## Introduction

**CUGER** is a Python-based open source algorithm, which is used for processing the complex building shape/plan to convex geometry, and transfer all kinds of building models to unified graph structure. The method is inspired by **Arkin, Ronald C.'s report (1987)** "Path planning for a vision-based autonomous robot". 

CUGER is constructed over **MOOOSAS**, if you want to use the whole functions, including transfer and recognize the building to structured data. You should install both CUGER and Moosas. Besides, this algorithm iS integrated in Moosas+, yo can find the Moosas version in `moosas/MoosasPy/encoding`.

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

### Example test

You can run the test code in `./tests`, this will process a example building model, and viualise the building convex optimization results.

```bash
python tests/test.py
```

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