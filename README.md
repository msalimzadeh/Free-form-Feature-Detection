# Free-form Feature Detection & Extraction

A tool for automatically detecting and extracting **free-form geometric features** from machined parts. Given the workpiece and a stock block (both as STEP files), the tool interactively identifies free-form surface patches and computes the exact 3D removal volume needed to machine the part from the stock — ready for downstream CAM planning and simulation.

---

## Overview

This tool:

1. Builds a **face adjacency graph** from the part's B-rep topology
2. Computes geometric attributes for every face (surface type, normal, dihedral angles)
3. Lets the user **interactively pick a seed face** in a 3D viewer
4. Runs a **region-growing algorithm** (BFS) to expand the selection to all connected free-form faces
5. **Extrudes** the detected face patch along the machining direction into a solid prism
6. Computes the **Boolean intersection** of the prism with the stock shape to get the exact removal volume
7. Exports the result as a STEP file

---

## Pipeline

```
STEP Files (part + stock)
        │
        ▼
  Load & extract faces
        │
        ▼
  Build face adjacency graph
  (nodes = faces, edges = shared boundaries)
        │
        ▼
  Compute face attributes
  (area, normal, curvature, surface type)
        │
        ▼
  Compute dihedral angles on all edges
        │
        ▼
  User picks a seed face  ◄── Interactive 3D viewer
        │
        ▼
  Region growing (BFS)
  - Expand to neighbours with dihedral angle < 40°
  - Exclude planes not aligned with machining direction
        │
        ▼
  Sew faces → Extrude along machining direction
        │
        ▼
  Boolean intersection: stock ∩ extrusion prism
        │
        ▼
  Export removal volume → feature_removal.step
```

---


## Installation

### Requirements

- Python 3.9+
- [OCP (opencascade-core)](https://github.com/CadQuery/OCP) — Python bindings for OpenCASCADE
- numpy
- pyvista
- networkx
- trimesh
- matplotlib

### Setup

```bash
git clone https://github.com/autonomiq-dev/Free-form-Feature-Detection.git
cd Free-form-Feature-Detection-Extraction

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install ocp numpy pyvista networkx trimesh matplotlib
```

> **Note:** OCP installation can vary by platform. Refer to [CadQuery OCP releases](https://github.com/CadQuery/OCP/releases) for pre-built wheels if a direct `pip install` fails.

---

## Usage

```bash
python main.py --part <part.step> --stock <stock.step> --dir <DX> <DY> <DZ> [--out <output.step>]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--part` | Yes | Path to the finished part STEP file |
| `--stock` | Yes | Path to the stock/blank STEP file |
| `--dir` | Yes | Machining direction vector, e.g. `0 0 1` for Z-axis |
| `--out` | No | Output STEP file path (default: `feature_removal.step`) |

### Examples

```bash
# Machine along Z-axis
python main.py --part Samples/freecad_models/case_4/ff_case_4.step \
               --stock Samples/freecad_models/case_4/ff_case_4_stock.step \
               --dir 0 0 1

# Custom output file
python main.py --part model.step --stock stock.step --dir 0 0 -1 --out removal.step
```



