# NEMO 🐠

## 🔬 Nematics and Morphology Image Analysis Toolkit

*Konstantinos Andreadis*  
PhD Student @ University of Geneva  
Aurélien Roux Lab & Guillaume Salbreux Lab

---

## 👋 Hello !

Welcome to **NEMO**, the **Nematics and Morphology Image Analysis Toolkit**.

This toolkit is designed to provide efficient and modular image analysis modules, enabling the study of nematics and
morphological features through a combination of customizable scripts and batch processing capabilities.

Fully developed in Python, NEMO reads hyperstack TZCYX .tif(f) files, and performs:

- Mesh segmentation
- Mesh curvature and thickness measurements
- Projection onto mesh
- Nematic order quantification
- Topological defects characeterisation

on:

- Planar surfaces
- Curved surfaces
- Bulk volumes

The output consists of a folder structure containing the raw data and result figures, customisable and loadable by the
user at any step in the pipeline.

It has been tested on both Apple (MacBook Pro 16-inch, 2023) and Windows (Analysis PC @Roux Lab).

I am actively developing this pipeline, with new commits adding/changing modules in the backend.
If you come accross any issue, don't hesitate to post a GitLab issue or email me (konstantinos.andreadis@unige.ch) !

---

## 📂 **Project Structure**

The project is structured as follows:

```
NEMO/  
├── scripts/                         # Folder of Module Scripts
│   ├── analysis.py                  # Image / Mesh Analysis Modules
│   ├── visuals.py                   # Visualisation Modules
│   ├── datahandler.py               # Input / Output Data Handling Modules
│   └── simulation.py                # Simulation Modules
├── debugging/                       # Folder of Debugging Scripts
│   ├── fix_dependencies.ipynb       # Dependency Fix Notebook
│   ├── sandbox.py                   # Debugging Sandbox Notebook
│   └── generator.py                 # Simulation Notebook
├── batch_analysis/                  # Folder of Batch Analysis Scripts
│   ├── batch_analysis.py            # Example Batch Analysis
│   ├── embl_spherical_projection/   # Folder of Spherical Projection (for EMBL)
│   │   ├── EMBL_README.md           # Explanation README
│   │   ├── spherical_projection.py  # Automated Batch Script
│   │   ├── run_params.json          # Batch Run Configuration
│   │   ├── run.sh                   # Example Run Shell Script
│   └── └── line_measure.ijm         # FIJI Module for Line Measurements
├── main.ipynb                       # Main Pipeline Workspace Notebook
├── requirements.txt                 # Required Python Libraries
└── README.md                        # This README
```  

---

## 🚀 **Getting Started**

### Prerequisites

- Make sure you have exactly **Python 3.9.6** installed (to allow napari and trimesh to work correctly). You can check
  your version by running:

```bash
python --version
```  

### Installation

1. Clone this repository:

- Download the source code from GitHub (https://gitlab.unige.ch/salbreux-group/konstantinos-andreadis/nemo.git)
- Or extract the .zip file if sent privately.

2. Install the required dependencies in the virtual environment:

If you have a working **Python 3.9.6** compiler installed, you can directly install all packages using:

```bash
pip install -r requirements.txt  
```  

However, if you have conda installed, you can also follow the following steps in your Anaconda prompt which have been
tested and work:

```bash
conda create -n nemo_env python=3.9.6
```

```bash
activate nemo_env
```

```bash
pip install -r requirements.txt
```

```bash
jupyter lab
```  

3. If any of the dependencies are not installing correctly, please consult **`fix_dependencies.ipynb`** using Jupyter:

```bash
jupyter fix_dependencies main.ipynb  
```

---

## 📝 **Usage**

### Main Notebook

- The primary interface for interactive analysis is the **`main.ipynb`**. You can open it using Jupyter:

```bash
jupyter notebook main.ipynb
```

- Or, in conda, use

### Batch Analysis

- The **`batch_analysis.py`** script inside the folder **`batch_analysis/`** allows batch processing by combining multiple modules.
- To use it, simply configure your desired pipeline within the script and run:

    ```bash
    python batch_analysis.py  
    ```  

You can customize the pipeline by combining modules from the **`scripts/`** folder.

---

## 🧩 **Modular Scripts**

The **`scripts/`** folder contains Python scripts with modular functions that can be (re-)combined for batch analysis or
used individually within the main notebook.

Each script serves a different purpose, such as image/mesh analysis, data input/output handling, visualisation or
generation of test data.


---

## 🪲 **Debugging**

- The **`debugging/`** folder contains scripts and logs that help diagnose issues or develop new modules.
- If you encounter problems during execution, please test your tools in the  **`sandbox.ipynb`** using Jupyter:

```bash
jupyter sandbox.ipynb  
```
- Or, if you wish to generate a .tiff, or e.g. curved director field, consult **`generator.ipynb`** using Jupyter:

```bash
jupyter generator.ipynb  
```

---

## 🛠️ **Contributing**

Contributions are welcome!

If you have an idea for a new module that should be added to NEMO, please contact me :)

---

## 📄 **License**

MIT License with Citation Requirement

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

- **Citation Requirement**: If the Software is used directly or indirectly in any scientific publication or
  presentation, proper citation of the original work and author is required.

- Please cite the project as shown below:

**[!] publication incl. NEMO to be announced  [!]**
 
2024-2028 Copyright *Konstantinos Andreadis 