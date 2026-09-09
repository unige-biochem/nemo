# NEMO 🐠

## 🔬 Nematics and Morphology Image Analysis Toolkit

*Konstantinos Andreadis*  
PhD Student @ University of Geneva  
Aurélien Roux Lab & Guillaume Salbreux Lab

---

## 👋 Hello !

Welcome to **NEMO**, the **Nematics and Morphology Image Analysis Toolkit**.

This toolkit provides advanced 3D image analysis modules, enabling the study of nematics and
morphological features through a combination of customizable scripts and batch processing capabilities.

Fully developed in Python, NEMO reads hyperstack TZCYX .tif(f) files, and can perform:

- Mesh segmentations
- Mesh curvature and inter-mesh thickness calculations
- Mesh projections
- Tangential nematic analyses
- Topological defect characterisations

The output consists of a folder structure containing the raw data and result figures, customisable and loadable by the
user at any step in the pipeline.

It has been tested on both Apple (MacBook Pro 16-inch, 2023) and Windows (Analysis PC @Roux Lab).

I am actively developing this pipeline, with new commits adding/changing modules in the backend.
If you come accross any issue, don't hesitate to post a GitLab issue or email me (konstantinos.andreadis@unige.ch) :)

---

## 📜 **Citation**

- **Citation Requirement**: If the Software is used directly or indirectly in any scientific publication or
  presentation, proper citation of the original work and author is required.

- Please cite the project as shown below:

**[!] publication incl. NEMO to be announced asap.  [!]**

---

## 📂 **Project Structure**

The project is structured as follows:

```
NEMO/  
├── module_scripts/                  # Folder of Module Scripts (Core Library)
│   ├── analysis.py                  # Image / Mesh Analysis Modules
│   ├── visuals.py                   # Visualisation Modules
│   └── datahandler.py               # Input / Output Data Handling Modules
├── automated_scripts/               # Folder of Automated Analysis Scripts
├── environment.yml                  # Required Python libraries compatible with any conda
├── MAIN.ipynb                       # Main Pipeline Workspace Notebook
├── nematic-on-vesicle.ipynb         # Generation of synthetic test example
└── README.md                        # This README
```  

---

## 🚀 **Getting Started**

### Prerequisites

- Make sure you have exactly <= **3.10** installed (to allow napari and trimesh to work correctly). You can check
  your version by running:

```bash
python --version
```  

### Installation

Clone this repository:

- Download the source code from GitHub
- Or extract the .zip file if sent privately.

Install the required dependencies in the virtual environment:

```bash
conda env create -f environment.yml
```

```bash
activate nemo-env
```

```bash
jupyter lab
```

---

## 📝 **Usage**

### Main Notebook

- The primary interface for interactive analysis is the **`MAIN.ipynb`**. You can open it using Jupyter:

```bash
jupyter notebook MAIN.ipynb
```

---

## 🧩 **Modular Scripts**

The **`modle_scripts/`** folder contains Python scripts with modular functions that can be (re-)combined for batch
analysis or
used individually within the main notebook.

Each script serves a different purpose, such as image/mesh analysis, data input/output handling, visualisation or
generation of test data.


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

© Konstantinos Andreadis 2024 (PhD @ Roux Lab & Salbreux Lab at UNIGE, Switzerland)