# NEMO 🐠
## 🔬 Nematics and Morphology Image Analysis Toolkit  

*Konstantinos Andreadis*  
PhD Student @ University of Geneva  
Aurélien Roux Lab & Guillaume Salbreux Lab 

---

Welcome to **NEMO**, the **Nematics and Morphology Image Analysis Toolkit**. 
This toolkit is designed to provide efficient and modular image analysis tools, enabling the study of nematics and morphological features through a combination of customizable scripts and batch processing capabilities.  

---

## 📂 **Project Structure**  

The project is structured as follows:  

```
NEMO/  
├── scripts/                    # Folder of Module Scripts
│   ├── analysis.py             # Image / Mesh Analysis Modules
│   ├── visuals.py              # Visualisation Modules
│   ├── datahandler.py          # Input / Output Data Handling Modules
│   └── simulation.py           # Simulation Modules
├── debugging/                  # Folder of Debugging Scripts
│   ├── fix_dependencies.ipynb  # Dependency Fix Notebook
│   ├── sandbox.py              # Debugging Sandbox Notebook
│   └── generator.py            # Simulation Notebook
├── main.ipynb                  # Main Pipeline Workspace Notebook
├── batch_analysis.py           # Batch Analysis Script
├── requirements.txt            # Required Python Libraries
└── README.md                   # This File
```  

---

## 🚀 **Getting Started**  

### Prerequisites  

- Make sure you have at least Python 3.9.6 installed. You can check your version by running:  

    ```bash
    python --version
    ```  

### Installation  

1. Clone this repository:

- If a Roux Lab Member: Go to ```NAS/...```

2. Install the required dependencies:  

    ```bash
    pip install -r requirements.txt  
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

### Batch Analysis  

- The **`batch_analysis.py`** script allows batch processing by combining multiple modules. 
- To use it, simply configure your desired pipeline within the script and run:  

    ```bash
    python batch_analysis.py  
    ```  

You can customize the pipeline by combining modules from the **`scripts/`** folder.  

---

## 🧩 **Modular Scripts**  

The **`scripts/`** folder contains Python scripts with modular functions that can be (re-)combined for batch analysis or used individually within the main notebook. 
Each script serves a different purpose, such as image/mesh analysis, data input/output handling, visualisation or generation of test data.

### Adding New Modules  

To add a new module:  

1. Create a new Python script in the **`scripts/`** folder.  
2. Define a main function that takes input and outputs results.  
3. Ensure the script follows the expected input-output format for pipeline integration.  

---

## 🪲 **Debugging**  

- The **`debugging/`** folder contains scripts and logs that help diagnose issues. 
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

Contributions are welcome! Please fork the repository and submit a pull request with your improvements. 
Make sure to follow the existing code style and include appropriate documentation.  

---

## 📄 **License**

MIT License with Citation Requirement

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

- **Citation Requirement**: If the Software is used directly or indirectly in any scientific publication or presentation, proper citation of the original work is required. 

- Please cite the project as follows: t.b.a.
 