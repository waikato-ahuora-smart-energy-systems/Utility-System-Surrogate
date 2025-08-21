# Surrogate Modelling of Industrial Steam Systems

Data analysis and surrogate model development of industrial utility system

The current version is 0.5. The full version will involve 
- Combined bulk and marginal steam cost calculation methods using surrogate model
- Code refactor for removing redundancy. Likely as a generic class that can be imported
- User guides & documentation
- Package through pip
- Integration into the Ahuora Platform


## 🚀 Installation (via Conda + setup.py)

Follow these steps to install and run OpenHENS using a Conda environment.

### 1. Clone the Repository

```bash
git clone https://github.com/waikato-ahuora-smart-energy-systems/Utility-System-Surrogate 
cd Utility-System-Surrogate
```

### 2. Install Miniconda (if not already installed)

Download and install **Miniconda** from:  
https://docs.conda.io/en/latest/miniconda.html

> During setup, check the box to "Add Miniconda to my PATH environment variable" if you want to use it from any terminal.

Once installed, open **Anaconda Prompt** (Windows) or terminal (macOS/Linux). Do not use virtual environments as they dont work with packages outside of Python

---

### 3. Create and Activate a Conda Environment

```bash
conda create -n surrogate-env python=3.12
conda activate surrogate-env
```

### 4. Install the Package (Using setup.py)

From the project root:

```bash
pip install -e .
```

This installs the `utility-system-surrogate` package in **editable mode** and uses `requirements.txt` automatically.


## Deleting the Conda Environment

To delete the environment:

```bash
conda deactivate
conda remove -n surrogate-env --all
```

This will **not affect** any other Conda environments or your base Python install.

---

# Citation

Please cite this work as:

```shell
Utilty-System-Surrogate v0.5
Ahuora Centre for Smart Energy Systems https://www.waikato.ac.nz/research/institutes-centres-entities/centres/ahuora-centre-for-smart-energy-systems/
https://github.com/waikato-ahuora-smart-energy-systems/Utility-System-Surrogate
```

## 💡 Notes

- If using **VSCode**, make sure to install the **Python extension by Microsoft**, and select the `surrogate-env` interpreter.

---
