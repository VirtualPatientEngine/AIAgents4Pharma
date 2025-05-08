# Miniconda and FAISS Installation Guide

This guide provides instructions for setting up a Python environment using Miniconda and installing FAISS for similarity search and clustering.

---

## Step 1: Install Miniconda

Follow the official Quickstart install instructions provided by Anaconda:

[Official Link](https://www.anaconda.com/docs/getting-started/miniconda/install)

Scroll down to the **"Quickstart install instructions"** section. It contains platform-specific setup guides for:

- Windows
- macOS (Intel and Apple Silicon)
- Linux

Make sure to follow the steps based on your operating system.

---

## Step 2: Set Up a Conda Environment

Once Miniconda is installed, use the following command to create and activate a new environment and install required Python packages:

```bash
conda create --name AIAgents4Pharma python=3.12 -y && conda activate AIAgents4Pharma && pip install --upgrade pip && pip install -r requirements.txt
```

This command:

- Creates a new environment named `AIAgents4Pharma`
- Installs Python 3.12
- Activates the environment
- Upgrades `pip`
- Installs required packages from a `requirements.txt` file

---

## Step 3: Install FAISS

Follow the official FAISS installation instructions [here](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)

---

## Step 4: Verify Installation

To verify that FAISS is correctly installed, run the following in a Python shell:

```python
import faiss
print(faiss.__version__)
```

If FAISS is installed properly, it will print the version without any errors.

---

## Resources

- [Miniconda Installation Guide](https://www.anaconda.com/docs/getting-started/miniconda/install)
- [FAISS GitHub Repository](https://github.com/facebookresearch/faiss)
- [FAISS Installation Instructions](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)
