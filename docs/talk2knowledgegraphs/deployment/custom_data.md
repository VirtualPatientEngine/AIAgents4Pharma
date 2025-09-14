# 🚀 Knowledge Graph Preparation for Talk2KnowledgeGraphs (T2KG)

## 📌 Overview

By default, **Talk2KnowledgeGraphs (T2KG)** includes a small subset of the **PrimeKG** knowledge graph focused on **inflammatory bowel disease (IBD)**. This subset is enriched with multimodal biomedical metadata and embedded node/edge representations, powered by [BioBridge](https://github.com/RyanWangZf/BioBridge) and [StarkQA](https://github.com/snap-stanford/stark).

These default files are available at:

```bash
aiagents4pharma/talk2knowledgegraphs/tests/files/biobridge_multimodal
```

If you'd like to use a **different disease-specific graph** or build your own **custom PrimeKG graph**, follow the step-by-step instructions below.

---

## 🧰 Preparing Your Local Environment

Before preprocessing your custom knowledge graph, you must set up your local environment.
Please follow the general setup instructions in the repository's [main README](https://virtualpatientengine.github.io/AIAgents4Pharma/).

### ✅ Prerequisites

After installing the required Python packages, make sure you have the following:

* ✅ **OpenAI API Key** — for generating text embeddings.
* ✅ **NVIDIA API Key** — for creating a NIM instance.
* ✅ **NVIDIA NIM for MolMIM** — for embedding drug SMILES representations.

➡️ Refer to this notebook to enable MolMIM-based SMILES embedding:
`AIAgents4Pharma/aiagents4pharma/docs/notebooks/talk2knowledgegraphs/tutorial_primekg_smiles_enrich_embed.ipynb`

---

## 🏗️ Constructing a Custom PrimeKG Graph

T2KG supports both **disease-specific** and **full PrimeKG** multimodal knowledge graphs.

---

### 🔹 Disease-Specific Multimodal Graph

You can filter and process subgraphs from PrimeKG using:

* [🧬 IBD-Specific PrimeKG Subgraph](https://virtualpatientengine.github.io/AIAgents4Pharma/notebooks/talk2knowledgegraphs/tutorial_biobridge_ibd_multimodal/)
  → Generates a focused graph for **IBD** with enriched and embedded node/edge features.

* [📤 Migrate IBD Data to Milvus](https://virtualpatientengine.github.io/AIAgents4Pharma/notebooks/talk2knowledgegraphs/tutorial_primekg_milvus_ibd_primekg_dump)
  → Prepares and formats the dataframes for Milvus ingestion.
  *(Tip: You only need to follow steps up to storing the dataframes as Parquet files.)*

---

### 🔹 Full PrimeKG Multimodal Graph

For processing the **complete PrimeKG**, use:

* [🔬 BioBridge-PrimeKG Multimodal](https://virtualpatientengine.github.io/AIAgents4Pharma/notebooks/talk2knowledgegraphs/tutorial_biobridge_primekg_multimodal/)
  → Utilizes preloaded multimodal BioBridge data to enrich PrimeKG.

* [📚 PrimeKG Enrichment Pipeline](https://virtualpatientengine.github.io/AIAgents4Pharma/notebooks/talk2knowledgegraphs/tutorial_primekg_enrichment/)
  → Enriches and Embeds the entire PrimeKG using BioBridge, MolMIM, and textual embeddings.

* [📤 Migrate Full PrimeKG to Milvus](https://virtualpatientengine.github.io/AIAgents4Pharma/notebooks/talk2knowledgegraphs/tutorial_primekg_milvus_primekg_dump)
  → Formats and dumps the full graph into Milvus-ready Parquet files.
  *(Tip: You only need to follow steps up to storing the dataframes as Parquet files.)*
---

## ▶️ Running T2KG with Your Custom Graph

### 1. Copy the Environment Template

```bash
cp aiagents4pharma/talk2knowledgegraphs/.env.example .env
```

### 2. Set Environment Variables

Edit the `.env` file to match your custom setup. Most importantly, set your custom data directory:

```env
...
DATA_DIR=/absolute/path/to/your/data/
...
```

---

### 3. Ensure Correct Folder Structure

T2KG expects the following folder structure inside your data directory:

```
project/
├── edges/
│   ├── embedding/
│   │   ├── edges_0.parquet.gzip
│   │   ├── edges_1.parquet.gzip
│   │   └── ...
│   └── enrichment/
│       └── edges.parquet.gzip
├── nodes/
│   ├── embedding/
│   │   ├── biological_process.parquet.gzip
│   │   ├── cellular_component.parquet.gzip
│   │   └── ...
│   └── enrichment/
│       ├── biological_process.parquet.gzip
│       ├── cellular_component.parquet.gzip
│       └── ...
```

This layout ensures that T2KG can properly load and query your graph content using Milvus database.

---

## 🧠 Launching the T2KG Interface

Once your environment and data are ready, you can launch T2KG and start interacting with your graph using natural language!

You can either:

* 🐳 **Use Docker** (recommended for easy deployment), or
* 🖥️ **Run Milvus and Streamlit manually**

For more information, you can find various ways to launching of the app [here](https://virtualpatientengine.github.io/AIAgents4Pharma/)
and [here](https://virtualpatientengine.github.io/AIAgents4Pharma/talk2knowledgegraphs/deployment/deployment/)
