**Talk2KnowledgeGraphs** is an AI agent designed to interact with biomedical knowledge graphs. Biomedical knowledge graphs contains crucial information in the form of entities (nodes) and their relationships (edges). These graphs are used to represent complex biological systems, such as metabolic pathways, protein-protein interactions, and gene regulatory networks. In order to easily interact with this information, Talk2KnowledgeGraphs uses natural language processing (NLP) to enable users to ask questions and make requests. By simply asking questions or making requests, users can:

- Dataset loading: load knowledge graph from datasets.
- Embedding: embed entities and relationships in the knowledge graph.
- Knowledge graph construction: construct a knowledge graph from dataframes.
- Subgraph extraction: extract subgraphs from the initial knowledge graph.
- Retrieval: retrieve information from the (sub-) knowledge graph.
- Reasoning: reason over the (sub-) knowledge graph.
- Visualization: visualize the (sub-) knowledge graph.

## Installation

### Prerequisites
- Python 3.10 or higher

### Installing Talk2KnowledgeGraphs in two ways

#### Option 1: Git

1. Clone the repository:

    git clone https://github.com/<your-repo>/ aiagents4pharma.git
    cd aiagents4pharma/talk2knowledgegraphs

2. Install the package and its dependencies:

    pip install .

3. Alternatively, install from source:

    pip install -e .


#### Option 2: PyPI *(coming soon)*
   ```bash
   pip install aiagents4pharma
   ```
