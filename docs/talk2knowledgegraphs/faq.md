# T2KG - Frequently Asked Questions (FAQs)

---

### 1. How can I extract a subgraph relevant to my IBD query?
Use the `multimodal_subgraph_extraction_tool` to extract a subgraph from the knowledge graph containing information relevant to your IBD-related question. Specify keywords, entities, or relationships of interest.

**Example:**
```
Extract a subgraph about cytokine signaling pathways in inflammatory bowel disease (IBD).
```

---

### 2. How do I summarize the extracted IBD subgraph?
After extracting a subgraph, use the `subgraph_summarization_tool` to get a concise summary of the most important nodes and edges related to IBD. This helps you quickly understand the key information.

**Example:**
```
Summarize the extracted subgraph on TNF-alpha signaling in IBD.
```

---

### 3. How can I reason over the extracted IBD subgraph to answer specific questions?
Use the `graphrag_reasoning` tool to answer questions based on the extracted and summarized IBD subgraph. This tool combines subgraph context and any relevant documents or chat history to provide thorough answers.

**Example:**
```
Based on the summarized subgraph, what are the key regulators of intestinal inflammation in IBD?
```

---

### 4. Can I query using multimodal data in the knowledge graph for extracting a subgraph?

Yes, the `multimodal_subgraph_extraction_tool` supports queries involving multimodal data existed in the biomedical knowledge graph. For instance, gene/protein node has protein sequence embedding or drug has SMILES embedding in addition to their textual description.

**Example:**

User can upload an excel file consisting of a list of genes. Please see an example in the following path:
```
aiagents4pharma/talk2knowledgegraphs/tests/files/multimodal-analysis_sample_genes.xlsx
```
And then user can put the following prompt:
```
Extract an IBD-related subgraph by using the listed genes in the uploaded file.
```

---

### 5. How do I follow up with additional questions about the same IBD subgraph?

Ask follow-up questions using the `subgraph_summarization_tool` and `graphrag_reasoning` tools. The agent uses the most recent IBD subgraph and its summary to answer your queries.

**Examples:**
```
What are the most connected nodes in the current IBD subgraph?
```

```
How does the expression of IL-6 affect the pathway in the extracted IBD subgraph?
```

---

### 6. How do I compare different IBD subgraphs or reasoning results?

Extract and summarize multiple IBD subgraphs, then use `graphrag_reasoning` to compare their properties or the answers to your questions.

**Example:**
```
Extract and summarize subgraphs for both Crohn's disease and ulcerative colitis pathways. Compare their key regulators.
```

---

### 7. Can I use uploaded files or previous chat history in IBD reasoning?

Yes, the agent can incorporate uploaded files and previous chat history when using the `graphrag_reasoning` tool to provide more contextually relevant answers for IBD research.

**Example:**
```
Use the uploaded clinical study and the current subgraph to explain the role of JAK1 in IBD.
```
