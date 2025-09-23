#!/usr/bin/env python
# coding: utf-8

"""
Fixed BioNeMo Geneformer Pipeline with LangGraph Generic Agent Architecture
This version uses ChatOllama with gpt-oss:20b as the agent
"""

import os, json, subprocess, shlex, time, pathlib, warnings
from typing import TypedDict, List, Optional, Dict, Any, Literal
import shutil
from collections import Counter
from pathlib import Path

# ---- LangGraph / LLM
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import ChatOllama
from langchain_core.messages import (
    AnyMessage, 
    HumanMessage, 
    SystemMessage, 
    AIMessage, 
    ToolCall, 
    ToolMessage,
    BaseMessage
)
from langchain_core.tools import tool
from IPython.display import Image, display
from langchain_core.runnables import RunnableConfig

# ---- Eval utils
import numpy as np
import scanpy as sc
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- ML evaluation
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    precision_recall_fscore_support
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ---------- Shared graph state ----------
class PipelineState(TypedDict):
    messages: List[BaseMessage]
    cfg: Dict[str, Any]
    artefacts: Dict[str, str]
    metrics: Dict[str, Any]
    labels: Optional[List[str]]
    preds: Optional[List[str]]

# ---------- Tool: convert h5ad → SCDL ----------
@tool
def convert_h5ad_to_scdl(h5ad_path: str, scdl_dir: str) -> str:
    """
    Convert a directory containing h5ad files to BioNeMo SCDL memmap directory
    
    Args:
        h5ad_path: Path to directory containing h5ad files
        scdl_dir: Output directory for SCDL format data
    
    Returns:
        JSON string with status and scdl_dir path
    """
    scdl_path = pathlib.Path(scdl_dir)
    
    # Check if already converted - look for the features directory as indicator
    features_dir = scdl_path / "features"
    if features_dir.exists() and any(features_dir.iterdir()):
        print(f"✓ SCDL directory already exists with features: {scdl_dir}")
        return json.dumps({"status": "success", "scdl_dir": str(scdl_dir), "message": "Already converted"})
    
    # Clean up if partial/empty directory exists
    if scdl_path.exists():
        print(f"Cleaning up incomplete SCDL directory: {scdl_dir}")
        shutil.rmtree(scdl_dir)
        time.sleep(0.5)
    
    # Create parent directory
    scdl_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = f"/usr/local/bin/convert_h5ad_to_scdl --data-path {shlex.quote(h5ad_path)} --save-path {shlex.quote(scdl_dir)}"
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ Conversion successful: {scdl_dir}")
        return json.dumps({"status": "success", "scdl_dir": str(scdl_dir)})
    except subprocess.CalledProcessError as e:
        print(f"✗ Conversion failed: {e.stderr}")
        return json.dumps({"status": "error", "message": str(e)})

# ---------- Tool: infer Geneformer ----------
@tool
def infer_geneformer(scdl_dir: str, checkpoint_path: str, results_path: str,
                     micro_batch_size: int = 8, seq_len: int = 2048,
                     num_workers: int = 8, num_gpus: int = 1) -> str:
    """
    Run BioNeMo Geneformer inference on SCDL dataset
    
    Args:
        scdl_dir: Path to SCDL format data directory
        checkpoint_path: Path to Geneformer model checkpoint
        results_path: Output path for inference results
        micro_batch_size: Batch size for inference
        seq_len: Maximum sequence length
        num_workers: Number of data loading workers
        num_gpus: Number of GPUs to use
    
    Returns:
        JSON string with status and results_path
    """
    results_file = pathlib.Path(results_path)
    
    # FIXED: Check if it's a directory and clean it up
    if results_file.exists():
        if results_file.is_dir():
            print(f"⚠️ Found directory instead of file at {results_path}, cleaning up...")
            shutil.rmtree(results_file)
            time.sleep(0.5)
        elif results_file.is_file():
            print(f"✓ Results already exist: {results_path}")
            return json.dumps({"status": "success", "results_path": str(results_path), "message": "Already inferred"})
    
    # Ensure output directory exists
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = (
        "/usr/local/bin/infer_geneformer "
        f"--data-dir {shlex.quote(scdl_dir)} "
        f"--checkpoint-path {shlex.quote(checkpoint_path)} "
        f"--results-path {shlex.quote(results_path)} "
        f"--micro-batch-size {micro_batch_size} --seq-len {seq_len} "
        f"--num-dataset-workers {num_workers} --num-gpus {num_gpus} --include-input-ids"
    )
    
    print(f"Running inference: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ Inference complete: {results_path}")
        return json.dumps({"status": "success", "results_path": str(results_path)})
    except subprocess.CalledProcessError as e:
        print(f"✗ Inference failed: {e.stderr}")
        return json.dumps({"status": "error", "message": str(e)})

# ---------- Helper function for ML benchmark ----------
def run_benchmark(data, labels, use_pca=True):
    """Run the accuracy, precision, recall, and F1-score benchmarks using MLP with cross-validation.

    Args:
        data: (R, C) contains the single cell expression (or whatever feature) in each row.
        labels: (R,) contains the string label for each cell
        use_pca: whether to fit PCA to the data.

    Returns:
        results_out: (dict) contains the accuracy, precision, recall, and F1-score for each class.
        conf_matrix: (R, R) contains the confusion matrix.
    """
    np.random.seed(1337)
    # Get input and output dimensions
    n_features = data.shape[1]
    hidden_size = 128

    # Define the target dimension 'n_components' for PCA
    n_components = min(10, n_features)  # ensure we don't try to get more components than features

    # Create a pipeline that includes scaling and MLPClassifier
    if use_pca:
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("projection", PCA(n_components=n_components)),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(hidden_size,),
                        max_iter=500,
                        random_state=1337,
                        early_stopping=True,  # Enable early stopping
                        validation_fraction=0.1,  # Use 10% of training data for validation
                        n_iter_no_change=50,  # Stop if validation score doesn't improve for 50 iterations
                        verbose=False,  # Print convergence messages
                    ),
                ),
            ]
        )
    else:
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(hidden_size,),
                        max_iter=500,
                        random_state=1337,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=50,
                        verbose=False,
                    ),
                ),
            ]
        )

    # Set up StratifiedKFold to ensure each fold reflects the overall distribution of labels
    cv = StratifiedKFold(n_splits=5)

    # Define the scoring functions
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, average="macro"),  # 'macro' averages over classes
        "recall": make_scorer(recall_score, average="macro"),
        "f1_score": make_scorer(f1_score, average="macro"),
    }

    # Track convergence warnings
    convergence_warnings = []
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings("always", category=ConvergenceWarning)

        # Perform stratified cross-validation with multiple metrics using the pipeline
        results = cross_validate(pipeline, data, labels, cv=cv, scoring=scoring, return_train_score=False)

        # Collect any convergence warnings
        convergence_warnings = [warn.message for warn in w if issubclass(warn.category, ConvergenceWarning)]

    # Print the cross-validation results
    print("Cross-validation metrics:")
    results_out = {}
    for metric, scores in results.items():
        if metric.startswith("test_"):
            metric_name = metric[5:]
            results_out[metric_name] = {
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "scores": scores.tolist()
            }
            print(f"{metric_name}: {scores.mean():.3f} (+/- {scores.std():.3f})")

    predictions = cross_val_predict(pipeline, data, labels, cv=cv)

    # Return confusion matrix and metrics.
    conf_matrix = confusion_matrix(labels, predictions)

    # Print convergence information
    if convergence_warnings:
        print("\nConvergence Warnings:")
        for warning in convergence_warnings:
            print(f"- {warning}")
    else:
        print("\nAll folds converged successfully")

    return results_out, conf_matrix

# ---------- Enhanced Tool: evaluate predictions with ML benchmark ----------
@tool
def evaluate_predictions(h5ad_path: str, scdl_dir: str, results_path: str,
                        label_key: str = "cell_type", out_dir: Optional[str] = None,
                        use_pca: bool = True, compare_baseline: bool = True) -> str:
    """
    Evaluate Geneformer predictions using MLP classifier with cross-validation.
    Optionally compares against log-normalized expression baseline.
    
    Args:
        h5ad_path: Path to h5ad data directory
        scdl_dir: Path to SCDL format data directory
        results_path: Path to Geneformer inference results
        label_key: Column name for cell type labels
        out_dir: Output directory for evaluation results
        use_pca: Whether to use PCA in evaluation
        compare_baseline: Whether to compare against baseline
    
    Returns:
        JSON string with status, report_path and metrics
    """
    try:
        out_dir = out_dir or os.path.join(os.path.dirname(results_path), "eval")
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        
        # Check if already evaluated
        report_path = os.path.join(out_dir, "benchmark_report.json")
        if os.path.exists(report_path):
            print(f"✓ Evaluation already exists: {report_path}")
            with open(report_path, 'r') as f:
                full_report = json.load(f)
            # Extract key metrics for state
            if "geneformer" in full_report:
                summary_metrics = {
                    "accuracy": full_report["geneformer"]["metrics"]["accuracy"]["mean"],
                    "macro_precision": full_report["geneformer"]["metrics"]["precision"]["mean"],
                    "macro_recall": full_report["geneformer"]["metrics"]["recall"]["mean"],
                    "macro_f1": full_report["geneformer"]["metrics"]["f1_score"]["mean"],
                    "num_classes": full_report["geneformer"]["num_classes"],
                    "num_samples": full_report["geneformer"]["num_samples"]
                }
            else:
                summary_metrics = {}
            return json.dumps({"status": "success", "report_path": report_path, "metrics": summary_metrics})
        
        # Check if results_path exists and handle directory case
        results_file = pathlib.Path(results_path)
        if not results_file.exists():
            return json.dumps({"status": "error", "message": f"Results not found: {results_path}"})
        
        if results_file.is_dir():
            # If it's a directory, look for predictions__rank_0.pt files
            pt_files = list(results_file.glob("predictions__rank_0.pt"))
            if not pt_files:
                # Fallback to any .pt file
                pt_files = list(results_file.glob("*.pt"))
            if pt_files:
                actual_results_path = str(pt_files[0])
                print(f"Found .pt file in directory: {actual_results_path}")
            else:
                return json.dumps({"status": "error", "message": f"No .pt files found in: {results_path}"})
        elif results_file.is_file():
            actual_results_path = results_path
        else:
            return json.dumps({"status": "error", "message": f"Results path is neither file nor directory: {results_path}"})
        
        # Find and load h5ad file
        h5ad_files = list(pathlib.Path(h5ad_path).glob("*.h5ad"))
        if not h5ad_files:
            return json.dumps({"status": "error", "message": f"No .h5ad files found in {h5ad_path}"})
        
        print(f"Loading data from {h5ad_files[0]}")
        adata = sc.read_h5ad(h5ad_files[0])
        
        if label_key not in adata.obs.columns:
            return json.dumps({"status": "error", "message": f"Label key '{label_key}' not found"})
        
        # Get labels
        labels = adata.obs[label_key].values
        
        # Encode labels to integers for sklearn
        label_encoder = LabelEncoder()
        integer_labels = label_encoder.fit_transform(labels)
        
        # Load Geneformer embeddings
        print(f"Loading predictions from {actual_results_path}")
        blob = torch.load(actual_results_path, map_location="cpu")
        
        # Extract embeddings (feature representations from Geneformer)
        if "embeddings" in blob:
            embeddings = blob["embeddings"].float().cpu().numpy()
        elif "logits" in blob:
            # Fallback to using logits if embeddings not available
            embeddings = np.asarray(blob["logits"])
        else:
            return json.dumps({"status": "error", "message": "Neither embeddings nor logits found in results"})
        
        # Ensure alignment
        min_len = min(len(labels), len(embeddings))
        embeddings = embeddings[:min_len]
        integer_labels = integer_labels[:min_len]
        
        print(f"\nData shape: {embeddings.shape}")
        print(f"Number of samples: {min_len}")
        print(f"Number of classes: {len(np.unique(integer_labels))}")
        
        # Analyze label distribution
        label_counts = Counter(labels[:min_len])
        print("\nCell type distribution:")
        for cell_type, count in label_counts.most_common():
            print(f"  {cell_type}: {count} ({100*count/min_len:.1f}%)")
        
        # Run benchmark on Geneformer embeddings
        print("\n" + "="*60)
        print("Evaluating Geneformer embeddings with MLP classifier")
        print("="*60)
        geneformer_results, geneformer_cm = run_benchmark(embeddings, integer_labels, use_pca=use_pca)
        
        full_report = {
            "geneformer": {
                "metrics": geneformer_results,
                "confusion_matrix": geneformer_cm.tolist(),
                "num_samples": min_len,
                "num_classes": len(label_encoder.classes_),
                "class_names": label_encoder.classes_.tolist(),
                "class_distribution": dict(label_counts),
                "embedding_dim": embeddings.shape[1],
                "used_pca": use_pca
            }
        }
        
        # Optionally compare with baseline (log-normalized expression)
        if compare_baseline and hasattr(adata, 'X'):
            print("\n" + "="*60)
            print("Evaluating baseline (log-normalized expression) with MLP classifier")
            print("="*60)
            
            # Get raw expression and normalize
            raw_X = np.asarray(adata.X.todense()) if hasattr(adata.X, 'todense') else np.asarray(adata.X)
            raw_X = raw_X[:min_len]
            
            # Log-normalize (add pseudocount for stability)
            normed_X = (raw_X + 1) / raw_X.sum(axis=1, keepdims=True)
            logp1_X = np.log(normed_X)
            
            baseline_results, baseline_cm = run_benchmark(logp1_X, integer_labels, use_pca=use_pca)
            
            full_report["baseline_logp1"] = {
                "metrics": baseline_results,
                "confusion_matrix": baseline_cm.tolist(),
                "feature_dim": logp1_X.shape[1],
                "used_pca": use_pca
            }
            
            # Compare performance
            print("\n" + "="*60)
            print("Performance Comparison")
            print("="*60)
            print(f"{'Metric':<15} {'Geneformer':<20} {'Baseline (log+1)':<20}")
            print("-" * 55)
            for metric in ["accuracy", "precision", "recall", "f1_score"]:
                gf_val = geneformer_results[metric]["mean"]
                gf_std = geneformer_results[metric]["std"]
                bl_val = baseline_results[metric]["mean"]
                bl_std = baseline_results[metric]["std"]
                print(f"{metric:<15} {gf_val:.3f} (±{gf_std:.3f})    {bl_val:.3f} (±{bl_std:.3f})")
        
        # Save confusion matrix plots
        if True:  # Set to True to save plots
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # Plot Geneformer confusion matrix
            plt.figure(figsize=(12, 10))
            normalized_cm = geneformer_cm / geneformer_cm.sum(axis=0)
            sns.heatmap(
                normalized_cm,
                cmap=sns.color_palette("Blues", as_cmap=True),
                vmin=0,
                vmax=1,
                linewidth=0.1,
                linecolor="lightgrey",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                cbar_kws={'label': 'Normalized Accuracy'}
            )
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.title("Geneformer - Normalized Confusion Matrix")
            plt.tight_layout()
            cm_path = os.path.join(out_dir, "geneformer_confusion_matrix.png")
            plt.savefig(cm_path, dpi=150)
            plt.close()
            print(f"\nSaved confusion matrix plot: {cm_path}")
            
            full_report["geneformer"]["confusion_matrix_plot"] = cm_path
        
        # Save comprehensive report
        with open(report_path, "w") as f:
            json.dump(full_report, f, indent=2)
        
        print(f"\n✓ Comprehensive evaluation complete: {report_path}")
        
        # Create summary for return
        summary_metrics = {
            "accuracy": geneformer_results["accuracy"]["mean"],
            "macro_precision": geneformer_results["precision"]["mean"],
            "macro_recall": geneformer_results["recall"]["mean"],
            "macro_f1": geneformer_results["f1_score"]["mean"],
            "num_classes": len(label_encoder.classes_),
            "num_samples": min_len,
            "cv_folds": 5
        }
        
        return json.dumps({"status": "success", "report_path": report_path, "metrics": summary_metrics})
        
    except Exception as e:
        import traceback
        error_msg = f"Error in evaluation: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return json.dumps({"status": "error", "message": error_msg})

# ---------- Initialize LLM and bind tools ----------
# Create the LLM instance
llm = ChatOllama(model="gpt-oss:20b", temperature=0)

# Create tools list
tools = [convert_h5ad_to_scdl, infer_geneformer, evaluate_predictions]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# Create tool node
tool_node = ToolNode(tools)

# ---------- System prompt for the agent ----------
SYSTEM_PROMPT = """You are an AI agent managing a BioNeMo Geneformer pipeline for cell type classification.

Your task is to execute a 3-step pipeline:
1. Convert H5AD files to SCDL format
2. Run Geneformer inference on the SCDL data
3. Evaluate predictions using MLP classifier with cross-validation

The pipeline configuration is provided in the state. You should:
- Check which steps have been completed by examining the artifacts
- Execute the next required step
- Update the state after each successful tool execution
- Report the final results when all steps are complete

Current artifacts status will be available in the state:
- scdl_dir: Path to SCDL converted data (step 1 complete)
- results_pt: Path to inference results (step 2 complete)  
- report_json: Path to evaluation report (step 3 complete)

Execute tools sequentially and monitor for successful completion.
If a tool returns an error, you may retry or report the issue.
"""

# ---------- Generic Agent Node (following LangChain Academy pattern) ----------
def agent(state: PipelineState) -> Dict[str, Any]:
    """
    Generic agent that uses LLM to decide next actions.
    Following the pattern from LangChain Academy module-1/agent.ipynb
    """
    # Get configuration and current artifacts
    cfg = state.get("cfg", {})
    artifacts = state.get("artefacts", {})
    
    # Build context message about current state
    state_summary = f"""
        Current Pipeline State:
        - Configuration loaded: {bool(cfg)}
        - SCDL conversion complete: {'scdl_dir' in artifacts}
        - Inference complete: {'results_pt' in artifacts}
        - Evaluation complete: {'report_json' in artifacts}
        
        Configuration:
        - H5AD Path: {cfg.get('h5ad_path', 'Not set')}
        - SCDL Dir: {cfg.get('scdl_dir', 'Not set')}
        - Checkpoint: {cfg.get('checkpoint_path', 'Not set')}
        - Results Path: {cfg.get('results_path', 'Not set')}
        
        Artifacts:
        {json.dumps(artifacts, indent=2)}
        """
    
    # Prepare messages for the LLM
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    
    # Add the state summary as a user message if this is the first run
    if len(state.get("messages", [])) == 1:  # Only initial human message
        messages.append(HumanMessage(content=f"Execute the pipeline with the following state:\n{state_summary}"))
    else:
        # Include recent messages but keep context manageable
        recent_messages = state.get("messages", [])[-5:]  # Last 5 messages
        for msg in recent_messages:
            if not isinstance(msg, SystemMessage):  # Skip system messages
                messages.append(msg)
        
        # Add current state update
        messages.append(HumanMessage(content=f"Current state update:\n{state_summary}"))
    
    # Invoke the LLM with tools
    response = llm_with_tools.invoke(messages)
    
    # Return the response to be added to messages
    return {"messages": [response]}

# ---------- State Updater Node ----------
def update_state(state: PipelineState) -> Dict[str, Any]:
    """
    Process tool results and update artifacts.
    This node runs after tools to update the state based on tool outputs.
    """
    messages = state["messages"]
    
    # Create new dictionaries to ensure updates persist
    artifacts = dict(state.get("artefacts", {}))
    metrics = dict(state.get("metrics", {}))
    
    # Look for the most recent ToolMessage (tool response)
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            try:
                if isinstance(msg.content, str) and msg.content.strip().startswith('{'):
                    result = json.loads(msg.content)
                    
                    if result.get("status") == "success":
                        # Update based on what the tool returned
                        if "scdl_dir" in result:
                            artifacts["scdl_dir"] = result["scdl_dir"]
                            print(f"→ State updated: scdl_dir = {result['scdl_dir']}")
                            break
                        elif "results_path" in result:
                            artifacts["results_pt"] = result["results_path"]
                            print(f"→ State updated: results_pt = {result['results_path']}")
                            break
                        elif "report_path" in result:
                            artifacts["report_json"] = result["report_path"]
                            if "metrics" in result:
                                metrics = result["metrics"]
                            print(f"→ State updated: report_json = {result['report_path']}")
                            break
                    elif result.get("status") == "error":
                        print(f"✗ Tool error: {result.get('message', 'Unknown error')}")
                        
                        # Special handling for directory error in inference step
                        if "[Errno 21] Is a directory" in result.get("message", "") and "results_pt" in artifacts:
                            print(f"→ Removing invalid results_pt from state to retry inference")
                            del artifacts["results_pt"]
                        break
            except (json.JSONDecodeError, AttributeError) as e:
                continue
    
    print(f"  Current artifacts: {list(artifacts.keys())}")
    return {"artefacts": artifacts, "metrics": metrics}

# ---------- Routing function ----------
def should_continue(state: PipelineState) -> Literal["tools", "__end__"]:
    """
    Determine whether to continue with tools or end.
    This is used instead of tools_condition for more control.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if pipeline is complete
    artifacts = state.get("artefacts", {})
    if all(k in artifacts for k in ["scdl_dir", "results_pt", "report_json"]):
        return "__end__"
    
    # Check for tool calls in the last message
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # Check for max iterations safety
    tool_call_count = sum(1 for msg in messages 
                         if hasattr(msg, 'tool_calls') and msg.tool_calls)
    if tool_call_count > 15:
        print(f"⚠️ Safety limit reached: {tool_call_count} tool calls")
        return "__end__"
    
    return "__end__"

# ---------- Build the graph using generic agent pattern ----------
def build_pipeline_graph():
    """
    Build the pipeline graph following the generic agent architecture pattern
    from https://github.com/langchain-ai/langchain-academy/blob/main/module-1/agent.ipynb
    """
    # Create the graph with our state schema
    graph = StateGraph(PipelineState)
    
    # Add nodes
    graph.add_node("agent", agent)  # The LLM-based agent
    graph.add_node("tools", tool_node)  # Tool execution
    graph.add_node("update_state", update_state)  # State updater
    
    # Define the flow
    graph.add_edge(START, "agent")
    
    # Use conditional edge from agent
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "__end__": END
        }
    )
    
    # After tools, update state then go back to agent
    graph.add_edge("tools", "update_state")
    graph.add_edge("update_state", "agent")
    
    # Compile with memory
    return graph.compile(checkpointer=MemorySaver())

# ---------- Main execution ----------
if __name__ == "__main__":
    path="/home/sheng/data/bionemo/data/"
    
    notebook_workdir = Path(path) / "notebook_tutorials" / "geneformer_celltype_classification"
    notebook_workdir.mkdir(parents=True, exist_ok=True)
    
    input_dir = notebook_workdir / "celltype-bench-dataset-input"
    data_dir = notebook_workdir / "celltype-bench-dataset"
    result_path = notebook_workdir / "results_106M_240530_nemo2_with_llm_agents.pt"
    
    # Optional: Clean up previous runs
    CLEANUP = True  # Set to True to clean up the problematic directory
    if CLEANUP:
        # Clean up the problematic results directory/file
        if result_path.exists():
            if result_path.is_dir():
                shutil.rmtree(result_path)
                print(f"Cleaned directory: {result_path}")
            else:
                result_path.unlink()
                print(f"Cleaned file: {result_path}")
        
        # Optionally clean other artifacts
        if False:  # Set to True for full cleanup
            if data_dir.exists():
                shutil.rmtree(data_dir)
                print(f"Cleaned: {data_dir}")
            eval_dir = notebook_workdir / "eval"
            if eval_dir.exists():
                shutil.rmtree(eval_dir)
                print(f"Cleaned: {eval_dir}")
    
    # Pipeline configuration
    cfg = {
        "h5ad_path": str(input_dir),
        "scdl_dir": str(data_dir),
        "checkpoint_path": "/home/sheng/data/bionemo/models/7d67a526379eb8581f2aaaf03425ae9ec81a38570b24ddc8b22818e5d26ea772-geneformer_106M_240530_nemo2.tar.gz.untar",
        "results_path": str(result_path),
        "label_key": "cell_type",
        "micro_batch_size": 8,
        "seq_len": 2048,
        "num_workers": 8,
        "num_gpus": 1,
        "use_pca": True,  # Whether to use PCA in evaluation
        "compare_baseline": True,  # Whether to compare against log-normalized expression
    }
    
    print("\n" + "="*60)
    print("Generic Agent Architecture Pipeline Configuration")
    print("="*60)
    print(f"LLM Model: gpt-oss:20b (via ChatOllama)")
    print(f"Temperature: 0")
    print("-"*60)
    for key, value in cfg.items():
        print(f"{key:20s}: {value}")
    print("="*60)
    
    # Build the graph
    app = build_pipeline_graph()
    
    # Visualize the graph
    try:
        display(Image(app.get_graph(xray=True).draw_mermaid_png()))
    except Exception as e:
        print(f"Could not display graph: {e}")
    
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content="Execute the BioNeMo Geneformer pipeline for cell type classification")],
        "cfg": cfg,
        "artefacts": {},
        "metrics": {},
        "labels": None,
        "preds": None,
    }
    
    # Configuration for the run
    config = RunnableConfig(
        configurable={"thread_id": f"run_{int(time.time())}"},
        recursion_limit=30  # Allow sufficient recursion for retries
    )
    
    print("\nStarting pipeline with Generic Agent Architecture...")
    print("-" * 60)
    
    try:
        # Invoke the graph
        final_state = app.invoke(initial_state, config=config)
        
        # Print final results
        print("\n" + "="*60)
        print("✅ Pipeline Complete with Generic Agent Architecture")
        print("="*60)
        
        # Display final metrics if available
        if final_state.get("metrics"):
            print("\nFinal Results (5-fold Cross-Validation):")
            metrics = final_state["metrics"]
            print(f"• Accuracy: {metrics.get('accuracy', 0):.3f}")
            print(f"• Macro Precision: {metrics.get('macro_precision', 0):.3f}")
            print(f"• Macro Recall: {metrics.get('macro_recall', 0):.3f}")
            print(f"• Macro F1: {metrics.get('macro_f1', 0):.3f}")
            print(f"• Classes: {metrics.get('num_classes', 0)}")
            print(f"• Samples: {metrics.get('num_samples', 0)}")
        
        # Display artifacts
        if final_state.get("artefacts"):
            print("\nGenerated Artifacts:")
            for key, path in final_state["artefacts"].items():
                print(f"• {key}: {path}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()