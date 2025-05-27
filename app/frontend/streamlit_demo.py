import streamlit as st
import networkx as nx
import gravis as gv
import streamlit.components.v1 as components
import re

def extract_inner_html(html):
    match = re.search(r"<body[^>]*>(.*?)</body>", html, re.DOTALL)
    return match.group(1) if match else html

figures_inner_html = ""

for i in range(6):
    G = nx.DiGraph()
    G.add_node("A", label=f"A{i}")
    G.add_node("B", label=f"B{i}")
    G.add_node("C", label=f"C{i}")
    G.add_edge("A", "B", label="A→B")
    G.add_edge("B", "C", label="B→C")
    G.add_edge("C", "A", label="C→A")

    fig = gv.d3(
        G,
        node_size_factor=3.0,
        show_edge_label=True,
        edge_label_data_source="label",
        edge_curvature=0.25,
        zoom_factor=1.0,
        many_body_force_strength=-500,
        many_body_force_theta=0.3,
        node_hover_neighborhood=True,
    )

    inner_html = extract_inner_html(fig.to_html())
    wrapped_html = f'''
    <div class="graph-content">
        {inner_html}
    </div>
    '''

    figures_inner_html += f'''
    <div class="graph-box">
        <h3 class="graph-title">Graph #{i+1}</h3>
        {wrapped_html}
    </div>
    '''

full_html = f"""
<!DOCTYPE html>
<html>
<head>
  <style>
    html, body {{
        margin: 0;
        padding: 0;
        overflow-y: hidden;
        height: 100%;
    }}
    .scroll-container {{
        display: flex;
        overflow-x: auto;
        overflow-y: hidden;
        gap: 1rem;
        padding: 1rem;
        background: #f5f5f5;
        height: 100%;
        box-sizing: border-box;
    }}
    .graph-box {{
        flex: 0 0 auto;
        width: 500px;
        height: 515px;
        border: 1px solid #ccc;
        border-radius: 8px;
        background: white;
        padding: 0.5rem;
        box-sizing: border-box;
        position: relative;
        display: flex;
        flex-direction: column;
        align-items: center;
    }}
    .graph-title {{
        margin: 0 0 16px 0;  /* Increased bottom margin */
        font-family: Arial, sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        text-align: center;
    }}
    .graph-content {{
        width: 100%;
        flex-grow: 1;
    }}
    .graph-box svg, .graph-box canvas {{
        max-width: 100% !important;
        max-height: 100% !important;
        height: 100% !important;
        width: 100% !important;
    }}
  </style>
</head>
<body>
  <div class="scroll-container">
    {figures_inner_html}
  </div>
</body>
</html>
"""

components.html(full_html, height=550, scrolling=False)