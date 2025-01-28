#!/usr/bin/env python3

'''
Utils for Streamlit.
'''

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
from langsmith import Client
import networkx as nx
import gravis

def submit_feedback(user_response):
    '''
    Function to submit feedback to the developers.
    '''
    client = Client()
    client.create_feedback(
        st.session_state.run_id,
        key="feedback",
        score=1 if user_response['score'] == "👍" else 0,
        comment=user_response['text']
    )
    st.info("Your feedback is on its way to the developers. Thank you!", icon="🚀")

def render_table_plotly(uniq_msg_id, content, df_selected):
    """
    Function to render the table and plotly chart in the chat.

    Args:
        uniq_msg_id: str: The unique message id
        msg: dict: The message object
        df_selected: pd.DataFrame: The selected dataframe
    """
    # Display the toggle button to suppress the table
    render_toggle(
        key="toggle_plotly_"+uniq_msg_id,
        toggle_text="Show Plot",
        toggle_state=True,
        save_toggle=True)
    # Display the plotly chart
    render_plotly(
        df_selected,
        key="plotly_"+uniq_msg_id,
        title=content,
        # tool_name=msg.name,
        # tool_call_id=msg.tool_call_id,
        save_chart=True)
    # Display the toggle button to suppress the table
    render_toggle(
        key="toggle_table_"+uniq_msg_id,
        toggle_text="Show Table",
        toggle_state=False,
        save_toggle=True)
    # Display the table
    render_table(
        df_selected,
        key="dataframe_"+uniq_msg_id,
        # tool_name=msg.name,
        # tool_call_id=msg.tool_call_id,
        save_table=True)
    st.empty()

def render_toggle(key: str,
                  toggle_text: str,
                  toggle_state: bool,
                  save_toggle: bool = False):
    """
    Function to render the toggle button to show/hide the table.
    """
    st.toggle(
        toggle_text,
        toggle_state,
        help='''Toggle to show/hide data''',
        key=key
        )
    # print (key)
    if save_toggle:
        # Add data to the chat history
        st.session_state.messages.append({
                "type": "toggle",
                "content": toggle_text,
                "toggle_state": toggle_state,
                "key": key
            })

def render_plotly(df: pd.DataFrame,
                key: str,
                title: str,
                # tool_name: str,
                save_chart: bool = False
                ):
    """
    Function to visualize the dataframe using Plotly.

    Args:
        df: pd.DataFrame: The input dataframe
    """
    # toggle_state = st.session_state[f'toggle_plotly_{tool_name}_{key.split("_")[-1]}']\
    toggle_state = st.session_state[f'toggle_plotly_{key.split("plotly_")[1]}']
    if toggle_state:
        df_simulation_results = df.melt(
                                    id_vars='Time',
                                    var_name='Species',
                                    value_name='Concentration')
        fig = px.line(df_simulation_results,
                        x='Time',
                        y='Concentration',
                        color='Species',
                        title=title,
                        height=500,
                        width=600
                )
        # Display the plotly chart
        st.plotly_chart(fig,
                        use_container_width=True,
                        key=key)
    if save_chart:
        # Add data to the chat history
        st.session_state.messages.append({
                "type": "plotly",
                "content": df,
                "key": key,
                "title": title,
                # "tool_name": tool_name
            })

def render_table(df: pd.DataFrame,
                #  tool_name: str,
                 key: str,
                 save_table: bool = False
                ):
    """
    Function to render the table in the chat.
    """
    # print (st.session_state['toggle_simulate_model_'+key.split("_")[-1]])
    # toggle_state = st.session_state[f'toggle_table_{tool_name}_{key.split("_")[-1]}']
    toggle_state = st.session_state[f'toggle_table_{key.split("dataframe_")[1]}']
    if toggle_state:
        st.dataframe(df,
                    use_container_width=True,
                    key=key)
    if save_table:
        # Add data to the chat history
        st.session_state.messages.append({
                "type": "dataframe",
                "content": df,
                "key": key,
                # "tool_name": tool_name
            })

def render_graph(graph_dict: dict,
                 key: str,
                 save_graph: bool = False):
    """
    Function to render the graph in the chat.

    Args:
        graph_dict: The graph dictionary
        key: The key for the graph
        save_graph: Whether to save the graph in the chat history
    """
    # Create a directed graph
    graph = nx.DiGraph()

    # Add nodes with attributes
    for node, attrs in graph_dict["nodes"]:
        graph.add_node(node, **attrs)

    # Add edges with attributes
    for source, target, attrs in graph_dict["edges"]:
        graph.add_edge(source, target, **attrs)

    # Render the graph
    fig = gravis.d3(
            graph,
            node_size_factor=3.0,
            show_edge_label=True,
            edge_label_data_source="label",
            edge_curvature=0.25,
            zoom_factor=1.0,
            many_body_force_strength=-500,
            many_body_force_theta=0.3,
            node_hover_neighborhood=True,
            # layout_algorithm_active=True,
        )
    components.html(fig.to_html(), height=475)

    if save_graph:
        # Add data to the chat history
        st.session_state.messages.append({
                "type": "graph",
                "content": graph_dict,
                "key": key,
            })

@st.dialog("Warning ⚠️")
def update_llm_model():
    """
    Function to update the LLM model.
    """
    llm_model = st.session_state.llm_model
    st.warning(f"Clicking 'Continue' will reset all agents, \
            set the selected LLM to {llm_model}. \
            This action will reset the entire app, \
            and agents will lose access to the \
            conversation history. Are you sure \
            you want to proceed?")
    if st.button("Continue"):
        # st.session_state.vote = {"item": item, "reason": reason}
        # st.rerun()
        # Delete all the items in Session state
        for key in st.session_state.keys():
            if key in ["messages", "app"]:
                del st.session_state[key]
        st.rerun()

@st.dialog("Get started with Talk2Biomodels 🚀")
def help_button():
    """
    Function to display the help dialog.
    """
    st.markdown('''I am an AI agent designed to assist you with biological
modeling and simulations. I can assist with tasks such as:
1. Search specific models in the BioModels database.

`Search models on Crohns disease`

2. Extract information about models, including species, parameters, units,
name and descriptions.

`Show me the name of the model 537 and parameters related to drug dosage`

3. Simulate models:
    - Run simulations of models to see how they behave over time.
    - Set the duration and the interval.
    - Specify which species/parameters you want to include and their starting concentrations.
    - Include recurring events.

`Simulate the model for 2016 hours and intervals 2016 with an initial concentration
of `DoseQ2W` set to 300 and `Dose` set to 0.`

4. Answer questions about simulation results.

`What is the concentration of species IL6 in serum at time 1000?`

5. Create custom plots to visualize the simulation results.

`Plot the concentration of all the interleukins over time`

6. Provide feedback to the developers by clicking on the feedback button.
''')
