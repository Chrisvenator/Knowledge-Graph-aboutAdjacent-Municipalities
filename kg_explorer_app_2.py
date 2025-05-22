import streamlit as st
import pandas as pd
import json
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from rdflib import Graph
import numpy as np
from collections import defaultdict, Counter
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import re

# Set page config
st.set_page_config(
    page_title="Austrian Municipalities Knowledge Graph Explorer",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏘️ Austrian Municipalities Knowledge Graph Explorer")


@st.cache_data
def load_rdf_data(file_path):
    """Load RDF data from TTL file."""
    if not os.path.exists(file_path):
        return None, None, None

    g = Graph()
    g.parse(file_path, format="turtle")

    # Extract triples
    triples = []
    entities = set()
    relations = set()

    for s, p, o in g:
        # Only include entity-entity relations (skip literals)
        if not str(o).startswith('"') and 'http' in str(o):
            subject = str(s)
            predicate = str(p)
            object_uri = str(o)

            # Extract simple names
            subject_name = subject.split('/')[-1].replace('_', ' ')
            predicate_name = predicate.split('/')[-1]
            object_name = object_uri.split('/')[-1].replace('_', ' ')

            triples.append({
                'subject': subject,
                'predicate': predicate,
                'object': object_uri,
                'subject_name': subject_name,
                'predicate_name': predicate_name,
                'object_name': object_name
            })

            entities.add((subject, subject_name))
            entities.add((object_uri, object_name))
            relations.add((predicate, predicate_name))

    return triples, list(entities), list(relations)


@st.cache_data
def load_attributes_data(file_path):
    """Load entity attributes from JSON file."""
    if not os.path.exists(file_path):
        return {}

    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_network_graph(triples, max_nodes=100):
    """Create a NetworkX graph from triples."""
    G = nx.Graph()

    # Add edges (this automatically adds nodes)
    edge_counter = Counter()
    for triple in triples[:max_nodes * 2]:  # Limit to avoid performance issues
        subject = triple['subject_name']
        object_name = triple['object_name']
        predicate = triple['predicate_name']

        # Skip self-loops and very long names
        if subject != object_name and len(subject) < 50 and len(object_name) < 50:
            edge_key = (subject, object_name)
            edge_counter[edge_key] += 1

            if not G.has_edge(subject, object_name):
                G.add_edge(subject, object_name, relation=predicate, weight=1)
            else:
                G[subject][object_name]['weight'] += 1

    return G


def visualize_network_plotly(G, title="Knowledge Graph Network"):
    """Create an interactive network visualization using Plotly."""
    if len(G.nodes()) == 0:
        st.warning("No network data to display")
        return None

    # Get node positions using spring layout
    pos = nx.spring_layout(G, k=1, iterations=50)

    # Extract edges
    edge_x = []
    edge_y = []
    edge_info = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        # Get edge info
        edge_data = G.get_edge_data(edge[0], edge[1])
        relation = edge_data.get('relation', 'connected')
        weight = edge_data.get('weight', 1)
        edge_info.append(f"{edge[0]} --{relation}--> {edge[1]} (weight: {weight})")

    # Extract nodes
    node_x = []
    node_y = []
    node_text = []
    node_info = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

        # Get node degree for sizing
        degree = G.degree(node)
        node_info.append(f"{node}<br>Connections: {degree}")

    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        hovertext=node_info,
        marker=dict(
            size=[G.degree(node) * 2 + 10 for node in G.nodes()],
            color=[G.degree(node) for node in G.nodes()],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Node Connections"),
            line=dict(width=2, color='white')
        )
    )

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        title=title,
        title_font_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[
            dict(
                text="Click and drag to pan, scroll to zoom",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.005,
                y=-0.002,
                xanchor='left',
                yanchor='bottom',
                font=dict(color="#888", size=12)
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )

    return fig


def analyze_attributes(attributes_data):
    """Analyze and summarize the attributes data."""
    if not attributes_data:
        return None, None

    # Count entities and attributes
    entity_count = len(attributes_data)
    all_attributes = []
    attribute_types = set()

    for entity_uri, attrs in attributes_data.items():
        for attr_name, attr_info in attrs.items():
            attr_simple_name = attr_name.split('/')[-1]
            attribute_types.add(attr_simple_name)
            all_attributes.append({
                'entity': entity_uri.split('/')[-1].replace('_', ' '),
                'attribute': attr_simple_name,
                'value': attr_info['value'],
                'datatype': attr_info.get('datatype', 'unknown')
            })

    df = pd.DataFrame(all_attributes)

    summary = {
        'total_entities': entity_count,
        'total_attributes': len(all_attributes),
        'unique_attribute_types': len(attribute_types),
        'attribute_types': sorted(list(attribute_types))
    }

    return df, summary


def create_attribute_visualization(df, attribute_name):
    """Create visualizations for a specific attribute."""
    if df is None or attribute_name not in df['attribute'].values:
        return None

    # Filter data for the specific attribute
    attr_data = df[df['attribute'] == attribute_name].copy()

    # Try to convert values to numeric for better visualization
    numeric_values = []
    for value in attr_data['value']:
        try:
            # Handle different number formats
            if isinstance(value, str):
                # Remove thousand separators and convert comma to dot
                clean_value = value.replace('.', '').replace(',', '.')
                numeric_values.append(float(clean_value))
            else:
                numeric_values.append(float(value))
        except (ValueError, TypeError):
            numeric_values.append(None)

    attr_data['numeric_value'] = numeric_values
    valid_data = attr_data.dropna(subset=['numeric_value'])

    if len(valid_data) == 0:
        st.warning(f"No numeric data available for {attribute_name}")
        return None

    # Create histogram
    fig = px.histogram(
        valid_data,
        x='numeric_value',
        title=f'Distribution of {attribute_name}',
        labels={'numeric_value': attribute_name, 'count': 'Number of Municipalities'},
        nbins=30
    )
    fig.update_layout(height=400)

    return fig


def main():
    # Sidebar for file selection
    st.sidebar.header("📁 Data Files")

    # File inputs
    ttl_file = st.sidebar.text_input(
        "RDF/TTL File Path",
        value="dataset_entities_only.ttl",
        help="Path to the Turtle (.ttl) file containing entity relationships"
    )

    attributes_file = st.sidebar.text_input(
        "Attributes JSON File Path",
        value="dataset_attributes.json",
        help="Path to the JSON file containing entity attributes"
    )

    # Load data
    if st.sidebar.button("🔄 Load Data"):
        st.cache_data.clear()

    # Main content
    with st.spinner("Loading data..."):
        triples, entities, relations = load_rdf_data(ttl_file)
        attributes_data = load_attributes_data(attributes_file)

    if triples is None:
        st.error(f"Could not load RDF data from {ttl_file}")
        st.stop()

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🕸️ Network Graph", "📈 Attributes Analysis", "🔍 Entity Explorer"])

    with tab1:
        st.header("Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Entities", len(entities))

        with col2:
            st.metric("Total Relations", len(relations))

        with col3:
            st.metric("Total Triples", len(triples))

        with col4:
            entities_with_attrs = len(attributes_data) if attributes_data else 0
            st.metric("Entities with Attributes", entities_with_attrs)

        # Show sample data
        st.subheader("Sample Relationships")
        if len(triples) > 0:
            sample_df = pd.DataFrame(triples[:20])[['subject_name', 'predicate_name', 'object_name']]
            st.dataframe(sample_df, use_container_width=True)

        # Show relation types
        st.subheader("Relation Types")
        if relations:
            relation_names = [r[1] for r in relations]
            relation_counts = Counter([t['predicate_name'] for t in triples])
            rel_df = pd.DataFrame([
                {'Relation': rel, 'Count': relation_counts.get(rel, 0)}
                for rel in relation_names
            ]).sort_values('Count', ascending=False)
            st.dataframe(rel_df, use_container_width=True)

    with tab2:
        st.header("Network Graph Visualization")

        # Controls
        col1, col2 = st.columns([1, 3])

        with col1:
            max_nodes = st.slider("Max Nodes to Display", 10, 200, 50)
            filter_relation = st.selectbox(
                "Filter by Relation",
                ["All"] + [r[1] for r in relations],
                index=0
            )

        # Filter triples if needed
        filtered_triples = triples
        if filter_relation != "All":
            filtered_triples = [t for t in triples if t['predicate_name'] == filter_relation]

        if len(filtered_triples) == 0:
            st.warning("No triples match the selected filter.")
        else:
            # Create and display network
            with st.spinner("Creating network visualization..."):
                G = create_network_graph(filtered_triples, max_nodes)

                if len(G.nodes()) > 0:
                    fig = visualize_network_plotly(G, f"Network Graph ({len(G.nodes())} nodes, {len(G.edges())} edges)")
                    st.plotly_chart(fig, use_container_width=True)

                    # Network statistics
                    st.subheader("Network Statistics")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Nodes", len(G.nodes()))
                    with col2:
                        st.metric("Edges", len(G.edges()))
                    with col3:
                        if len(G.nodes()) > 0:
                            avg_degree = sum(dict(G.degree()).values()) / len(G.nodes())
                            st.metric("Avg Degree", f"{avg_degree:.2f}")
                    with col4:
                        if len(G.nodes()) > 0:
                            try:
                                density = nx.density(G)
                                st.metric("Density", f"{density:.4f}")
                            except:
                                st.metric("Density", "N/A")
                else:
                    st.warning("No network data to display with current filters.")

    with tab3:
        st.header("Attributes Analysis")

        if attributes_data:
            df, summary = analyze_attributes(attributes_data)

            if summary:
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Entities with Attributes", summary['total_entities'])
                with col2:
                    st.metric("Total Attribute Values", summary['total_attributes'])
                with col3:
                    st.metric("Unique Attribute Types", summary['unique_attribute_types'])

                # Attribute type selection
                st.subheader("Attribute Distribution")
                selected_attribute = st.selectbox(
                    "Choose an attribute to analyze:",
                    summary['attribute_types']
                )

                if selected_attribute:
                    fig = create_attribute_visualization(df, selected_attribute)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                    # Show sample values
                    sample_data = df[df['attribute'] == selected_attribute].head(20)
                    st.subheader(f"Sample {selected_attribute} Values")
                    st.dataframe(sample_data[['entity', 'value', 'datatype']], use_container_width=True)

                # Full attributes table
                with st.expander("View All Attributes Data"):
                    st.dataframe(df, use_container_width=True)
        else:
            st.warning("No attributes data available. Please check the attributes file path.")

    with tab4:
        st.header("Entity Explorer")

        if entities:
            # Entity selection
            entity_names = [e[1] for e in entities]
            selected_entity = st.selectbox(
                "Select an entity to explore:",
                entity_names
            )

            if selected_entity:
                # Find the full URI
                selected_uri = None
                for uri, name in entities:
                    if name == selected_entity:
                        selected_uri = uri
                        break

                if selected_uri:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Relationships")

                        # Find all relationships for this entity
                        entity_relations = []
                        for triple in triples:
                            if triple['subject'] == selected_uri:
                                entity_relations.append({
                                    'Direction': 'Outgoing',
                                    'Relation': triple['predicate_name'],
                                    'Connected Entity': triple['object_name']
                                })
                            elif triple['object'] == selected_uri:
                                entity_relations.append({
                                    'Direction': 'Incoming',
                                    'Relation': triple['predicate_name'],
                                    'Connected Entity': triple['subject_name']
                                })

                        if entity_relations:
                            relations_df = pd.DataFrame(entity_relations)
                            st.dataframe(relations_df, use_container_width=True)
                        else:
                            st.info("No relationships found for this entity.")

                    with col2:
                        st.subheader("Attributes")

                        if attributes_data and selected_uri in attributes_data:
                            attrs = attributes_data[selected_uri]
                            attr_list = []
                            for attr_name, attr_info in attrs.items():
                                attr_list.append({
                                    'Attribute': attr_name.split('/')[-1],
                                    'Value': attr_info['value'],
                                    'Type': attr_info.get('datatype', 'unknown').split('/')[-1]
                                })

                            attrs_df = pd.DataFrame(attr_list)
                            st.dataframe(attrs_df, use_container_width=True)
                        else:
                            st.info("No attributes found for this entity.")


if __name__ == "__main__":
    main()