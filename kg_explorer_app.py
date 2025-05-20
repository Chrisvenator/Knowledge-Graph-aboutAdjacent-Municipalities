import streamlit as st
import torch
import numpy as np
import os
import json
from pykeen.triples import TriplesFactory
from pykeen.models import TransE
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Municipality Knowledge Graph Explorer",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("Municipality Knowledge Graph Explorer")
st.write("""
This app lets you explore municipalities and their relationships in a knowledge graph. 
You can make link predictions and find similar entities based on trained TransE embeddings.
""")

# Check if model directory exists
model_dir = st.sidebar.text_input("Model Directory", value="model")


@st.cache_resource
def load_model(model_dir):
    """Load a trained model and mappings from disk."""
    try:
        if not os.path.exists(model_dir):
            st.error(f"Model directory {model_dir} does not exist")
            return None, None

        # Load model configuration
        with open(os.path.join(model_dir, "model_config.json"), "r") as f:
            model_config = json.load(f)

        # Load mappings
        with open(os.path.join(model_dir, "mappings.json"), "r") as f:
            mappings_data = json.load(f)

        # Create mappings
        entity_to_id = {k: int(v) for k, v in mappings_data["entity_to_id"].items()}
        relation_to_id = {k: int(v) for k, v in mappings_data["relation_to_id"].items()}

        # Create dummy TriplesFactory with the mappings
        dummy_tf = TriplesFactory(
            mapped_triples=torch.zeros((1, 3), dtype=torch.long),  # Dummy tensor
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        )

        # Import the model class dynamically
        if model_config["model_class"] == "TransE":
            model_class = TransE
        else:
            st.error(f"Loading {model_config['model_class']} model is not implemented")
            return None, None

        # Create model instance
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model_class(
            triples_factory=dummy_tf,
            embedding_dim=model_config["embedding_dim"],
            scoring_fct_norm=model_config["scoring_fct_norm"],
            entity_initializer=None,  # Will be overwritten
            relation_initializer=None,  # Will be overwritten
        )

        # Load state dict
        model.load_state_dict(torch.load(
            os.path.join(model_dir, "model_state.pt"),
            map_location=device
        ))
        model.to(device)

        return model, dummy_tf
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


@st.cache_data
def extract_embeddings(_model):
    """Extract entity and relation embeddings from the model."""
    try:
        # Try to get entity embeddings
        entity_representation = model.entity_representations[0]
        if hasattr(entity_representation, 'weight'):
            entity_embeddings = entity_representation.weight.detach().cpu().numpy()
        elif hasattr(entity_representation, '_embeddings'):
            entity_embeddings = entity_representation._embeddings.weight.detach().cpu().numpy()
        else:
            st.error("Could not find entity embeddings")
            return None, None

        # Try to get relation embeddings
        relation_representation = model.relation_representations[0]
        if hasattr(relation_representation, 'weight'):
            relation_embeddings = relation_representation.weight.detach().cpu().numpy()
        elif hasattr(relation_representation, '_embeddings'):
            relation_embeddings = relation_representation._embeddings.weight.detach().cpu().numpy()
        else:
            st.error("Could not find relation embeddings")
            return None, None

        return entity_embeddings, relation_embeddings

    except Exception as e:
        st.error(f"Error extracting embeddings: {str(e)}")
        return None, None


def predict_links(head, relation, entity_to_id, relation_to_id, model, entity_embeddings, relation_embeddings, top_n=5):
    """Use the model for link prediction."""
    if head not in entity_to_id or relation not in relation_to_id:
        st.error(f"Entity {head} or relation {relation} not found in the graph")
        return None, None

    head_id = entity_to_id[head]
    relation_id = relation_to_id[relation]

    # Use the model's scoring function directly
    try:
        # Create tensors for all possible combinations
        device = next(model.parameters()).device
        head_tensor = torch.tensor([head_id], device=device)
        relation_tensor = torch.tensor([relation_id], device=device)
        tail_tensor = torch.tensor(list(range(len(entity_to_id))), device=device)

        # Expand head and relation tensors
        head_expanded = head_tensor.repeat(len(entity_to_id))
        relation_expanded = relation_tensor.repeat(len(entity_to_id))

        # Create triple tensor
        triples = torch.stack([head_expanded, relation_expanded, tail_tensor], dim=1)

        # Get scores
        with torch.no_grad():
            scores = model.score_hrt(triples).detach().cpu().numpy()

        # For TransE with L1 norm, lower scores are better
        # So we negate them for ranking
        scores = -scores
    except Exception as e:
        st.warning(f"Error using model's scoring function: {str(e)}")
        st.write("Falling back to manual scoring...")

        # Manual scoring using embeddings
        head_emb = entity_embeddings[head_id]
        rel_emb = relation_embeddings[relation_id]
        scores = []

        for tail_id in range(len(entity_to_id)):
            tail_emb = entity_embeddings[tail_id]
            # TransE scoring: ||h + r - t||_1
            # Lower is better for TransE with L1 norm
            score = -np.linalg.norm(head_emb + rel_emb - tail_emb, ord=1)
            scores.append(score)

        scores = np.array(scores)

    # Sort and display results
    indices = np.argsort(scores)[::-1]  # Sort in descending order

    # Get results
    results = []
    for i in range(min(top_n, len(indices))):
        idx = indices[i]
        tail = [k for k, v in entity_to_id.items() if v == idx][0]
        tail_simple = tail.split('/')[-1]
        # Convert score to float to avoid numpy formatting issues
        score_value = float(scores[idx])
        results.append({
            "tail_id": idx,
            "tail_uri": tail,
            "tail_name": tail_simple,
            "score": score_value
        })

    return results


def find_similar_entities(entity_label, entity_to_id, entity_embeddings, top_n=5):
    """Find similar entities based on embedding distance."""
    if entity_label not in entity_to_id:
        st.error(f"Entity {entity_label} not found in the graph")
        return None

    entity_id = entity_to_id[entity_label]
    entity_emb = entity_embeddings[entity_id]

    # Calculate distances
    distances = np.linalg.norm(entity_embeddings - entity_emb, axis=1)

    # Sort by distance (excluding itself)
    indices = np.argsort(distances)

    # Get results
    results = []
    for i in range(1, min(top_n + 1, len(indices))):
        idx = indices[i]
        entity = [k for k, v in entity_to_id.items() if v == idx][0]
        simple_entity = entity.split('/')[-1]  # Extract name from URI
        # Convert distance to float to avoid numpy formatting issues
        distance_value = float(distances[idx])
        results.append({
            "entity_id": idx,
            "entity_uri": entity,
            "entity_name": simple_entity,
            "distance": distance_value
        })

    return results


def visualize_graph(center_entity, related_entities, relation_label=None):
    """Create a graph visualization for an entity and related entities."""
    # Create a directed graph
    G = nx.DiGraph()

    # Add center node
    center_name = center_entity.split('/')[-1]
    G.add_node(center_name, type="center")

    # Add related nodes and edges
    for entity in related_entities:
        entity_name = entity["entity_name"] if "entity_name" in entity else entity["tail_name"]
        G.add_node(entity_name, type="related", score=entity.get("score", entity.get("distance", 0)))
        if relation_label:
            G.add_edge(center_name, entity_name, label=relation_label.split('/')[-1], weight=entity.get("score", 1.0))
        else:
            G.add_edge(center_name, entity_name, label="similar_to", weight=1.0 / entity.get("distance", 1.0))

    # Create a colormap
    colors = []
    for node in G.nodes():
        if G.nodes[node]["type"] == "center":
            colors.append("lightblue")
        else:
            colors.append("lightgreen")

    # Set up the figure
    plt.figure(figsize=(10, 8))

    # Create a spring layout
    pos = nx.spring_layout(G, seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=700, alpha=0.8)

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.5, edge_color="gray",
                           connectionstyle="arc3,rad=0.1", arrowsize=15)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")

    # Draw edge labels
    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(f"Graph visualization for {center_name}")
    plt.axis("off")

    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return Image.open(buf)


# Load model
with st.sidebar:
    st.header("Model Information")
    model_loading_spinner = st.spinner("Loading model...")
    with model_loading_spinner:
        model, tf = load_model(model_dir)

    if model is not None and tf is not None:
        st.success("Model loaded successfully!")
        entity_embeddings, relation_embeddings = extract_embeddings(model)
        st.write(f"Entities: {len(tf.entity_to_id)}")
        st.write(f"Relations: {len(tf.relation_to_id)}")
        if entity_embeddings is not None:
            st.write(f"Embedding dimension: {entity_embeddings.shape[1]}")
    else:
        st.error("Failed to load model. Please check the model directory.")
        st.stop()

# Main content
st.header("Knowledge Graph Explorer")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Link Prediction", "Similar Entities", "Entity Browser"])

# Extract all entities and relations
entities = list(tf.entity_to_id.keys())
entity_names = [e.split('/')[-1] for e in entities]
entities_with_names = [(e, e.split('/')[-1]) for e in entities]
entities_with_names.sort(key=lambda x: x[1].lower())  # Sort by entity name

relations = list(tf.relation_to_id.keys())
relation_names = [r.split('/')[-1] for r in relations]
relations_with_names = [(r, r.split('/')[-1]) for r in relations]
relations_with_names.sort(key=lambda x: x[1].lower())  # Sort by relation name

# Tab 1: Link Prediction
with tab1:
    st.subheader("Predict Links")

    col1, col2 = st.columns(2)

    with col1:
        # Dropdown for entities (municipalities)
        st.subheader("Select Municipality")
        municipality_list = [e[0] for e in entities_with_names]
        municipality_display = [e[1] for e in entities_with_names]
        head_entity_index = st.selectbox(
            "Select municipality",
            range(len(municipality_list)),
            format_func=lambda i: municipality_display[i]
        )
        head_entity = municipality_list[head_entity_index]

    with col2:
        # Dropdown for relations
        st.subheader("Select Relation")
        relation_list = [r[0] for r in relations_with_names]
        relation_display = [r[1] for r in relations_with_names]
        relation_index = st.selectbox(
            "Select relation",
            range(len(relation_list)),
            format_func=lambda i: relation_display[i]
        )
        relation = relation_list[relation_index]

    # Number of predictions to show
    top_n = st.slider("Number of predictions to show", min_value=1, max_value=20, value=5)

    if st.button("Predict Links", key="predict_links_button"):
        with st.spinner("Predicting links..."):
            results = predict_links(
                head_entity, relation,
                tf.entity_to_id, tf.relation_to_id,
                model, entity_embeddings, relation_embeddings,
                top_n=top_n
            )

        if results:
            st.success("Prediction complete!")

            # Create a DataFrame for display
            df = pd.DataFrame(results)
            df = df[["tail_name", "score"]]
            df.columns = ["Predicted Entity", "Score"]

            # Display results in a table
            st.dataframe(df)

            # Visualize the results
            st.subheader("Graph Visualization")
            img = visualize_graph(head_entity, results, relation)
            st.image(img)

            # Display additional entity information
            selected_entity = st.selectbox(
                "Select entity for more details",
                [result["tail_uri"] for result in results],
                format_func=lambda x: x.split('/')[-1]
            )

            if selected_entity:
                st.write(f"**URI:** {selected_entity}")
                st.write(f"**ID in graph:** {tf.entity_to_id[selected_entity]}")

# Tab 2: Similar Entities
with tab2:
    st.subheader("Find Similar Entities")

    # Dropdown for selecting entity
    st.subheader("Select Municipality")
    similar_entity_index = st.selectbox(
        "Select entity to find similar municipalities",
        range(len(municipality_list)),
        format_func=lambda i: municipality_display[i],
        key="similar_entity_dropdown"
    )
    selected_entity = municipality_list[similar_entity_index]

    # Number of similar entities to show
    top_n = st.slider("Number of similar entities to show", min_value=1, max_value=20, value=5, key="similar_slider")

    if st.button("Find Similar Entities", key="find_similar_button"):
        with st.spinner("Finding similar entities..."):
            results = find_similar_entities(
                selected_entity, tf.entity_to_id, entity_embeddings, top_n=top_n
            )

        if results:
            st.success("Search complete!")

            # Create a DataFrame for display
            df = pd.DataFrame(results)
            df = df[["entity_name", "distance"]]
            df.columns = ["Similar Entity", "Distance"]

            # Display results in a table
            st.dataframe(df)

            # Visualize the results
            st.subheader("Graph Visualization")
            img = visualize_graph(selected_entity, results)
            st.image(img)

            # Display additional entity information
            selected_similar = st.selectbox(
                "Select entity for more details",
                [result["entity_uri"] for result in results],
                format_func=lambda x: x.split('/')[-1]
            )

            if selected_similar:
                st.write(f"**URI:** {selected_similar}")
                st.write(f"**ID in graph:** {tf.entity_to_id[selected_similar]}")

# Tab 3: Entity Browser
with tab3:
    st.subheader("Browse All Entities")

    # Filters
    st.write("### Filter Options")
    col1, col2 = st.columns(2)

    with col1:
        # Dropdown filter for entities
        entity_filter = st.selectbox(
            "Filter by municipality type",
            ["All"] + sorted(set(municipality_display)),
            key="entity_filter_dropdown"
        )

    with col2:
        # Text search
        search_term = st.text_input("Search entities by name", key="search_browser")

    # Apply filters
    if entity_filter == "All" and not search_term:
        filtered_entities = entities
    elif entity_filter == "All":
        filtered_entities = [e for e in entities if search_term.lower() in e.lower()]
    elif not search_term:
        filtered_entities = [e for e in entities if entity_filter.lower() in e.split('/')[-1].lower()]
    else:
        filtered_entities = [e for e in entities if entity_filter.lower() in e.split('/')[-1].lower() and search_term.lower() in e.lower()]

    # Create a DataFrame of filtered entities
    entity_df = pd.DataFrame({
        "Entity Name": [e.split('/')[-1] for e in filtered_entities],
        "URI": filtered_entities,
        "ID": [tf.entity_to_id[e] for e in filtered_entities]
    })

    # Display in a table with pagination
    if not entity_df.empty:
        st.dataframe(entity_df)
        st.write(f"Showing {len(entity_df)} entities")
    else:
        st.warning("No entities match your filters")

# Add a footer
st.markdown("---")
st.markdown("Municipality Knowledge Graph Explorer | Built with Streamlit and PyKEEN")