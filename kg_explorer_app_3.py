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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import requests
import time
import folium
from streamlit_folium import st_folium
import math
import geopandas as gpd
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Austrian Municipalities Geographic Explorer",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🗺️ Austrian Municipalities Geographic Knowledge Graph Explorer")


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


@st.cache_data
def get_austrian_municipality_coordinates():
    """
    Load Austrian municipality coordinates from shapefile.
    """
    try:
        shp_path = Path("data/Gliederung_OEsterreichs_in_Gemeinden.zip")
        gdf = gpd.read_file(shp_path)
        print(f"Shapefile columns: {gdf.columns.tolist()}")
        NAME_FIELD = "g_name"  # <— HIER ggf. anpassen!

        gdf_proj = gdf.to_crs(31287)  # MGI / Austria Lambert
        centroids = gdf_proj.geometry.centroid.to_crs(4326)

        municipality_coords = {name: {'lat': float(pt.y), 'lon': float(pt.x)}
                               for name, pt in zip(gdf[NAME_FIELD], centroids)}

        st.success(f"✅ Loaded {len(municipality_coords)} municipalities from shapefile")
        return municipality_coords

    except Exception as e:
        st.error(f"Error loading shapefile: {e}")
        st.info("Falling back to sample coordinates...")
        # Fallback to sample data if shapefile is not available
        municipality_coords = {
            'Vienna': {'lat': 48.2082, 'lon': 16.3738},
            'Graz': {'lat': 47.0707, 'lon': 15.4395},
            'Linz': {'lat': 48.3069, 'lon': 14.2858},
            'Salzburg': {'lat': 47.8095, 'lon': 13.0550},
            'Innsbruck': {'lat': 47.2692, 'lon': 11.4041},
            'Klagenfurt': {'lat': 46.6247, 'lon': 14.3054},
            'Villach': {'lat': 46.6111, 'lon': 13.8558},
            'Wels': {'lat': 48.1597, 'lon': 14.0294},
            'Sankt Pölten': {'lat': 48.2062, 'lon': 15.6256},
            'Dornbirn': {'lat': 47.4125, 'lon': 9.7417},
        }
        return municipality_coords


def create_feature_vectors(attributes_data, entity_to_uri_map):
    """Create numerical feature vectors from entity attributes for similarity calculation."""
    if not attributes_data:
        return {}, []

    # Collect all possible attribute types
    all_attribute_types = set()
    for entity_attrs in attributes_data.values():
        all_attribute_types.update(entity_attrs.keys())

    attribute_names = sorted(list(all_attribute_types))
    feature_vectors = {}

    # Create feature vectors for each entity
    for entity_uri, entity_attrs in attributes_data.items():
        features = []

        for attr_name in attribute_names:
            if attr_name in entity_attrs:
                attr_info = entity_attrs[attr_name]
                value = attr_info['value']
                # Fix: Handle None datatype properly
                datatype = attr_info.get('datatype') or ''  # Use 'or' to handle None values

                # Convert to numerical value based on datatype
                if any(t in datatype for t in ['integer', 'positiveInteger', 'nonNegativeInteger', 'decimal', 'double', 'euro']):
                    try:
                        # Handle different number formats
                        if isinstance(value, str):
                            clean_value = value.replace('.', '').replace(',', '.')
                            features.append(float(clean_value))
                        else:
                            features.append(float(value))
                    except ValueError:
                        features.append(0.0)
                elif 'gYear' in datatype:
                    try:
                        features.append(float(value))
                    except ValueError:
                        features.append(2019.0)  # Default year
                elif 'string' in datatype:
                    # For strings, use hash as feature (simple approach)
                    features.append(float(hash(value) % 10000))
                else:
                    features.append(0.0)
            else:
                features.append(0.0)  # Missing attribute

        # Get entity name for mapping
        entity_name = entity_uri.split('/')[-1].replace('_', ' ')
        feature_vectors[entity_name] = np.array(features)

    return feature_vectors, attribute_names


def calculate_similarity_matrix(feature_vectors):
    """Calculate cosine similarity matrix between entities."""
    if not feature_vectors:
        return {}

    entity_names = list(feature_vectors.keys())
    feature_matrix = np.array([feature_vectors[name] for name in entity_names])

    # Standardize features
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(feature_matrix)

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(standardized_features)

    # Create similarity dictionary
    similarity_dict = {}
    for i, entity1 in enumerate(entity_names):
        similarity_dict[entity1] = {}
        for j, entity2 in enumerate(entity_names):
            if i != j:  # Don't include self-similarity
                similarity_dict[entity1][entity2] = similarity_matrix[i][j]

    return similarity_dict


def get_connected_municipalities(triples):
    """Get set of municipalities that have connections in the graph."""
    connected_municipalities = set()

    # Filter for adjacency relationships
    adjacency_triples = [t for t in triples if 'adjacent' in t['predicate_name'].lower()]

    for triple in adjacency_triples:
        connected_municipalities.add(triple['subject_name'])
        connected_municipalities.add(triple['object_name'])

    return connected_municipalities


def create_basic_geographic_map(triples, municipality_coords, similarity_dict, max_edges=50, hide_unconnected=False, use_confidence_mode=True):
    """Create a basic geographic map with simple points and clear similarity lines."""

    # Filter for adjacency relationships
    adjacency_triples = [t for t in triples if 'adjacent' in t['predicate_name'].lower()]

    if not adjacency_triples:
        st.warning("No adjacency relationships found in the data.")
        return None

    # Get connected municipalities if we need to hide unconnected ones
    connected_municipalities = get_connected_municipalities(triples) if hide_unconnected else set()

    # Create base map centered on Austria with simpler settings
    austria_center = [47.5, 14.5]
    m = folium.Map(
        location=austria_center,
        zoom_start=7,
        tiles='OpenStreetMap',
        prefer_canvas=True  # Better performance
    )

    # Track municipalities that we can plot
    plotted_municipalities = set()
    plotted_edges = []

    # First pass: Add simple circle markers for municipalities
    municipalities_to_plot = []

    # Get all municipalities from adjacency relationships
    for entity_uri, entity_name in [(t['subject'], t['subject_name']) for t in adjacency_triples] + \
                                   [(t['object'], t['object_name']) for t in adjacency_triples]:
        if entity_name not in plotted_municipalities:
            municipalities_to_plot.append(entity_name)
            plotted_municipalities.add(entity_name)

    # Add all municipalities if not hiding unconnected
    if not hide_unconnected:
        for municipality_name in municipality_coords.keys():
            if municipality_name not in plotted_municipalities:
                municipalities_to_plot.append(municipality_name)
                plotted_municipalities.add(municipality_name)

    # Plot municipalities
    for entity_name in municipalities_to_plot:
        if entity_name in municipality_coords:
            # Skip if hiding unconnected and this municipality is not connected
            if hide_unconnected and entity_name not in connected_municipalities:
                continue

            coords = municipality_coords[entity_name]

            # Different colors for connected vs unconnected
            if entity_name in connected_municipalities:
                color = 'blue'
                fillColor = 'blue'
                radius = 5
            else:
                color = 'gray'
                fillColor = 'lightgray'
                radius = 3

            # Use simple CircleMarker instead of complex Marker
            folium.CircleMarker(
                location=[coords['lat'], coords['lon']],
                radius=radius,
                popup=entity_name,
                tooltip=entity_name,
                color=color,
                fill=True,
                fillColor=fillColor,
                fillOpacity=0.7,
                weight=2
            ).add_to(m)

    # Second pass: Add similarity-colored edges (limit to improve performance)
    edge_count = 0
    valid_edges = []

    # Pre-filter edges to only those with both municipalities having coordinates
    for triple in adjacency_triples:
        source_name = triple['subject_name']
        target_name = triple['object_name']

        if (source_name in municipality_coords and
                target_name in municipality_coords and
                edge_count < max_edges):

            # Calculate similarity if available
            similarity = 0.0
            if (similarity_dict and
                    source_name in similarity_dict and
                    target_name in similarity_dict[source_name]):
                similarity = similarity_dict[source_name][target_name]

            valid_edges.append({
                'source': source_name,
                'target': target_name,
                'similarity': similarity
            })
            edge_count += 1

    # Sort edges by similarity (for confidence mode) or by absolute similarity (for similarity mode)
    if use_confidence_mode:
        # Sort by similarity for confidence ranking (highest first)
        valid_edges.sort(key=lambda x: x['similarity'], reverse=True)
    else:
        # Sort by absolute similarity to show most interesting ones first
        valid_edges.sort(key=lambda x: abs(x['similarity']), reverse=True)

    # Calculate color scheme based on mode
    num_edges = len(valid_edges)

    # Add the edges to the map
    for i, edge in enumerate(valid_edges):
        source_coords = municipality_coords[edge['source']]
        target_coords = municipality_coords[edge['target']]
        similarity = edge['similarity']

        if use_confidence_mode:
            # CONFIDENCE MODE: Color based on ranking position
            if num_edges > 1:
                confidence_rank = i / (num_edges - 1)  # Normalized rank from 0 to 1
            else:
                confidence_rank = 0

            # Create smooth gradient: Green (confident) -> Yellow -> Red (uncertain)
            if confidence_rank <= 0.5:
                # Green to Yellow transition
                red = int(confidence_rank * 2 * 255)
                green = 255
                blue = 0
            else:
                # Yellow to Red transition
                red = 255
                green = int(255 * (2 - confidence_rank * 2))
                blue = 0

            # Line thickness based on confidence (thicker = more confident)
            line_weight = max(1, int(5 * (1 - confidence_rank)))
            popup_text = f"{edge['source']} ↔ {edge['target']}<br>Similarity: {similarity:.3f}<br>Confidence Rank: {i + 1}/{num_edges}"

        else:
            # SIMILARITY MODE: Color based on similarity value
            normalized_similarity = max(0, min(1, (similarity + 1) / 2))

            # Simple color mapping: red (low) to yellow (medium) to green (high)
            if normalized_similarity < 0.5:
                # Red to Yellow
                red = 255
                green = int(255 * (normalized_similarity * 2))
                blue = 0
            else:
                # Yellow to Green
                red = int(255 * (2 - normalized_similarity * 2))
                green = 255
                blue = 0

            # Line thickness based on similarity
            line_weight = max(1, int(5 * normalized_similarity))
            popup_text = f"{edge['source']} ↔ {edge['target']}<br>Similarity: {similarity:.3f}"

        color = f'#{red:02x}{green:02x}{blue:02x}'

        # Add edge as a simple line
        line = folium.PolyLine(
            locations=[
                [source_coords['lat'], source_coords['lon']],
                [target_coords['lat'], target_coords['lon']]
            ],
            color=color,
            weight=line_weight,
            opacity=0.8,
            popup=popup_text
        )
        line.add_to(m)

        plotted_edges.append({
            'source': edge['source'],
            'target': edge['target'],
            'similarity': similarity,
            'confidence_rank': i / max(1, num_edges - 1) if use_confidence_mode else None,
            'color': color,
            'mode': 'confidence' if use_confidence_mode else 'similarity'
        })

    # Add appropriate legend based on mode
    if use_confidence_mode:
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 180px; height: 140px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px;">
        <p><b>Confidence Gradient</b></p>
        <div style="background: linear-gradient(to right, #00ff00, #ffff00, #ff0000); 
                    height: 20px; width: 150px; border: 1px solid black;"></div>
        <div style="display: flex; justify-content: space-between; width: 150px; margin-top: 2px;">
            <span style="font-size: 10px;">High</span>
            <span style="font-size: 10px;">Medium</span>
            <span style="font-size: 10px;">Low</span>
        '''
    else:
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 180px; height: 140px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px;">
        <p><b>Similarity Values</b></p>
        <div style="background: linear-gradient(to right, #ff0000, #ffff00, #00ff00); 
                    height: 20px; width: 150px; border: 1px solid black;"></div>
        <div style="display: flex; justify-content: space-between; width: 150px; margin-top: 2px;">
            <span style="font-size: 10px;">Low</span>
            <span style="font-size: 10px;">Medium</span>
            <span style="font-size: 10px;">High</span>
        </div>
        <p style="margin-top: 10px;"><b>Municipalities</b></p>
        <p><span style="color:blue;">●</span> Connected</p>
        <p><span style="color:gray;">●</span> Unconnected</p>
        </div>
        '''

    m.get_root().html.add_child(folium.Element(legend_html))

    return m, plotted_edges


def create_similarity_distribution_plot(plotted_edges):
    """Create a histogram of similarity scores with confidence ranking."""
    if not plotted_edges:
        return None

    similarities = [edge['similarity'] for edge in plotted_edges]
    confidence_ranks = [edge.get('confidence_rank', 0) for edge in plotted_edges]

    # Create subplot with two histograms
    fig = go.Figure()

    # Similarity distribution
    fig.add_trace(go.Histogram(
        x=similarities,
        nbinsx=20,
        name="Similarity Scores",
        marker_color='lightblue',
        opacity=0.7
    ))

    fig.update_layout(
        title="Distribution of Municipality Similarity Scores",
        xaxis_title="Similarity Score",
        yaxis_title="Count",
        height=400
    )

    return fig


def main():
    # Sidebar for file selection and options
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

    # Map options
    st.sidebar.header("🗺️ Map Options")
    max_edges = st.sidebar.slider("Maximum edges to display", 5, 10000, 30,
                                  help="More edges = more complete view but slower performance")

    hide_unconnected = st.sidebar.checkbox(
        "Hide unconnected municipalities",
        value=False,
        help="Only show municipalities that have adjacency relationships"
    )

    use_confidence_mode = st.sidebar.checkbox(
        "Use relative coloring",
        value=True,
        help="If checked: Ranks colors by similarity from green=most similar to red=least similar on the map. "
             "Note that this applies ONLY to the edges displayed on the map. "
             "It conveys only relative edges like is there are 2 edges:"
             "\"A is similar to B\" and \"C is medium similar to D\" then A-B will be green and C-D will be red."
             "\nIf unchecked: colors based on similarity values as predicted by TransE (green=high similarity, red=low similarity)"
    )

    # Load data button
    if st.sidebar.button("🔄 Load Data"):
        st.cache_data.clear()

    # Main content
    with st.spinner("Loading data..."):
        triples, entities, relations = load_rdf_data(ttl_file)
        attributes_data = load_attributes_data(attributes_file)
        municipality_coords = get_austrian_municipality_coordinates()

    if triples is None:
        st.error(f"Could not load RDF data from {ttl_file}")
        st.stop()

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🗺️ Geographic Map", "📊 Similarity Analysis", "ℹ️ Data Info"])

    with tab1:
        st.header("Basic Geographic Network Map")
        if use_confidence_mode:
            st.write("**Confidence mode**: 🟢 Green = most confident connections, 🟡 Yellow = medium confidence, 🔴 Red = least confident connections.")
        else:
            st.write("**Similarity mode**: 🟢 Green = high similarity values, 🟡 Yellow = medium similarity, 🔴 Red = low similarity values.")
        st.write("Line thickness also indicates confidence/similarity level - thicker lines represent higher values.")

        # Calculate similarity if attributes are available
        similarity_dict = {}
        if attributes_data:
            with st.spinner("Calculating similarities..."):
                # Create entity-to-URI mapping
                entity_to_uri_map = {name: uri for uri, name in entities}

                # Create feature vectors
                feature_vectors, attribute_names = create_feature_vectors(attributes_data, entity_to_uri_map)

                if feature_vectors:
                    similarity_dict = calculate_similarity_matrix(feature_vectors)
                    st.success(f"✅ Calculated similarities for {len(feature_vectors)} municipalities")
                else:
                    st.warning("Could not create feature vectors from attributes data.")
        else:
            st.info("ℹ️ No attributes data - edges will be colored uniformly.")

        # Create and display the map
        with st.spinner("Creating map..."):
            map_result = create_basic_geographic_map(
                triples, municipality_coords, similarity_dict, max_edges, hide_unconnected, use_confidence_mode
            )

            if map_result:
                m, plotted_edges = map_result

                # Display the map with larger size
                map_data = st_folium(m, width=1400, height=1000, returned_objects=["last_clicked"])

                # Display statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total_munis = len(municipality_coords)
                    st.metric("Total Municipalities", total_munis)
                with col2:
                    connected_munis = len(get_connected_municipalities(triples))
                    st.metric("Connected", connected_munis)
                with col3:
                    st.metric("Edges Shown", len(plotted_edges))
                with col4:
                    if plotted_edges:
                        avg_similarity = np.mean([edge['similarity'] for edge in plotted_edges])
                        st.metric("Avg Similarity", f"{avg_similarity:.3f}")
            else:
                st.error("Could not create geographic visualization.")

    with tab2:
        st.header("Similarity Analysis")

        if 'plotted_edges' in locals() and plotted_edges:
            # Similarity distribution
            fig = create_similarity_distribution_plot(plotted_edges)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # Top similar pairs
            st.subheader("Most Similar Municipality Pairs")
            sorted_edges = sorted(plotted_edges, key=lambda x: x['similarity'], reverse=True)

            similarity_df = pd.DataFrame([
                {
                    'Municipality 1': edge['source'],
                    'Municipality 2': edge['target'],
                    'Similarity Score': f"{edge['similarity']:.4f}"
                }
                for edge in sorted_edges[:10]  # Show top 10
            ])

            st.dataframe(similarity_df, use_container_width=True)

            # Least similar pairs
            st.subheader("Least Similar Municipality Pairs")
            least_similar_df = pd.DataFrame([
                {
                    'Municipality 1': edge['source'],
                    'Municipality 2': edge['target'],
                    'Similarity Score': f"{edge['similarity']:.4f}"
                }
                for edge in sorted_edges[-10:]  # Show bottom 10
            ])

            st.dataframe(least_similar_df, use_container_width=True)
        else:
            st.info("Please generate the geographic map first to see similarity analysis.")

    with tab3:
        st.header("Data Information")

        # Dataset overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Entities", len(entities) if entities else 0)
        with col2:
            st.metric("Total Relations", len(relations) if relations else 0)
        with col3:
            st.metric("Municipalities with Coordinates", len(municipality_coords))

        # Available municipalities
        st.subheader("Available Municipalities")
        if municipality_coords:
            coords_df = pd.DataFrame([
                {
                    'Municipality': name,
                    'Latitude': f"{coords['lat']:.4f}",
                    'Longitude': f"{coords['lon']:.4f}"
                }
                for name, coords in municipality_coords.items()
            ])
            st.dataframe(coords_df, use_container_width=True)

        # Show sample relationships
        if triples:
            st.subheader("Sample Relationships")
            adjacency_triples = [t for t in triples if 'adjacent' in t['predicate_name'].lower()]
            if adjacency_triples:
                sample_df = pd.DataFrame([
                    {
                        'Source': t['subject_name'],
                        'Relationship': t['predicate_name'],
                        'Target': t['object_name']
                    }
                    for t in adjacency_triples[:10]
                ])
                st.dataframe(sample_df, use_container_width=True)


if __name__ == "__main__":
    main()