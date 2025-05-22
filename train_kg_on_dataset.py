from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from rdflib import Graph
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import json
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train knowledge graph embeddings for municipalities data")
    parser.add_argument("--input", default="dataset_entities_only.ttl", help="Input TTL file with entity-entity triples")
    parser.add_argument("--attributes", default="dataset_attributes.json", help="JSON file with entity attributes")
    parser.add_argument("--output-image", default="Municipalities.png", help="Output visualization image")
    parser.add_argument("--model-dir", default="model", help="Directory to save the model")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--embedding-dim", type=int, default=50, help="Dimension of entity and relation embeddings")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--use-attributes", action='store_true', help="Include attributes in training")
    return parser.parse_args()


def load_entity_attributes(attributes_file):
    """Load entity attributes from JSON file."""
    if not os.path.exists(attributes_file):
        print(f"Attributes file {attributes_file} not found. Training without attributes.")
        return {}

    with open(attributes_file, 'r', encoding='utf-8') as f:
        attributes = json.load(f)

    print(f"Loaded attributes for {len(attributes)} entities")
    return attributes


def create_attribute_features(entity_to_id, attributes):
    """Create numerical feature vectors from entity attributes."""
    feature_dict = {}
    all_features = []

    # Collect all possible attribute types
    all_attribute_types = set()
    for entity_attrs in attributes.values():
        all_attribute_types.update(entity_attrs.keys())

    print(f"Found {len(all_attribute_types)} different attribute types")

    # Create feature vectors for each entity
    for entity_uri, entity_id in entity_to_id.items():
        features = []
        entity_attrs = attributes.get(entity_uri, {})

        for attr_type in sorted(all_attribute_types):
            if attr_type in entity_attrs:
                attr_info = entity_attrs[attr_type]
                value = attr_info['value']
                datatype = attr_info.get('datatype', '')

                # Convert to numerical value based on datatype
                if 'integer' in datatype or 'positiveInteger' in datatype or 'nonNegativeInteger' in datatype:
                    try:
                        features.append(float(value))
                    except ValueError:
                        features.append(0.0)
                elif 'decimal' in datatype or 'double' in datatype or 'euro' in datatype:
                    try:
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

        feature_dict[entity_id] = np.array(features)
        all_features.append(features)

    # Standardize features
    if all_features:
        scaler = StandardScaler()
        standardized_features = scaler.fit_transform(all_features)

        # Update feature dictionary with standardized values
        for i, (entity_uri, entity_id) in enumerate(entity_to_id.items()):
            feature_dict[entity_id] = standardized_features[i]

    print(f"Created feature vectors of dimension {len(all_attribute_types)} for {len(feature_dict)} entities")
    return feature_dict, list(sorted(all_attribute_types))


def save_model(model, tf, model_dir, embedding_dim, attribute_features=None, attribute_names=None):
    """Save the trained model and mappings to disk."""
    os.makedirs(model_dir, exist_ok=True)

    # Save the PyKEEN model state
    torch.save(model.state_dict(), os.path.join(model_dir, "model_state.pt"))

    # Save the model configuration
    model_config = {
        "model_class": model.__class__.__name__,
        "embedding_dim": embedding_dim,
        "scoring_fct_norm": getattr(model, "scoring_fct_norm", 1),
        "entity_count": len(tf.entity_to_id),
        "relation_count": len(tf.relation_to_id),
    }

    with open(os.path.join(model_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    # Save the TriplesFactory mappings
    mappings = {
        "entity_to_id": {str(k): v for k, v in tf.entity_to_id.items()},
        "relation_to_id": {str(k): v for k, v in tf.relation_to_id.items()},
        "id_to_entity": {str(v): k for k, v in tf.entity_to_id.items()},
        "id_to_relation": {str(v): k for k, v in tf.relation_to_id.items()},
    }

    with open(os.path.join(model_dir, "mappings.json"), "w") as f:
        json.dump(mappings, f, indent=2)

    # Save attribute features if provided
    if attribute_features is not None:
        # Convert numpy arrays to lists for JSON serialization
        attr_features_serializable = {
            str(k): v.tolist() for k, v in attribute_features.items()
        }
        with open(os.path.join(model_dir, "attribute_features.json"), "w") as f:
            json.dump(attr_features_serializable, f, indent=2)

        if attribute_names is not None:
            with open(os.path.join(model_dir, "attribute_names.json"), "w") as f:
                json.dump(attribute_names, f, indent=2)

    print(f"Model and mappings saved to {model_dir}")


def extract_embeddings(model):
    """Extract entity and relation embeddings from the model."""
    try:
        # Try to get entity embeddings
        entity_representation = model.entity_representations[0]
        if hasattr(entity_representation, 'weight'):
            entity_embeddings = entity_representation.weight.detach().cpu().numpy()
        elif hasattr(entity_representation, '_embeddings'):
            entity_embeddings = entity_representation._embeddings.weight.detach().cpu().numpy()
        else:
            print(f"Entity representation type: {type(entity_representation)}")
            print(f"Entity representation attributes: {dir(entity_representation)}")
            raise AttributeError("Could not find entity embeddings")

        # Try to get relation embeddings
        relation_representation = model.relation_representations[0]
        if hasattr(relation_representation, 'weight'):
            relation_embeddings = relation_representation.weight.detach().cpu().numpy()
        elif hasattr(relation_representation, '_embeddings'):
            relation_embeddings = relation_representation._embeddings.weight.detach().cpu().numpy()
        else:
            print(f"Relation representation type: {type(relation_representation)}")
            print(f"Relation representation attributes: {dir(relation_representation)}")
            raise AttributeError("Could not find relation embeddings")

        print("\nSuccessfully extracted embeddings")
        print("Entity embeddings shape:", entity_embeddings.shape)
        print("Relation embeddings shape:", relation_embeddings.shape)

        return entity_embeddings, relation_embeddings

    except Exception as e:
        print(f"Error extracting embeddings: {e}")
        # Create dummy embeddings for demonstration
        print("Creating dummy embeddings for demonstration")
        entity_count = len(model.entity_representations[0])
        relation_count = len(model.relation_representations[0])
        embedding_dim = model.entity_representations[0].embedding_dim
        entity_embeddings = np.random.rand(entity_count, embedding_dim)
        relation_embeddings = np.random.rand(relation_count, embedding_dim)
        return entity_embeddings, relation_embeddings


def find_similar_entities(entity_label, entity_to_id, entity_embeddings, attribute_features=None, top_n=3):
    """Find similar entities based on embedding distance and optionally attribute similarity."""
    if entity_label not in entity_to_id:
        print(f"Entity {entity_label} not found in the graph")
        return

    entity_id = entity_to_id[entity_label]
    entity_emb = entity_embeddings[entity_id]

    # Calculate embedding distances
    embedding_distances = np.linalg.norm(entity_embeddings - entity_emb, axis=1)

    # If attributes are available, combine with attribute similarity
    if attribute_features is not None and entity_id in attribute_features:
        entity_attrs = attribute_features[entity_id]

        # Calculate attribute distances
        attr_distances = []
        for other_id in range(len(entity_embeddings)):
            if other_id in attribute_features:
                other_attrs = attribute_features[other_id]
                attr_dist = np.linalg.norm(entity_attrs - other_attrs)
                attr_distances.append(attr_dist)
            else:
                attr_distances.append(float('inf'))  # No attributes available

        attr_distances = np.array(attr_distances)

        # Normalize distances to [0, 1] range
        if np.max(embedding_distances) > 0:
            norm_emb_dist = embedding_distances / np.max(embedding_distances)
        else:
            norm_emb_dist = embedding_distances

        if np.max(attr_distances[attr_distances != float('inf')]) > 0:
            norm_attr_dist = attr_distances / np.max(attr_distances[attr_distances != float('inf')])
            norm_attr_dist[attr_distances == float('inf')] = 1.0
        else:
            norm_attr_dist = np.zeros_like(attr_distances)

        # Combine distances (equal weight)
        combined_distances = 0.5 * norm_emb_dist + 0.5 * norm_attr_dist
        distances = combined_distances
        print(f"\nEntities most similar to {entity_label.split('/')[-1]} (embedding + attributes):")
    else:
        distances = embedding_distances
        print(f"\nEntities most similar to {entity_label.split('/')[-1]} (embedding only):")

    # Sort by distance (excluding itself)
    indices = np.argsort(distances)

    for i in range(1, min(top_n + 1, len(indices))):
        idx = indices[i]
        entity = [k for k, v in entity_to_id.items() if v == idx][0]
        simple_entity = entity.split('/')[-1]  # Extract name from URI
        distance_value = float(distances[idx])
        print(f"  {simple_entity} (distance: {distance_value:.4f})")


def visualize_embeddings(entity_to_id, entity_embeddings, output_file, attribute_features=None):
    """Visualize entity embeddings in 2D, optionally enhanced with attributes."""
    try:
        # Combine embeddings with attributes if available
        if attribute_features is not None:
            print("Combining embeddings with attribute features for visualization...")
            combined_features = []
            for entity, idx in entity_to_id.items():
                emb = entity_embeddings[idx]
                if idx in attribute_features:
                    attrs = attribute_features[idx]
                    combined = np.concatenate([emb, attrs])
                else:
                    # Pad with zeros if no attributes
                    max_attr_dim = max(len(attr_features[0]) for attr_features in attribute_features.values())
                    attrs = np.zeros(max_attr_dim)
                    combined = np.concatenate([emb, attrs])
                combined_features.append(combined)

            features_to_plot = np.array(combined_features)
            title_suffix = " (Embeddings + Attributes)"
        else:
            features_to_plot = entity_embeddings
            title_suffix = " (Embeddings Only)"

        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_to_plot)

        plt.figure(figsize=(12, 10))

        # Plot all points
        plt.scatter(features_2d[:, 0], features_2d[:, 1], c='blue', alpha=0.6, s=50)

        # Annotate points with entity labels (only show a subset to avoid clutter)
        entity_items = list(entity_to_id.items())
        step = max(1, len(entity_items) // 50)  # Show at most 50 labels

        for i in range(0, len(entity_items), step):
            entity, idx = entity_items[i]
            simple_name = entity.split('/')[-1][:15]  # Truncate long names
            plt.annotate(simple_name, (features_2d[idx, 0], features_2d[idx, 1]),
                         fontsize=8, alpha=0.7)

        plt.title(f'2D Projection of Municipality Embeddings{title_suffix}')
        plt.xlabel(f'PCA Component 1 (explains {pca.explained_variance_ratio_[0]:.1%} of variance)')
        plt.ylabel(f'PCA Component 2 (explains {pca.explained_variance_ratio_[1]:.1%} of variance)')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()

        # Save the figure
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nEmbedding visualization saved as {output_file}")

    except ImportError:
        print("Visualization skipped - sklearn not available")
    except Exception as e:
        print(f"Error during visualization: {e}")


def train_model(input_file, attributes_file, model_dir, output_image, epochs, embedding_dim, batch_size, device, use_attributes):
    """Train a new knowledge graph embedding model."""
    # Check if the file exists
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found. Exiting...")
        exit(1)

    # Load the RDF data from the TTL file (entity relations only)
    g = Graph()
    g.parse(input_file, format="turtle")
    print(f"Loaded graph with {len(g)} triples")

    # Extract the triples from the graph
    triples = []
    for s, p, o in g:
        s_str = str(s)
        p_str = str(p)
        o_str = str(o)
        triples.append([s_str, p_str, o_str])

    print(f"Extracted {len(triples)} entity-entity triples")

    # Convert to numpy array as required by PyKEEN
    triples_array = np.array(triples)

    # Create a TriplesFactory from the triples
    tf = TriplesFactory.from_labeled_triples(triples_array)

    # Load entity attributes if requested
    attribute_features = None
    attribute_names = None
    if use_attributes:
        attributes = load_entity_attributes(attributes_file)
        if attributes:
            attribute_features, attribute_names = create_attribute_features(tf.entity_to_id, attributes)

    # Debug: Print the entity and relation mappings
    print(f"\nDataset Summary:")
    print(f"  Entities: {len(tf.entity_to_id)}")
    print(f"  Relations: {len(tf.relation_to_id)}")
    print(f"  Triples: {len(tf.mapped_triples)}")

    print("\nSample entities:")
    for i, (entity, idx) in enumerate(list(tf.entity_to_id.items())[:5]):
        simple_name = entity.split('/')[-1]
        print(f"  {simple_name} (ID: {idx})")

    print("\nRelations:")
    for relation, idx in tf.relation_to_id.items():
        simple_relation = relation.split('/')[-1]
        print(f"  {simple_relation} (ID: {idx})")

    # Train a TransE model
    print(f"\nTraining TransE model (embedding_dim={embedding_dim}, epochs={epochs})...")
    result = pipeline(
        training=tf,
        testing=tf,  # Using same data for testing due to small dataset
        model='TransE',
        model_kwargs=dict(
            embedding_dim=embedding_dim,
            scoring_fct_norm=1,  # L1 distance
        ),
        optimizer_kwargs=dict(
            lr=0.01,
        ),
        training_kwargs=dict(
            num_epochs=epochs,
            batch_size=batch_size,
        ),
        evaluation_kwargs=dict(
            batch_size=batch_size,
        ),
        random_seed=42,
        device=device,
    )

    # Access the trained model
    model = result.model
    print("\nModel training complete!")

    # Save the model for future use
    save_model(model, tf, model_dir, embedding_dim, attribute_features, attribute_names)

    # Get entity and relation embeddings
    entity_embeddings, relation_embeddings = extract_embeddings(model)

    # Example similarity analysis
    print("\nAnalyzing entity similarities:")
    sample_entities = list(tf.entity_to_id.keys())[:3]
    for entity in sample_entities:
        find_similar_entities(entity, tf.entity_to_id, entity_embeddings, attribute_features)

    # Generate visualization
    visualize_embeddings(tf.entity_to_id, entity_embeddings, output_image, attribute_features)

    print(f"\nTraining completed successfully!")
    print(f"Final training metrics available in result object")

    return model, tf


def main():
    """Main function for training a knowledge graph embedding model."""
    args = parse_args()

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Train a new model
    train_model(
        input_file=args.input,
        attributes_file=args.attributes,
        model_dir=args.model_dir,
        output_image=args.output_image,
        epochs=args.epochs,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        device=device,
        use_attributes=args.use_attributes
    )


if __name__ == "__main__":
    main()