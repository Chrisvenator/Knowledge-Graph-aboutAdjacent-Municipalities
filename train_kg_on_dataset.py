from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from rdflib import Graph
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import json
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train knowledge graph embeddings for municipalities data")
    parser.add_argument("--input", default="dataset.ttl", help="Input TTL file with RDF triples")
    parser.add_argument("--output-image", default="Municipalities.png", help="Output visualization image")
    parser.add_argument("--model-dir", default="model", help="Directory to save the model")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--embedding-dim", type=int, default=50, help="Dimension of entity and relation embeddings")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    return parser.parse_args()


def save_model(model, tf, model_dir, embedding_dim):
    """Save the trained model and mappings to disk."""
    os.makedirs(model_dir, exist_ok=True)

    # Save the PyKEEN model state
    torch.save(model.state_dict(), os.path.join(model_dir, "model_state.pt"))

    # Save the model configuration
    model_config = {
        "model_class": model.__class__.__name__,
        "embedding_dim": embedding_dim,  # Use the provided embedding_dim parameter
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


def find_similar_entities(entity_label, entity_to_id, entity_embeddings, top_n=3):
    """Find similar entities based on embedding distance."""
    if entity_label not in entity_to_id:
        print(f"Entity {entity_label} not found in the graph")
        return

    entity_id = entity_to_id[entity_label]
    entity_emb = entity_embeddings[entity_id]

    # Calculate distances
    distances = np.linalg.norm(entity_embeddings - entity_emb, axis=1)

    # Sort by distance (excluding itself)
    indices = np.argsort(distances)

    simple_name = entity_label.split('/')[-1]
    print(f"\nEntities most similar to {simple_name}:")
    for i in range(1, min(top_n + 1, len(indices))):
        idx = indices[i]
        entity = [k for k, v in entity_to_id.items() if v == idx][0]
        simple_entity = entity.split('/')[-1]  # Extract name from URI
        # Convert distance to float to avoid numpy formatting issues
        distance_value = float(distances[idx])
        print(f"  {simple_entity} (distance: {distance_value:.4f})")


def visualize_embeddings(entity_to_id, entity_embeddings, output_file):
    """Visualize entity embeddings in 2D."""
    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        entity_embeddings_2d = pca.fit_transform(entity_embeddings)

        plt.figure(figsize=(10, 8))

        # Plot all points
        plt.scatter(entity_embeddings_2d[:, 0], entity_embeddings_2d[:, 1], c='blue', alpha=0.5)

        # Annotate each point with its entity label
        for entity, idx in entity_to_id.items():
            simple_name = entity.split('/')[-1]  # Extract name from URI
            plt.annotate(simple_name, (entity_embeddings_2d[idx, 0], entity_embeddings_2d[idx, 1]))

        plt.title('2D Projection of Entity Embeddings from TransE Model')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save the figure
        plt.savefig(output_file)
        print(f"\nEmbedding visualization saved as {output_file}")
    except ImportError:
        print("Visualization skipped - sklearn not available")
    except Exception as e:
        print(f"Error during visualization: {e}")


def train_model(input_file, model_dir, output_image, epochs, embedding_dim, batch_size, device):
    """Train a new knowledge graph embedding model."""
    # Check if the file exists in the current directory
    if not os.path.exists(input_file):
        print(f".ttl File not found. Exiting...")
        exit(1)

    # Load the RDF data from the TTL file
    g = Graph()
    g.parse(input_file, format="turtle")

    print(f"Loaded graph with {len(g)} triples")

    # Extract the triples from the graph
    triples = []
    for s, p, o in g:
        s_str = str(s)
        p_str = str(p)
        o_str = str(o)
        triples.append([s_str, p_str, o_str])  # Using lists instead of tuples

    print(f"Extracted {len(triples)} triples")

    # Convert to numpy array as required by PyKEEN
    triples_array = np.array(triples)

    # Create a TriplesFactory from the triples
    tf = TriplesFactory.from_labeled_triples(triples_array)

    # Since the dataset is small, we'll use all data for training
    training_tf = tf

    # Debug: Print the entity and relation mappings
    print("\nEntities:")
    entity_count = min(5, len(training_tf.entity_to_id))  # Show first 5 entities
    for i, (entity, idx) in enumerate(training_tf.entity_to_id.items()):
        if i >= entity_count:
            break
        print(f"  {entity} (ID: {idx})")
    print(f"  ... and {len(training_tf.entity_to_id) - entity_count} more entities")

    print("\nRelations:")
    for relation, idx in training_tf.relation_to_id.items():
        print(f"  {relation} (ID: {idx})")

    # Train a TransE model
    print("\nTraining TransE model...")
    result = pipeline(
        training=training_tf,
        testing=training_tf,  # Using same data for testing due to small dataset
        model='TransE',
        model_kwargs=dict(
            embedding_dim=embedding_dim,  # Dimension of the embeddings
            scoring_fct_norm=1,  # L1 distance
        ),
        optimizer_kwargs=dict(
            lr=0.01,  # Learning rate
        ),
        training_kwargs=dict(
            num_epochs=epochs,  # Number of training epochs
            batch_size=batch_size,  # Batch size
        ),
        evaluation_kwargs=dict(
            batch_size=batch_size,
        ),
        random_seed=42,  # For reproducibility
        device=device,  # Use GPU if available
    )

    # Access the trained model
    model = result.model
    print("\nModel training complete!")

    # Save the model for future use
    save_model(model, tf, model_dir, embedding_dim)

    # Get entity and relation embeddings
    entity_embeddings, relation_embeddings = extract_embeddings(model)

    # Example of entity lookup
    print("\nLooking up similar entities:")
    # Find municipalities in the dataset
    municipalities = [e for e in training_tf.entity_to_id if "Municipality" in e or "Gemeinde" in e]
    if municipalities:
        find_similar_entities(municipalities[0], training_tf.entity_to_id, entity_embeddings)
    else:
        # Try to find some meaningful entities
        for entity in list(training_tf.entity_to_id.keys())[:3]:
            find_similar_entities(entity, training_tf.entity_to_id, entity_embeddings)

    # Generate visualization
    visualize_embeddings(training_tf.entity_to_id, entity_embeddings, output_image)

    print("\nTraining metrics:")
    print(f"Final loss: {result.losses[-1]:.4f}")

    return model, tf


def main():
    """Main function for training a knowledge graph embedding model."""
    args = parse_args()

    # Check if GPU is available
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda')
    print(f"Using device: {device}")

    # Train a new model
    train_model(
        input_file=args.input,
        model_dir=args.model_dir,
        output_image=args.output_image,
        epochs=args.epochs,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        device=device
    )


if __name__ == "__main__":
    main()