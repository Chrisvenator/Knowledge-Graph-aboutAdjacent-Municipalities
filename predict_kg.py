from pykeen.triples import TriplesFactory
from pykeen.models import TransE
import torch
import numpy as np
import json
import os
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Use trained knowledge graph embeddings for predictions")
    parser.add_argument("--model-dir", default="model", help="Directory with saved model and mappings")
    return parser.parse_args()


def load_model(model_dir, device):
    """Load a trained model and mappings from disk."""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} does not exist")

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
        raise NotImplementedError(f"Loading {model_config['model_class']} model is not implemented")

    # Create model instance - using the triples_factory rather than passing num_entities and num_relations
    model = model_class(
        triples_factory=dummy_tf,
        embedding_dim=model_config["embedding_dim"],
        scoring_fct_norm=model_config["scoring_fct_norm"],
        entity_initializer=None,  # Will be overwritten
        relation_initializer=None,  # Will be overwritten
    )

    # Load state dict
    model.load_state_dict(torch.load(os.path.join(model_dir, "model_state.pt"),
                                     map_location=device))
    model.to(device)

    print(f"Model loaded from {model_dir}")
    return model, dummy_tf


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


def predict_links(head, relation, entity_to_id, relation_to_id, model, entity_embeddings, relation_embeddings, device, top_n=3):
    """Use the model for link prediction."""
    if head not in entity_to_id or relation not in relation_to_id:
        print(f"Entity {head} or relation {relation} not found in the graph")
        return None, None

    head_id = entity_to_id[head]
    relation_id = relation_to_id[relation]

    # Use the model's scoring function directly
    try:
        # Create tensors for all possible combinations
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
        print(f"Error using model's scoring function: {e}")
        print("Falling back to manual scoring...")

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

    head_simple = head.split('/')[-1]
    relation_simple = relation.split('/')[-1]
    print(f"\nTop predictions for {head_simple} -> {relation_simple} -> ?")

    for i in range(min(top_n, len(indices))):
        idx = indices[i]
        tail = [k for k, v in entity_to_id.items() if v == idx][0]
        tail_simple = tail.split('/')[-1]
        # Convert score to float to avoid numpy formatting issues
        score_value = float(scores[idx])
        print(f"  {tail_simple} (score: {score_value:.4f})")

    return indices[:top_n], scores[indices[:top_n]]


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


def interactive_prediction(model, tf, device):
    """Interactive mode for making predictions with the model."""
    # Extract embeddings
    entity_embeddings, relation_embeddings = extract_embeddings(model)

    # Print stats about the loaded model
    print(f"\nUsing model with {len(tf.entity_to_id)} entities and {len(tf.relation_to_id)} relations")

    # Interactive mode for predictions
    print("\nEntering interactive prediction mode.")
    print("Type 'exit' to quit.")

    while True:
        print("\n=== Link Prediction ===")
        head = input("Enter head entity (full URI or just the name part): ")
        if head.lower() == 'exit':
            break

        # If only the name was entered, try to match it to a full URI
        if not head.startswith('http'):
            matching_entities = [e for e in tf.entity_to_id.keys() if head.lower() in e.lower()]
            if matching_entities:
                if len(matching_entities) > 1:
                    print("Multiple matches found:")
                    for i, entity in enumerate(matching_entities[:5]):
                        print(f"  {i + 1}. {entity}")
                    choice = input("Enter the number of your choice (or 'skip' to try another entity): ")
                    if choice.lower() == 'skip':
                        continue
                    try:
                        head = matching_entities[int(choice) - 1]
                    except (ValueError, IndexError):
                        print("Invalid choice. Try again.")
                        continue
                else:
                    head = matching_entities[0]
                    print(f"Using entity: {head}")
            else:
                print(f"No entities matching '{head}' found. Try again.")
                continue

        relation = input("Enter relation (full URI or just the name part): ")
        if relation.lower() == 'exit':
            break

        # If only the name was entered, try to match it to a full URI
        if not relation.startswith('http'):
            matching_relations = [r for r in tf.relation_to_id.keys() if relation.lower() in r.lower()]
            if matching_relations:
                if len(matching_relations) > 1:
                    print("Multiple matches found:")
                    for i, rel in enumerate(matching_relations):
                        print(f"  {i + 1}. {rel}")
                    choice = input("Enter the number of your choice (or 'skip' to try another relation): ")
                    if choice.lower() == 'skip':
                        continue
                    try:
                        relation = matching_relations[int(choice) - 1]
                    except (ValueError, IndexError):
                        print("Invalid choice. Try again.")
                        continue
                else:
                    relation = matching_relations[0]
                    print(f"Using relation: {relation}")
            else:
                print(f"No relations matching '{relation}' found. Try again.")
                continue

        # Number of predictions to show
        try:
            top_n = int(input("Number of predictions to show (default 5): ") or 5)
        except ValueError:
            top_n = 5

        # Perform prediction
        predict_links(head, relation, tf.entity_to_id, tf.relation_to_id,
                      model, entity_embeddings, relation_embeddings, device, top_n)

        # Option to find similar entities
        find_similar = input("\nDo you want to find entities similar to the head entity? (y/n): ")
        if find_similar.lower() == 'y':
            # Number of similar entities to show
            try:
                top_n = int(input("Number of similar entities to show (default 5): ") or 5)
            except ValueError:
                top_n = 5

            find_similar_entities(head, tf.entity_to_id, entity_embeddings, top_n)


def main():
    """Main function for using the trained model to make predictions."""
    args = parse_args()

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model, tf = load_model(args.model_dir, device)

    # Run interactive prediction
    interactive_prediction(model, tf, device)


if __name__ == "__main__":
    main()