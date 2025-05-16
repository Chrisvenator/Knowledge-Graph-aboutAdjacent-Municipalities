from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from rdflib import Graph
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

input_file = "dataset.ttl"
save_png_as = "Municipalities.png"

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
import numpy as np
triples_array = np.array(triples)

# Create a TriplesFactory from the triples
tf = TriplesFactory.from_labeled_triples(triples_array)

# Since the dataset is small, we'll use all data for training
training_tf = tf

# Debug: Print the entity and relation mappings
print("\nEntities:")
for entity, idx in training_tf.entity_to_id.items():
    print(f"  {entity} (ID: {idx})")

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
        embedding_dim=50,  # Dimension of the embeddings
        scoring_fct_norm=1,  # L1 distance
    ),
    optimizer_kwargs=dict(
        lr=0.01,  # Learning rate
    ),
    training_kwargs=dict(
        num_epochs=500,  # Number of training epochs
        batch_size=128,  # Batch size
    ),
    evaluation_kwargs=dict(
        batch_size=128,
    ),
    random_seed=42,  # For reproducibility
    device=device,  # Use GPU if available
)

# Access the trained model
model = result.model
print("\nModel training complete!")

# Debug: Print model structure
print("\nModel structure:")
print(f"Type: {type(model)}")
print(f"Model attributes: {dir(model)}")
print(f"Model state dict keys: {model.state_dict().keys()}")

# Get entity and relation mappings
entity_to_id = training_tf.entity_to_id
relation_to_id = training_tf.relation_to_id

# Get embeddings for entities and relations
try:
    # Try to get entity embeddings
    # For PyKEEN 1.0+
    entity_representation = model.entity_representations[0]
    if hasattr(entity_representation, 'weight'):
        entity_embeddings = entity_representation.weight.detach().cpu().numpy()
    elif hasattr(entity_representation, '_embeddings'):
        entity_embeddings = entity_representation._embeddings.weight.detach().cpu().numpy()
    else:
        # For different PyKEEN versions
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
        # For different PyKEEN versions
        print(f"Relation representation type: {type(relation_representation)}")
        print(f"Relation representation attributes: {dir(relation_representation)}")
        raise AttributeError("Could not find relation embeddings")

    print("\nSuccessfully extracted embeddings")
    print("Entity embeddings shape:", entity_embeddings.shape)
    print("Relation embeddings shape:", relation_embeddings.shape)

except Exception as e:
    print(f"Error extracting embeddings: {e}")
    # Create dummy embeddings for demonstration
    print("Creating dummy embeddings for demonstration")
    entity_embeddings = np.random.rand(len(entity_to_id), 50)
    relation_embeddings = np.random.rand(len(relation_to_id), 50)

# Print some statistics
print(f"\nNumber of entities: {len(entity_to_id)}")
print(f"Number of relations: {len(relation_to_id)}")

# Example: Find similar entities based on embedding distance
def find_similar_entities(entity_label, top_n=3):
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

# Example: Use the model for link prediction
def predict_links(head, relation):
    if head not in entity_to_id or relation not in relation_to_id:
        print(f"Entity {head} or relation {relation} not found in the graph")
        return

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

    for i in range(min(3, len(indices))):
        idx = indices[i]
        tail = [k for k, v in entity_to_id.items() if v == idx][0]
        tail_simple = tail.split('/')[-1]
        # Convert score to float to avoid numpy formatting issues
        score_value = float(scores[idx])
        print(f"  {tail_simple} (score: {score_value:.4f})")

# Example of entity lookup
print("\nLooking up similar entities:")
find_similar_entities("http://example.org/Paris")
find_similar_entities("http://example.org/EiffelTower")

# Example of link prediction
print("\nLink prediction examples:")
predict_links("http://example.org/EiffelTower", "http://example.org/located_in")
predict_links("http://example.org/Paris", "http://example.org/country")

# Visualizing entity embeddings (2D projection)
def visualize_embeddings():
    # Use PCA to reduce dimensionality to 2D for visualization
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
        plt.savefig(save_png_as)
        print("\nEmbedding visualization saved as {}".format(save_png_as))
    except ImportError:
        print("Visualization skipped - sklearn not available")
    except Exception as e:
        print(f"Error during visualization: {e}")

# Generate visualization
visualize_embeddings()

print("\nTraining metrics:")
print(f"Final loss: {result.losses[-1]:.4f}")