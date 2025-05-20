# Municipality Knowledge Graph Explorer

This Streamlit web application allows you to explore knowledge graph embeddings of municipalities and their relationships. You can make link predictions and find similar entities based on trained TransE embeddings.

## Features

- **Link Prediction**: Predict potential relationships between entities
- **Similar Entity Search**: Find municipalities with similar embedding vectors
- **Entity Browser**: Explore all entities in the knowledge graph
- **Graph Visualization**: Visual representations of prediction results

## Setup and Installation

1. **Clone the repository**
   ```
   git clone https://github.com/yourusername/municipality-kg-explorer.git
   cd municipality-kg-explorer
   ```

2. **Create a virtual environment**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

4. **Prepare your model data**
   Place your trained PyKEEN model and entity/relation mappings in a directory (default: "model").
   Your model directory should contain:
   - `model_state.pt`: The PyTorch state dict of your trained model
   - `model_config.json`: Configuration of your model including embedding dimension
   - `mappings.json`: Entity and relation mappings

   Alternatively, you can create sample data for testing:
   ```
   python create_sample_data.py
   ```

5. **Run the app**
   ```
   streamlit run kg_explorer_app.py
   ```

## Requirements

- Python 3.7+
- PyTorch
- PyKEEN
- Streamlit
- NetworkX
- Matplotlib
- Pandas
- Pillow

## Usage Instructions

### Link Prediction

1. Search and select a head entity (municipality)
2. Search and select a relation type
3. Click "Predict Links" to find potential tail entities
4. View results in table and graph format

### Similar Entity Search

1. Search and select an entity
2. Click "Find Similar Entities" to discover municipalities with similar embeddings
3. View results in table and graph format

### Entity Browser

1. Search entities by name
2. View all matching entities and their URIs

## Model Directory Structure

Your model directory should have the following files:

```
model/
├── model_state.pt      # PyTorch state dict of the trained model
├── model_config.json   # Model configuration (model_class, embedding_dim, etc.)
└── mappings.json       # Entity and relation mappings
```

The `model_config.json` should include:
```json
{
  "model_class": "TransE",
  "embedding_dim": 50,
  "scoring_fct_norm": 1
}
```

The `mappings.json` should include:
```json
{
  "entity_to_id": {
    "http://municipality.org/entity/Municipality_1": "0",
    ...
  },
  "relation_to_id": {
    "http://municipality.org/relation/adjacentTo": "0",
    ...
  }
}
```

## Troubleshooting

- **Model loading issues**: Ensure your model files follow the required structure
- **Memory errors**: Try reducing the number of entities or embedding dimension
- **Visualization problems**: Adjust graph parameters in the `visualize_graph` function