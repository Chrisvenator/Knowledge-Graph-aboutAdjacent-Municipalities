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
   git clone https://github.com/Chrisvenator/Knowledge-Graph-aboutAdjacent-Municipalities
   cd Knowledge-Graph-aboutAdjacent-Municipalities
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

4. **Run the app**
   ```
   streamlit run kg_explorer_app.py
   ```

_You can use a custom model by pasting its folder into the root dir of this project and then 
specifying the folder name on the left side_

## Requirements

- streamlit
- torch
- pykeen>
- networkx
- matplotlib
- pandas
- Pillow
- numpy
- geopandas
- libpysal
- rdflib
- tqdm
- requests

## Usage Instructions

### __Network Graph Tab__ 
  - interactive Plotly-based network visualisation with  
  - Node sizing based on connection degree  
  - Colour-coded nodes based on connectivity  
  - Adjustable node limits and relation filtering  
  - Network statistics (density, average degree, etc.)


### Attributes Analysis Tab
  - comprehensive attribute analysis with  
  - Summary statistics for all attributes  
  - Interactive attribute selection and visualisation  
  - Histogram distributions for numeric attributes  
  - Sample-data viewing and full attribute tables  


### Similar Entities Tab
  - TransE-model-based similarity finding with  
  - Entity-embedding similarity calculation  
  - Configurable numb

### Entity Browser
1. Search entities by name  
2. View all matching entities and their URIs  


## Model Directory Structure

Your model directory should have the following files:

```
project-root/
├──dataset.ttl
├──dataset_entities_only.ttl
├──dataset_attributes.json
└──model/
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