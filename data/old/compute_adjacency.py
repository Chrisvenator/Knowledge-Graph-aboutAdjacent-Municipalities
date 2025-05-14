import geopandas as gpd
from libpysal.weights import Queen
import json

gdf = gpd.read_file("./gemeinden_95_topo.json")

w = Queen.from_dataframe(gdf)

with open("./adjacency_indices.json", "w", encoding="utf-8") as f:
    json.dump(w.neighbors, f, indent=2)

adjacency = {
    gdf.loc[i, "name"]: [gdf.loc[j, "name"] for j in neighbors]
    for i, neighbors in w.neighbors.items()
}

with open("./adjacency.json", "w", encoding="utf-8") as f:
    json.dump(adjacency, f, indent=2)

print("Adjacency mapping saved to 'adjacency.json'.")
