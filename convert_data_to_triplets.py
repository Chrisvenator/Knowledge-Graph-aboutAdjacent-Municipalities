from rdflib import Graph, Namespace, RDF, URIRef, Literal
import json
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import XSD




### 1) Read the two files --------------------------------------------
with open('./data/adjacent_municipalities_names.json', 'r', encoding='utf-8') as file:
    adjacency_data = json.load(file)                              # {city: [neighbours]}

with open("data/nrw2019.json", encoding="utf-8") as f:
    election_data = json.load(f)["data"]                # Nationalratswahl Daten


### 2) RDF setup -------------------------------------------------------
g = Graph()
EX = Namespace("http://example.org/")
g.bind("ex", EX)

def to_uri(label: str):
    """convert a place name to the same URI you used before"""
    return EX[label.replace(" ", "_")]

# Iterate and add triples
# create triples: (subject, predicate, object)

for city, neighbours in adjacency_data.items():
    city_uri = to_uri(city)
    g.add((city_uri, EX.hasNeighborCount,
           Literal(len(neighbours), datatype=XSD.integer)))

    for neighbor in neighbours:
        suburb_uri = EX[neighbor.replace(" ", "_")]
        g.add((city_uri, EX.adjacent, suburb_uri))
        g.add((city_uri, RDF.type, EX.municipality))


### 4) Add election-result literals -----------------------------------
# map JSON keys -> RDF property names that read nicely
field_map = {
    "eligible": "eligibleVoters",
    "votes":    "ballotsCast",
    "valid":    "validBallots",
    "invalid":  "invalidBallots",
    "oevp":     "votesOeVP",
    "spoe":     "votesSPOe",
    "fpoe":     "votesFPOe",
    "gruene":   "votesGruene",
    "neos":     "votesNeos",
    "pilz":     "votesPilz",
    "kpoe":     "votesKPOe",
    "wandl":    "votesWandl",
    "cpoe":     "votesCPOe"
}


for rec in election_data:
    city_uri = to_uri(rec["name"])

    # a) give every data row its spatial ID
    g.add((city_uri, EX.spatialId, Literal(rec["spatial_id"])))

    # b) type information – useful for SPARQL later
    if rec.get("level") == "Gemeinde":
        g.add((city_uri, RDF.type, EX.Municipality))

    # c) numeric literals for the result columns
    for json_key, prop in field_map.items():
        value = rec.get(json_key)
        if value is not None:                         # skip nulls
            g.add((city_uri, EX[prop], Literal(value, datatype=XSD.integer)))

### 5) (optional) Write out or query the graph -------------------------
g.serialize("dataset.ttl", format="turtle")