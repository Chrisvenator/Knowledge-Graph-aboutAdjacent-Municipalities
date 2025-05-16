from rdflib import Graph, Namespace, RDF, URIRef, Literal

g = Graph()
EX = Namespace("http://example.org/")
g.bind("ex", EX)

import json
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import XSD

# Load JSON from a file
with open('../adjacent_municipalities_names.json', 'r', encoding='utf-8') as file:
    data = json.load(file)


# Iterate and add triples
# create triples: (subject, predicate, object)

for city, suburbs in data.items():
    city_uri = EX[city.replace(" ", "_")]
    print(f"Top-level key: {city}")

    neighbor_count = len(suburbs)
    g.add((city_uri, EX.hasNeighborCount, Literal(neighbor_count, datatype=XSD.integer)))

    for suburb in suburbs:
        suburb_uri = EX[suburb.replace(" ", "_")]
        g.add((city_uri, EX.adjacent, suburb_uri))
        g.add((city_uri, RDF.type, EX.municipality))

g.serialize("adjacent_municipalities.ttl", format="turtle")