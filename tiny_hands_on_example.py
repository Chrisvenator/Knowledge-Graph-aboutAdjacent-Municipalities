# pip install rdflib
from rdflib import Graph, Namespace, RDF, URIRef, Literal

g = Graph()
EX = Namespace("http://example.org/")

# --- create triples (subject, predicate, object) ---
g.add((EX.EiffelTower, RDF.type, EX.Monument))
g.add((EX.EiffelTower, EX.located_in, EX.Paris))
g.add((EX.Paris, RDF.type, EX.City))
g.add((EX.Paris, EX.country, EX.France))
g.add((EX.France, RDF.type, EX.Country))

g.add((EX.Riesenrad, RDF.type, EX.Monument))
g.add((EX.Riesenrad, EX.located_in, EX.Vienna))
g.add((EX.Vienna, RDF.type, EX.City))
g.add((EX.Vienna, EX.country, EX.Austria))
g.add((EX.Austria, RDF.type, EX.Country))
# --- save to disk ---
g.serialize("tiny_hands_on_example.ttl", format="turtle")

# --- ask a question ---
q = """
PREFIX ex: <http://example.org/>
SELECT ?monument ?city
WHERE { ?monument ex:located_in ?city . }
"""
for row in g.query(q):
    print(f"{row.monument.split('/')[-1]} is in {row.city.split('/')[-1]}")
