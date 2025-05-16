from rdflib import Graph, Namespace, RDF, RDFS, Literal, URIRef
from rdflib.namespace import XSD
import json
import csv
import pathlib

"""
convert_data_to_triplets.py  –  builds an RDF graph that combines
* adjacency relations between Austrian municipalities,
* the 2019 Nationalrat election results,
* financial statistics for Austrian municipalities (revenues, expenditures …), and
* a code→name mapping for municipalities (C‑C41‑2 → Gemeindename).

The script writes the merged graph to ``dataset.ttl`` in Turtle format.
"""

###########################################################################
# 1) File locations – adjust if your files live elsewhere
###########################################################################
DATA_DIR = pathlib.Path("./data")
ADJACENT_FILE  = DATA_DIR / "adjacent_municipalities_names.json"
ELECTION_FILE  = DATA_DIR / "nrw2019.json"
FINANCE_FILE   = DATA_DIR / "Ausgaben_Oesterreichische_Gemeinden_2019.csv"   # uploaded file
MAPPING_FILE   = DATA_DIR / "Gemeinden-mapping-names-indicies.csv"           # new file (muni‑id→name)
OUTPUT_FILE    = "dataset.ttl"

###########################################################################
# 2) Helper functions
###########################################################################

def to_uri(label: str, namespace: Namespace) -> URIRef:
    """Return a URIRef from a label (spaces & special chars → underscores)."""
    return namespace[label.replace(" ", "_")]


def clean_bom(text: str) -> str:
    """Remove a potential UTF‑8 BOM character from the beginning of *text*."""
    return text.lstrip("\ufeff")


def parse_int(value: str) -> int:
    """Parse integer strings that may contain thousand‑separators or a BOM."""
    value = clean_bom(value.strip())
    if not value:
        raise ValueError("empty string")
    return int(value.replace(".", "").replace(",", ""))


def parse_float(value: str) -> float:
    """Parse floats with Austrian number formatting ("," as decimal separator)."""
    value = clean_bom(value.strip())
    if not value:
        raise ValueError("empty string")
    return float(value.replace(".", "").replace(",", "."))

###########################################################################
# 3) Load source data                                                     #
###########################################################################
print("Reading source files …")

# 3a) Adjacency list (name→[neighbour names])
with ADJACENT_FILE.open("r", encoding="utf-8") as fh:
    adjacency_data = json.load(fh)

# 3b) Election results
with ELECTION_FILE.open("r", encoding="utf-8") as fh:
    election_data = json.load(fh)["data"]

# 3c) Municipality‑ID→name mapping – open with utf‑8‑sig to drop BOM
id2name: dict[str, str] = {}
if MAPPING_FILE.exists():
    with MAPPING_FILE.open("r", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) < 2:
                continue
            muni_id = clean_bom(row[0].strip())
            muni_name = row[1].strip()
            if muni_id and muni_name:
                id2name[muni_id] = muni_name
else:
    print("WARNING: mapping file not found – municipality names might be missing in finance section.")

# 3d) Finance CSV – delimiter is ';' and quoted with '"'
finance_rows: list[dict] = []
with FINANCE_FILE.open("r", encoding="utf-8-sig") as fh:
    reader = csv.DictReader(fh, delimiter=";")
    for row in reader:
        finance_rows.append(row)

###########################################################################
# 4) RDF setup                                                             #
###########################################################################
print("Building graph …")

g = Graph()
EX = Namespace("http://example.org/")
g.bind("ex", EX)

###########################################################################
# 5) Municipality entities from mapping file                               #
###########################################################################
for muni_id, name in id2name.items():
    muni_uri = to_uri(name, EX)
    g.add((muni_uri, RDF.type, EX.Municipality))
    g.add((muni_uri, EX.municipalityId, Literal(parse_int(muni_id), datatype=XSD.integer)))
    g.add((muni_uri, RDFS.label, Literal(name)))

###########################################################################
# 6) Adjacency triples                                                     #
###########################################################################
for city, neighbours in adjacency_data.items():
    city_uri = to_uri(city, EX)
    g.add((city_uri, RDF.type, EX.Municipality))
    g.add((city_uri, EX.hasNeighborCount, Literal(len(neighbours), datatype=XSD.integer)))

    for neighbor in neighbours:
        neighbor_uri = to_uri(neighbor, EX)
        g.add((city_uri, EX.adjacent, neighbor_uri))

###########################################################################
# 7) Election‑result literals                                              #
###########################################################################
field_map_election = {
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
    "cpoe":     "votesCPOe",
}

for rec in election_data:
    city_uri = to_uri(rec["name"], EX)

    # spatial ID & type
    g.add((city_uri, EX.spatialId, Literal(rec["spatial_id"])))
    if rec.get("level") == "Gemeinde":
        g.add((city_uri, RDF.type, EX.Municipality))

    # numeric literals
    for json_key, prop in field_map_election.items():
        if (value := rec.get(json_key)) is not None:
            g.add((city_uri, EX[prop], Literal(value, datatype=XSD.integer)))

###########################################################################
# 8) Finance statistics                                                    #
###########################################################################

# Mapping: CSV column → (predicate, datatype parser)
field_map_finance = {
    "F-ANZAHL":       ("municipalityCount",     parse_int,   XSD.integer),
    "F-SUMME_1":      ("totalExpenditures",     parse_float, XSD.decimal),
    "F-SUMME_2":      ("totalRevenues",         parse_float, XSD.decimal),
    "F-SUMME_5":      ("currentTransfersOut",   parse_float, XSD.decimal),
    "F-SUMME_6":      ("investmentTransfersOut",parse_float, XSD.decimal),
    "F-SEIN_SUMME":   ("otherOwnRevenues",      parse_float, XSD.decimal),
    "F-SEIN_GEBUEHREN":("feeRevenues",          parse_float, XSD.decimal),
    "F-SEIN_ERTRAG":  ("operatingIncome",       parse_float, XSD.decimal),
    "F-SEIN_SPIELBANK":("casinoRevenue",        parse_float, XSD.decimal),
    "F-EINWOHNER":    ("population",            parse_int,   XSD.integer),
    "F-SCHU_SUMME":   ("debtTotal",             parse_float, XSD.decimal),
}

for row in finance_rows:
    # handle potential BOM in header names
    muni_id   = clean_bom(row.get("C-C41-2", row.get("\ufeffC-C41-2", "")).strip())
    year_code = clean_bom(row.get("C-A10-0", row.get("\ufeffC-A10-0", "")).strip())  # e.g. "A10-2019"
    if not muni_id or not year_code:
        continue
    year_part = year_code.split("-")[-1]

    # choose a human‑readable municipality URI if a name mapping exists
    if (muni_name := id2name.get(muni_id)):
        municipality_uri = to_uri(muni_name, EX)
    else:
        municipality_uri = EX[f"municipality_{muni_id}"]

    # create a unique resource for this muni‑year record
    record_uri = EX[f"MunicipalityFinancials_{muni_id}_{year_part}"]

    g.add((record_uri, RDF.type, EX.MunicipalityFinancials))
    g.add((record_uri, EX.forMunicipality, municipality_uri))
    g.add((record_uri, EX.year, Literal(int(year_part), datatype=XSD.gYear)))

    # numeric literals for every finance column
    for col, (prop, parser, dtype) in field_map_finance.items():
        raw_val = row.get(col) or row.get(f"\ufeff{col}")
        if raw_val and raw_val.strip():
            try:
                value_parsed = parser(raw_val)
            except ValueError:
                continue  # skip cells like "-" or "NA"
            g.add((record_uri, EX[prop], Literal(value_parsed, datatype=dtype)))

###########################################################################
# 9) Serialize                                                             #
###########################################################################
print(f"Writing graph to {OUTPUT_FILE} …")
g.serialize(OUTPUT_FILE, format="turtle")
print("Done.")
