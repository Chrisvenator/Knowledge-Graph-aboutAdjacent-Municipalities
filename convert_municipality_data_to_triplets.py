"""
convert_data_to_triplets.py  –  builds an RDF graph that combines
* adjacency relations between Austrian municipalities,
* the 2019 Nationalrat election results,
* financial statistics for Austrian municipalities (revenues, expenditures …), and
* a code→name mapping for municipalities (C‑C41‑2 → Gemeindename).

The script writes the merged graph to ``dataset.ttl`` in Turtle format.
"""

from rdflib import Graph, Namespace, RDF, RDFS, Literal, URIRef
from rdflib.namespace import XSD, VOID, DCTERMS
from datetime import datetime
import json
import csv
import pathlib
import sys
import argparse
import logging
from typing import Optional, Tuple, Dict, Any, List, Callable
from tqdm import tqdm
from pykeen.triples import TriplesFactory

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


###########################################################################
# 1) Command line arguments and file locations
###########################################################################

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert Austrian municipalities data to RDF")
    parser.add_argument("--data-dir", default="./data", help="Directory containing source data files")
    parser.add_argument("--output", default="dataset.ttl", help="Output RDF file")
    parser.add_argument("--format", default="turtle", choices=["turtle", "xml", "nt", "json-ld"],
                        help="RDF serialization format")
    parser.add_argument("--batch-size", type=int, default=10000,
                        help="Batch size for adding triples to the graph")
    return parser.parse_args()


args = parse_args()
DATA_DIR = pathlib.Path(args.data_dir)
OUTPUT_FILE = args.output
RDF_FORMAT = args.format
BATCH_SIZE = args.batch_size

# File paths
ADJACENT_FILE = DATA_DIR / "adjacent_municipalities_names.json"
ELECTION_FILE = DATA_DIR / "nrw2019.json"
FINANCE_FILE = DATA_DIR / "Ausgaben_Oesterreichische_Gemeinden_2019.csv"
MAPPING_FILE = DATA_DIR / "Gemeinden-mapping-names-indicies.csv"


###########################################################################
# 2) Helper functions
###########################################################################

def to_uri(label: str, namespace: Namespace) -> URIRef:
    """Return a URIRef from a label (spaces & special chars → underscores)."""
    return namespace[label.replace(" ", "_").replace("-", "_")]


def clean_bom(text: str) -> str:
    """Remove a potential UTF‑8 BOM character from the beginning of *text*."""
    return text.lstrip("\ufeff")


def parse_int_safe(value: str) -> Optional[int]:
    """Safely parse integer strings that may contain thousand‑separators or a BOM."""
    try:
        value = clean_bom(value.strip())
        if not value:
            return None  # Return None for empty values only
        return int(value.replace(".", "").replace(",", ""))
    except (ValueError, TypeError):
        return None


def parse_float_safe(value: str) -> Optional[float]:
    """Safely parse floats with Austrian number formatting ("," as decimal separator)."""
    try:
        value = clean_bom(value.strip())
        if not value:
            return None  # Return None for empty values only
        return float(value.replace(".", "").replace(",", "."))
    except (ValueError, TypeError):
        return None

class BatchedGraphUpdate:
    def __init__(self):
        self.buffer = []          # Liste von Tripeln

    def add(self, s, p, o):
        self.buffer.append((s, p, o))

    def commit(self, graph: Graph):
        for triple in self.buffer:
            graph.add(triple)
        self.buffer.clear()


def add_typed_literal(batch: BatchedGraphUpdate, subject: URIRef, predicate: URIRef,
                     value: Any, datatype: URIRef) -> None:
    """Add a triple with an explicitly typed literal, ensuring no numbers become entities."""
    if value is not None:
        # Ensure the literal is properly typed
        literal = Literal(value, datatype=datatype)
        batch.add(subject, predicate, literal)


def get_municipality_uri(identifier: str, name: Optional[str] = None) -> URIRef:
    """Get a consistent URI for a municipality, preferring name-based URIs."""
    if name:
        return to_uri(name, EX)
    return EX[f"municipality_{identifier}"]


def validate_files() -> None:
    """Validate that all required files exist."""
    for file_path, required in [
        (ADJACENT_FILE, True),
        (ELECTION_FILE, True),
        (FINANCE_FILE, True),
        (MAPPING_FILE, False)
    ]:
        if not file_path.exists() and required:
            raise FileNotFoundError(f"Required file not found: {file_path}")
        elif not file_path.exists():
            logger.warning(f"Optional file not found: {file_path}")


class BatchedGraphUpdate:
    """Helper class for batched updates to the RDF graph."""

    def __init__(self, graph: Graph, batch_size: int = 10000):
        self.graph = graph
        self.batch_size = batch_size
        self.triples_batch = []

    def add(self, s: Any, p: Any, o: Any) -> None:
        """Add a triple to the batch, flushing if batch size is reached."""
        self.triples_batch.append((s, p, o))
        if len(self.triples_batch) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        """Add all batched triples to the graph."""
        if self.triples_batch:
            for s, p, o in self.triples_batch:
                self.graph.add((s, p, o))
            self.triples_batch.clear()


###########################################################################
# 3) Load source data                                                     #
###########################################################################

def load_data():
    """Load all data sources and return them."""
    try:
        # Validate files first
        validate_files()

        # 3a) Adjacency list (name→[neighbour names])
        logger.info("Loading adjacency data...")
        with ADJACENT_FILE.open("r", encoding="utf-8") as fh:
            try:
                adjacency_data = json.load(fh)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing adjacency JSON: {e}")
                raise

        # 3b) Election results
        logger.info("Loading election results...")
        with ELECTION_FILE.open("r", encoding="utf-8") as fh:
            try:
                election_data = json.load(fh)["data"]
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error parsing election JSON: {e}")
                raise

        # 3c) Municipality‑ID→name mapping
        logger.info("Loading municipality ID to name mapping...")
        id2name: Dict[str, str] = {}
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
            logger.warning("Mapping file not found – municipality names might be missing in finance section.")

        # 3d) Finance CSV
        logger.info("Loading finance data...")
        finance_rows: List[Dict] = []
        with FINANCE_FILE.open("r", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh, delimiter=";")
            for row in reader:
                finance_rows.append(row)

        return adjacency_data, election_data, id2name, finance_rows

    except Exception as e:
        logger.error(f"Fatal error loading data: {e}")
        sys.exit(1)


###########################################################################
# 4) Build RDF Graph                                                      #
###########################################################################

def build_graph(adjacency_data, election_data, id2name, finance_rows):
    """Build the complete RDF graph from all data sources."""
    logger.info("Building RDF graph...")

    g = Graph()
    g.bind("atmun", EX)
    g.bind("void", VOID)
    g.bind("dcterms", DCTERMS)
    g.bind("dbpedia-dt", Namespace("http://dbpedia.org/datatype/"))

    # Create batched update helper
    batch = BatchedGraphUpdate(g, BATCH_SIZE)

    # Track data quality issues
    municipalities_without_name = set()
    municipalities_without_election_data = set()
    municipalities_without_adjacency = set()
    municipalities_without_finances = set()

    # Track all municipality URIs for stats and quality checks
    all_municipality_uris = set()

    ###########################################################################
    # 4a) Add schema definitions
    ###########################################################################
    logger.info("Adding schema definitions...")

    # Class definitions
    batch.add(EX.Municipality, RDF.type, RDFS.Class)
    batch.add(EX.Municipality, RDFS.label, Literal("Municipality"))
    batch.add(EX.Municipality, RDFS.comment, Literal("An Austrian municipality"))

    batch.add(EX.MunicipalityFinancials, RDF.type, RDFS.Class)
    batch.add(EX.MunicipalityFinancials, RDFS.label, Literal("Municipality Financials"))
    batch.add(EX.MunicipalityFinancials, RDFS.comment, Literal("Financial data for an Austrian municipality in a specific year"))

    # Define custom datatypes
    EUR = Namespace("http://dbpedia.org/datatype/")["euro"]

    # Property definitions with explicit ranges
    properties = [
        ("adjacent", EX.Municipality, EX.Municipality, "adjacent to"),
        ("hasNeighborCount", EX.Municipality, XSD.nonNegativeInteger, "number of neighboring municipalities"),
        ("municipalityId", EX.Municipality, XSD.positiveInteger, "official municipality ID"),
        ("spatialId", EX.Municipality, XSD.string, "spatial identifier"),
        ("forMunicipality", EX.MunicipalityFinancials, EX.Municipality, "municipality this financial data belongs to"),
        ("year", EX.MunicipalityFinancials, XSD.gYear, "year of the financial data"),
        ("population", EX.MunicipalityFinancials, XSD.nonNegativeInteger, "population count"),
        ("electionParticipationRate", EX.Municipality, XSD.decimal, "election participation rate"),
        # Financial properties
        ("municipalityCount", EX.MunicipalityFinancials, XSD.positiveInteger, "number of municipalities in aggregation"),
        ("totalExpenditures", EX.MunicipalityFinancials, EUR, "total expenditures in euros"),
        ("totalRevenues", EX.MunicipalityFinancials, EUR, "total revenues in euros"),
        ("currentTransfersOut", EX.MunicipalityFinancials, EUR, "current transfers out in euros"),
        ("investmentTransfersOut", EX.MunicipalityFinancials, EUR, "investment transfers out in euros"),
        ("otherOwnRevenues", EX.MunicipalityFinancials, EUR, "other own revenues in euros"),
        ("feeRevenues", EX.MunicipalityFinancials, EUR, "fee revenues in euros"),
        ("operatingIncome", EX.MunicipalityFinancials, EUR, "operating income in euros"),
        ("casinoRevenue", EX.MunicipalityFinancials, EUR, "casino revenue in euros"),
        ("debtTotal", EX.MunicipalityFinancials, EUR, "total debt in euros"),
        # Election properties
        ("eligibleVoters", EX.Municipality, XSD.nonNegativeInteger, "number of eligible voters"),
        ("ballotsCast", EX.Municipality, XSD.nonNegativeInteger, "number of ballots cast"),
        ("validBallots", EX.Municipality, XSD.nonNegativeInteger, "number of valid ballots"),
        ("invalidBallots", EX.Municipality, XSD.nonNegativeInteger, "number of invalid ballots"),
        ("votesOeVP", EX.Municipality, XSD.nonNegativeInteger, "votes for ÖVP"),
        ("votesSPOe", EX.Municipality, XSD.nonNegativeInteger, "votes for SPÖ"),
        ("votesFPOe", EX.Municipality, XSD.nonNegativeInteger, "votes for FPÖ"),
        ("votesGruene", EX.Municipality, XSD.nonNegativeInteger, "votes for Grüne"),
        ("votesNeos", EX.Municipality, XSD.nonNegativeInteger, "votes for NEOS"),
        ("votesPilz", EX.Municipality, XSD.nonNegativeInteger, "votes for Pilz"),
        ("votesKPOe", EX.Municipality, XSD.nonNegativeInteger, "votes for KPÖ"),
        ("votesWandl", EX.Municipality, XSD.nonNegativeInteger, "votes for Wandl"),
        ("votesCPOe", EX.Municipality, XSD.nonNegativeInteger, "votes for CPÖ"),
    ]

    for prop, domain, range_val, label in properties:
        batch.add(EX[prop], RDF.type, RDF.Property)
        batch.add(EX[prop], RDFS.domain, domain)
        batch.add(EX[prop], RDFS.range, range_val)
        batch.add(EX[prop], RDFS.label, Literal(label))

    ###########################################################################
    # 4b) Municipality entities from mapping file
    ###########################################################################
    logger.info("Adding municipality base data...")

    for muni_id, name in tqdm(id2name.items(), desc="Processing municipality mappings"):
        muni_uri = get_municipality_uri(muni_id, name)
        batch.add(muni_uri, RDF.type, EX.Municipality)

        # Only add municipality ID if it's valid and positive
        parsed_id = parse_int_safe(muni_id)
        if parsed_id is not None and parsed_id > 0:
            add_typed_literal(batch, muni_uri, EX.municipalityId, parsed_id, XSD.positiveInteger)

        batch.add(muni_uri, RDFS.label, Literal(name))
        all_municipality_uris.add(muni_uri)

    ###########################################################################
    # 4c) Adjacency triples
    ###########################################################################
    logger.info("Adding adjacency relationships...")

    for city, neighbours in tqdm(adjacency_data.items(), desc="Processing adjacency data"):
        city_uri = get_municipality_uri("", city)  # Use name-based URI for adjacency data
        batch.add(city_uri, RDF.type, EX.Municipality)

        # Always add neighbor count with explicit typing (including 0)
        neighbor_count = len(neighbours)
        add_typed_literal(batch, city_uri, EX.hasNeighborCount, neighbor_count, XSD.nonNegativeInteger)

        batch.add(city_uri, RDFS.label, Literal(city))
        all_municipality_uris.add(city_uri)

        for neighbor in neighbours:
            neighbor_uri = get_municipality_uri("", neighbor)
            batch.add(city_uri, EX.adjacent, neighbor_uri)
            all_municipality_uris.add(neighbor_uri)

    # Find municipalities without adjacency data
    municipalities_with_adjacency = set(adjacency_data.keys())
    for muni_uri in all_municipality_uris:
        muni_label = str(g.value(muni_uri, RDFS.label, None) or "").strip()
        if muni_label and muni_label not in municipalities_with_adjacency:
            municipalities_without_adjacency.add(muni_label)

    ###########################################################################
    # 4d) Election‑result literals
    ###########################################################################
    logger.info("Adding election result data...")

    field_map_election = {
        "eligible": "eligibleVoters",
        "votes": "ballotsCast",
        "valid": "validBallots",
        "invalid": "invalidBallots",
        "oevp": "votesOeVP",
        "spoe": "votesSPOe",
        "fpoe": "votesFPOe",
        "gruene": "votesGruene",
        "neos": "votesNeos",
        "pilz": "votesPilz",
        "kpoe": "votesKPOe",
        "wandl": "votesWandl",
        "cpoe": "votesCPOe",
    }

    # Track municipalities with election data
    municipalities_with_election_data = set()

    for rec in tqdm(election_data, desc="Processing election data"):
        city_name = rec["name"]
        city_uri = get_municipality_uri("", city_name)
        municipalities_with_election_data.add(city_name)

        # spatial ID as string literal
        if rec.get("spatial_id"):
            batch.add(city_uri, EX.spatialId, Literal(str(rec["spatial_id"]), datatype=XSD.string))

        if rec.get("level") == "Gemeinde":
            batch.add(city_uri, RDF.type, EX.Municipality)
            all_municipality_uris.add(city_uri)

        # numeric literals with explicit typing - include all values including 0
        for json_key, prop in field_map_election.items():
            value = rec.get(json_key)
            if value is not None:
                # Ensure value is integer and properly typed
                int_value = int(value) if isinstance(value, (int, float)) else parse_int_safe(str(value))
                if int_value is not None:
                    add_typed_literal(batch, city_uri, EX[prop], int_value, XSD.nonNegativeInteger)

        # Calculate election participation rate with explicit decimal typing
        eligible = rec.get("eligible")
        votes = rec.get("votes")
        if eligible is not None and votes is not None and eligible > 0:
            participation_rate = float(votes) / float(eligible)
            add_typed_literal(batch, city_uri, EX.electionParticipationRate,
                            round(participation_rate, 6), XSD.decimal)

    # Find municipalities without election data
    for muni_uri in all_municipality_uris:
        muni_label = str(g.value(muni_uri, RDFS.label, None) or "").strip()
        if muni_label and muni_label not in municipalities_with_election_data:
            municipalities_without_election_data.add(muni_label)

    ###########################################################################
    # 4e) Finance statistics
    ###########################################################################
    logger.info("Adding finance data...")

    # Define custom EUR datatype
    EUR = Namespace("http://dbpedia.org/datatype/")["euro"]

    # Mapping: CSV column → (predicate, datatype parser, XML Schema datatype)
    field_map_finance = {
        "F-ANZAHL": ("municipalityCount", parse_int_safe, XSD.positiveInteger),
        "F-SUMME_1": ("totalExpenditures", parse_float_safe, EUR),
        "F-SUMME_2": ("totalRevenues", parse_float_safe, EUR),
        "F-SUMME_5": ("currentTransfersOut", parse_float_safe, EUR),
        "F-SUMME_6": ("investmentTransfersOut", parse_float_safe, EUR),
        "F-SEIN_SUMME": ("otherOwnRevenues", parse_float_safe, EUR),
        "F-SEIN_GEBUEHREN": ("feeRevenues", parse_float_safe, EUR),
        "F-SEIN_ERTRAG": ("operatingIncome", parse_float_safe, EUR),
        "F-SEIN_SPIELBANK": ("casinoRevenue", parse_float_safe, EUR),
        "F-EINWOHNER": ("population", parse_int_safe, XSD.nonNegativeInteger),
        "F-SCHU_SUMME": ("debtTotal", parse_float_safe, EUR),
    }

    # Track municipalities with finance data
    municipalities_with_finances = set()

    for row in tqdm(finance_rows, desc="Processing finance data"):
        # handle potential BOM in header names
        muni_id = clean_bom(row.get("C-C41-2", row.get("\ufeffC-C41-2", "")).strip())
        year_code = clean_bom(row.get("C-A10-0", row.get("\ufeffC-A10-0", "")).strip())  # e.g. "A10-2019"
        if not muni_id or not year_code:
            continue

        year_part = year_code.split("-")[-1]

        # Track municipalities without names in the mapping
        if muni_id not in id2name:
            municipalities_without_name.add(muni_id)

        # choose a human‑readable municipality URI if a name mapping exists
        muni_name = id2name.get(muni_id)
        municipality_uri = get_municipality_uri(muni_id, muni_name)

        if muni_name:
            municipalities_with_finances.add(muni_name)

        # create a unique resource for this muni‑year record
        record_uri = EX[f"MunicipalityFinancials_{muni_id}_{year_part}"]

        batch.add(record_uri, RDF.type, EX.MunicipalityFinancials)
        batch.add(record_uri, EX.forMunicipality, municipality_uri)
        add_typed_literal(batch, record_uri, EX.year, int(year_part), XSD.gYear)

        # numeric literals for every finance column with explicit typing
        for col, (prop, parser, dtype) in field_map_finance.items():
            raw_val = row.get(col) or row.get(f"\ufeff{col}")
            if raw_val and raw_val.strip():
                value_parsed = parser(raw_val)
                if value_parsed is not None:
                    add_typed_literal(batch, record_uri, EX[prop], value_parsed, dtype)

    # Find municipalities without finance data
    for muni_uri in all_municipality_uris:
        muni_label = str(g.value(muni_uri, RDFS.label, None) or "").strip()
        if muni_label and muni_label not in municipalities_with_finances:
            municipalities_without_finances.add(muni_label)

    ###########################################################################
    # 4f) Dataset metadata
    ###########################################################################
    logger.info("Adding dataset metadata...")

    dataset_uri = EX.AustrianMunicipalitiesDataset
    batch.add(dataset_uri, RDF.type, VOID.Dataset)
    batch.add(dataset_uri, VOID.dataDump, URIRef(f"file://{OUTPUT_FILE}"))
    add_typed_literal(batch, dataset_uri, DCTERMS.created, datetime.now().isoformat(), XSD.dateTime)
    batch.add(dataset_uri, DCTERMS.creator, Literal("RDF Conversion Script"))
    batch.add(dataset_uri, DCTERMS.description, Literal(
        "Austrian municipalities data including adjacency, election results, and finances"
    ))
    add_typed_literal(batch, dataset_uri, VOID.entities, len(all_municipality_uris), XSD.nonNegativeInteger)

    # Flush any remaining triples
    batch.flush()

    # Log data quality issues
    logger.info(f"Found {len(municipalities_without_name)} municipalities without names in the mapping")
    logger.info(f"Found {len(municipalities_without_election_data)} municipalities without election data")
    logger.info(f"Found {len(municipalities_without_adjacency)} municipalities without adjacency data")
    logger.info(f"Found {len(municipalities_without_finances)} municipalities without finance data")

    return g


###########################################################################
# 5) Main function                                                        #
###########################################################################

def create_entity_only_graph(full_graph: Graph) -> Tuple[Graph, Dict]:
    """Create a new graph with only entity-entity relations and extract attributes separately."""
    entity_graph = Graph()
    entity_graph.bind("atmun", EX)

    # Copy namespace bindings
    for prefix, namespace in full_graph.namespaces():
        entity_graph.bind(prefix, namespace)

    # Dictionary to store entity attributes
    entity_attributes = {}

    # Only include triples where object is not a literal (i.e., entity-entity relations)
    entity_count = 0
    attribute_count = 0

    for s, p, o in full_graph:
        subject_str = str(s)
        predicate_str = str(p)

        if not isinstance(o, Literal):
            # This is an entity-entity relation
            entity_graph.add((s, p, o))
            entity_count += 1
        else:
            # This is an entity-attribute relation
            if subject_str not in entity_attributes:
                entity_attributes[subject_str] = {}

            # Store the literal value with its datatype
            literal_value = str(o)
            datatype = str(o.datatype) if o.datatype else None

            entity_attributes[subject_str][predicate_str] = {
                'value': literal_value,
                'datatype': datatype
            }
            attribute_count += 1

    logger.info(f"Entity-only graph contains {entity_count} entity-entity triples")
    logger.info(f"Extracted {attribute_count} entity-attribute relations")

    return entity_graph, entity_attributes


def save_attributes_to_json(attributes: Dict, filename: str):
    """Save entity attributes to a JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(attributes, f, indent=2, ensure_ascii=False)
    logger.info(f"Entity attributes saved to {filename}")


def main():
    """Main function to load data, build the graph, and save it."""
    try:
        # Load all data sources
        adjacency_data, election_data, id2name, finance_rows = load_data()

        # Build the complete RDF graph
        g = build_graph(adjacency_data, election_data, id2name, finance_rows)

        # Create entity-only version and extract attributes
        entity_graph, entity_attributes = create_entity_only_graph(g)

        # Save files
        full_output = OUTPUT_FILE
        entity_output = OUTPUT_FILE.replace('.ttl', '_entities_only.ttl')
        attributes_output = OUTPUT_FILE.replace('.ttl', '_attributes.json')

        logger.info(f"Writing full graph to {full_output} in {RDF_FORMAT} format...")
        g.serialize(full_output, format=RDF_FORMAT)

        logger.info(f"Writing entity-only graph to {entity_output} in {RDF_FORMAT} format...")
        entity_graph.serialize(entity_output, format=RDF_FORMAT)

        save_attributes_to_json(entity_attributes, attributes_output)

        logger.info("Full graph statistics:")
        logger.info(f"Total triples: {len(g)}")
        subjects = set(s for s, p, o in g)
        logger.info(f"Total subjects: {len(subjects)}")
        logger.info(f"Total predicates: {len(set(p for s, p, o in g))}")
        objects = set(o for s, p, o in g if not isinstance(o, Literal))
        logger.info(f"Total objects (excluding literals): {len(objects)}")

        logger.info("Entity-only graph statistics:")
        logger.info(f"Total triples: {len(entity_graph)}")
        entity_subjects = set(s for s, p, o in entity_graph)
        logger.info(f"Total subjects: {len(entity_subjects)}")
        logger.info(f"Total predicates: {len(set(p for s, p, o in entity_graph))}")
        entity_objects = set(o for s, p, o in entity_graph)
        logger.info(f"Total objects: {len(entity_objects)}")

        logger.info("Attributes statistics:")
        logger.info(f"Entities with attributes: {len(entity_attributes)}")
        total_attrs = sum(len(attrs) for attrs in entity_attributes.values())
        logger.info(f"Total attribute-value pairs: {total_attrs}")

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    # Define EX namespace at global level for get_municipality_uri
    EX = Namespace("http://municipalities.austria.at/")
    main()