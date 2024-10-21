import json
import os
import traceback

import execjs
from chainlite import get_logger

logger = get_logger(__name__)

# taken from https://www.wikidata.org/wiki/EntitySchema:E49
PREFIXES = """
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX cc: <http://creativecommons.org/ns#>
PREFIX dct: <http://purl.org/dc/terms/>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX ontolex: <http://www.w3.org/ns/lemon/ontolex#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
PREFIX pqn: <http://www.wikidata.org/prop/qualifier/value-normalized/>
PREFIX pqv: <http://www.wikidata.org/prop/qualifier/value/>
PREFIX pr: <http://www.wikidata.org/prop/reference/>
PREFIX prn: <http://www.wikidata.org/prop/reference/value-normalized/>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX prv: <http://www.wikidata.org/prop/reference/value/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX psn: <http://www.wikidata.org/prop/statement/value-normalized/>
PREFIX psv: <http://www.wikidata.org/prop/statement/value/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdata: <http://www.wikidata.org/wiki/Special:EntityData/>
PREFIX wdno: <http://www.wikidata.org/prop/novalue/>
PREFIX wdref: <http://www.wikidata.org/reference/>
PREFIX wds: <http://www.wikidata.org/entity/statement/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wdtn: <http://www.wikidata.org/prop/direct-normalized/>
PREFIX wdv: <http://www.wikidata.org/value/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""

js_sparql_to_json = """
function parse_sparql(sparql) {{
var SparqlParser = require('sparqljs').Parser;
var parser = new SparqlParser();
var parsedQuery = parser.parse( sparql);
return parsedQuery
  }}
var result = parse_sparql(`{0}`) 
  """

json_to_sparql = """ function parse_json(json) {{
var SparqlGenerator = require('sparqljs').Generator;
var generator = new SparqlGenerator({{ }});
var generatedQuery = generator.stringify(json);
return generatedQuery
}}
var obj = JSON.parse(JSON.stringify({0}));
var result = parse_json(obj)
"""


def get_json(sparql):
    script_directory = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_directory)
    if PREFIXES not in sparql:
        sparql = PREFIXES + sparql
    exec = js_sparql_to_json.format(sparql)
    ctx = execjs.compile(exec)
    try:
        result = ctx.eval("result")
    except Exception as e:
        logger.debug(str(e))
        return []
    return result


def get_sparql(json_ql):
    exec = json_to_sparql.format(json.dumps(json_ql))
    ctx = execjs.compile(exec)
    result = ctx.eval("result")
    return result


def prettify_sparql(sparql: str) -> str:
    try:
        ret = get_sparql(get_json(sparql))
    except Exception as e:
        logger.debug("Ran into error '%s' when prettifying sparql %s", str(e), sparql)
        return sparql
    lines = []
    for line in ret.split("\n"):
        if not line.startswith("PREFIX"):
            lines.append(line)
    return "\n".join(lines)


def count_triples(query_str, threashold=3):
    _json = get_json(query_str)
    if not _json:
        return False

    try:
        if len(_json["where"]) <= threashold:
            return True
        else:
            return False
    except Exception:
        return False
