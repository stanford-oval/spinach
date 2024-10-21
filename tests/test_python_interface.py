import pytest

from parser_python_interface import (count_triples, get_json, get_sparql,
                                     prettify_sparql)
from wikidata_utils import remove_whitespace_string


@pytest.mark.asyncio(scope="session")
async def test_count_triples():
    sparql = """SELECT ?item ?d
    {
        ?item wdt:P39 wd:Q208233 . 
        MINUS { ?item wdt:P6902 [] } 
        OPTIONAL { ?item wdt:P570 ?d  }
    }
    ORDER BY DESC(?d) 
    LIMIT 50
    """

    # Count triples in the query
    assert count_triples(sparql) == True


@pytest.mark.asyncio(scope="session")
async def test_convert_to_json_and_back():
    sparql = "SELECT DISTINCT ?x WHERE { wd:Q392396 p:P725 ?p. ?p ps:P725 ?x; pq:P453 wd:Q28704702. }"
    r = get_json(sparql)

    # expected result
    expected = [
        {
            "subject": {
                "termType": "NamedNode",
                "value": "http://www.wikidata.org/entity/Q392396",
            },
            "predicate": {
                "termType": "NamedNode",
                "value": "http://www.wikidata.org/prop/P725",
            },
            "object": {"termType": "Variable", "value": "p"},
        },
        {
            "subject": {"termType": "Variable", "value": "p"},
            "predicate": {
                "termType": "NamedNode",
                "value": "http://www.wikidata.org/prop/statement/P725",
            },
            "object": {"termType": "Variable", "value": "x"},
        },
        {
            "subject": {"termType": "Variable", "value": "p"},
            "predicate": {
                "termType": "NamedNode",
                "value": "http://www.wikidata.org/prop/qualifier/P453",
            },
            "object": {
                "termType": "NamedNode",
                "value": "http://www.wikidata.org/entity/Q28704702",
            },
        },
    ]

    assert r["where"][0]["triples"] == expected, f"Expected {expected}, but got {r}"

    assert remove_whitespace_string(get_sparql(r)) == remove_whitespace_string(
        """
        PREFIX p: <http://www.wikidata.org/prop/>
        PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
        PREFIX ps: <http://www.wikidata.org/prop/statement/>
        PREFIX wd: <http://www.wikidata.org/entity/>
        SELECT DISTINCT ?x WHERE {
        wd:Q392396 p:P725 ?p.
        ?p ps:P725 ?x;
            pq:P453 wd:Q28704702.
        }
        """
    )


@pytest.mark.asyncio(scope="session")
async def test_prettify_sparql():
    # Define SPARQL queries (PREFIX included and PREFIX not included cases)
    sparql = 'SELECT ?item ?itemLabel ?title ?itWikipediaArticle WHERE { ?category wdt:P31 wd:Q4167836; rdfs:label "Compositori per nazionalità"@it. ?item wdt:P31 wd:Q5; wdt:P910 ?category. FILTER NOT EXISTS { ?item wdt:P106 ?occupation. } OPTIONAL { ?item wdt:P1476 ?title. } OPTIONAL { ?itWikipediaArticle schema:about ?item; schema:isPartOf <https://it.wikipedia.org/>. } SERVICE wikibase:label { bd:serviceParam wikibase:language "it,en". } }'
    expected = """
        SELECT ?item ?itemLabel ?title ?itWikipediaArticle WHERE {
    ?category wdt:P31 wd:Q4167836;
        rdfs:label "Compositori per nazionalità"@it.
    ?item wdt:P31 wd:Q5;
        wdt:P910 ?category.
    FILTER(NOT EXISTS { ?item wdt:P106 ?occupation. })
    OPTIONAL { ?item wdt:P1476 ?title. }
    OPTIONAL {
        ?itWikipediaArticle schema:about ?item;
        schema:isPartOf <https://it.wikipedia.org/>.
    }
    SERVICE wikibase:label { bd:serviceParam wikibase:language "it,en". }
    }
    """
    assert remove_whitespace_string(
        prettify_sparql(sparql)
    ) == remove_whitespace_string(expected)

    # sparql = """
    # SELECT *
    # {
    # VALUES ?frwiki {
    # <https://fr.wikipedia.org/wiki/Andr%C3%A9_Ch%C3%A9radame>
    # <https://fr.wikipedia.org/wiki/Andr%C3%A9_Ch%C3%A9radame>
    # }
    # ?frwiki schema:about ?item ; schema:name ?name 
    # }
    # """
    # print(get_json(sparql))
    # print(get_sparql(get_json(sparql))) # this causes an error