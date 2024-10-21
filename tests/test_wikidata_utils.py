import sys

import pytest

sys.path.insert(0, "./")
from wikidata_utils import (
    check_sparql_syntax,
    convert_if_date,
    get_property_examples,
    search_span,
)


@pytest.mark.asyncio(scope="session")
async def test_get_property_examples():
    expected_output = [
        ("India", "located in time zone", "UTC+05:30"),
        ("Baranagar", "located in time zone", "Indian Standard Time"),
        ("Montreal", "located in time zone", "Eastern Time Zone"),
        ("Germany", "located in time zone", "UTC+01:00"),
    ]

    output = get_property_examples("P421")

    assert output == expected_output, f"Expected {expected_output} but got {output}"


@pytest.mark.asyncio(scope="session")
async def test_check_sparql_syntax_valid():
    # A valid SPARQL query
    valid_query = """
    SELECT ?subject ?predicate ?object
    WHERE {
      ?subject ?predicate ?object.
    }
    """
    assert (
        check_sparql_syntax(valid_query) == True
    ), "Expected True for valid SPARQL query"


@pytest.mark.asyncio(scope="session")
async def test_check_sparql_syntax_invalid():
    # An invalid SPARQL query
    invalid_query = """
    SELECT ?subject ?predicate ?object
    WHERE {
      ?subject ?predicate ?object
    """
    assert (
        check_sparql_syntax(invalid_query) == False
    ), "Expected False for invalid SPARQL query"


@pytest.mark.asyncio(scope="session")
async def test_check_sparql_syntax_empty():
    # An empty SPARQL query
    empty_query = ""
    assert (
        check_sparql_syntax(empty_query) == False
    ), "Expected False for empty SPARQL query"


@pytest.mark.asyncio(scope="session")
async def test_convert_if_date():
    assert convert_if_date("From 0000-12-01T00:00:00Z") == "From 1 December"


@pytest.mark.asyncio(scope="session")
async def test_search_span():
    output = search_span(
        "birth date",
        limit=5,
        return_full_results=True,
        type="item",
    )

    assert len(output) == 5

    output = search_span(
        "birth date",
        limit=5,
        return_full_results=True,
        type="property",
    )

    assert len(output) > 0  # There is only one property that matches the search
