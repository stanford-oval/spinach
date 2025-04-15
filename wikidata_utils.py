from enum import Enum
import json
import logging
import os
import re
from collections import OrderedDict
from datetime import datetime
from functools import lru_cache
from urllib.parse import urlencode

import requests
import tenacity
from jinja2 import Environment, FileSystemLoader, select_autoescape
from rdflib.plugins.sparql.parser import parseQuery
from redis import StrictRedis
from redis_cache import RedisCache

# Cache Wikidata queries in Redis
client = StrictRedis(host="localhost", decode_responses=True)
cache = RedisCache(redis_client=client, prefix="wikidata")
# example of how to clean cache for a function: cached_requests.invalidate_all()

from chainlite import get_logger
import time

logger = get_logger(__name__)

inverse_property_path_regex = re.compile(
    "^(\?\w+)\s+wdt:P31\/wdt:P279\*\s+([^\s]+)\s*.$", re.IGNORECASE
)
count_regex = re.compile("COUNT\(\?\w+\)", re.IGNORECASE)


def try_to_optimize_query(query: str) -> str:
    # inverse property path
    matches = re.findall(inverse_property_path_regex, query)
    if len(matches) > 0:
        for m in matches:
            subst = f"{m[1]} ^wdt:P279*/^wdt:P31 {m[0]} ."
            query = re.sub(inverse_property_path_regex, subst, query, count=1)

    # count
    matches = re.findall(count_regex, query)
    if len(matches) > 0:
        for m in matches:
            subst = "COUNT(*)"
            query = re.sub(count_regex, subst, query, count=1)

    return query


current_script_directory = os.path.dirname(__file__)
jinja_environment = Environment(
    loader=FileSystemLoader(current_script_directory),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
    line_comment_prefix="#",
)


def fill_template(template_file, prompt_parameter_values={}):
    template = jinja_environment.get_template(template_file)

    filled_prompt = template.render(**prompt_parameter_values)
    filled_prompt = "\n".join(
        [line.strip() for line in filled_prompt.split("\n")]
    )  # remove whitespace at the beginning and end of each line
    return filled_prompt


wikidata_url = "https://www.wikidata.org/w/api.php"

class SPARQLResultsTooLargeError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = "Response too large, saving it in Redis will run into errors."
    
    def __str__(self):
        return f"SPARQLResultsTooLargeError: {self.message}"

class SPARQLSyntaxError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message
    
    def __str__(self):
        return f"SPARQLSyntaxError: {self.message}"
    
class SPARQLTimeoutError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message
    
    def __str__(self):
        return f"SPARQLTimeoutError: Query timed out"

def _serialize_error(exception):
    return json.dumps({
        "type": type(exception).__name__,
        "message": exception.message
    })
    
def _if_known_SPARQL_errors(exception_str):
    """
    Whether the (cached) input json object denotes one of:
    - SPARQLResultsTooLarge
    - SPARQL syntax error
        
    in which case directly raise these errors.
    """
    try:
        exc_info = json.loads(exception_str)
    except Exception:
        return
    
    if 'type' in exc_info and 'message' in exc_info:
        exc_type = exc_info['type']
        message = exc_info['message']

        if exc_type == "SPARQLResultsTooLargeError":
            raise SPARQLResultsTooLargeError("Response too large, saving it in Redis will run into errors.")
        
        if exc_type == "SPARQLSyntaxError":
            raise SPARQLSyntaxError(message)
        
        if exc_type == "SPARQLTimeoutError":
            raise SPARQLTimeoutError("timeout")

def _extract_wikidata_syntax_error(response: str):
    """
    Wikidata server returns a error str for a syntactic invalid SPARQL query directly,
    meaning that parsing it using response.json() could raise a JSON decoder error.
    
    In case of a JSON decoder error, go through the response text to determine if it is
    a syntactic compliant.
    """
    
    if "java.util.concurrent.ExecutionException" in response:
        index = response.find("java.util.concurrent.ExecutionException")
        index_end = response.find("at java.util.concurrent", index)
        return SPARQLSyntaxError(response[index: index_end])
    elif "java.util.concurrent.TimeoutException" in response:
        return SPARQLTimeoutError("timeout")
    return None
        

@cache.cache()
def _cached_requests(url, params=None) -> tuple[dict, int]:
    
    if params is not None:
        params = dict(params)
    
    r = requests.get(
        url,
        params=params,
        timeout=70,
        headers={"User-Agent": "Stanford OVAL, WikiSP project"},
    )

    if len(r.content) > 10 * 1024 * 1024:  # 10MB
        return _serialize_error(SPARQLResultsTooLargeError("Response too large, saving it in Redis will run into errors.")), 500

    try:
        res_json = r.json()
        return res_json, r.status_code
    except requests.exceptions.JSONDecodeError as e:
        syntax_error_object = _extract_wikidata_syntax_error(r.text)
        if syntax_error_object is not None:
            return _serialize_error(syntax_error_object), 500
        raise e
    except Exception as e:
        raise e


def cached_requests(url, params=None) -> tuple[dict, int]:
    """
    params: instead of a dict, it is a tuple of (k, v) so that it is hashable and we can cache it
    """
    res = _cached_requests(url, params)
    json_object, status_code = res
    
    # raise error if it is a known SPARQL error
    _if_known_SPARQL_errors(json_object)
    
    return json_object, status_code

countries = [
    "united states of america",
    "usa",
    "us" "united kingdom",
    "gb",
    "canada",
    "ca",
    "australia",
    "au",
    "new zealand",
    "nz",
    "south africa",
    "za",
    "germany",
    "de",
    "france",
    "fr",
    "brazil",
    "br",
    "argentina",
    "ar",
    "china",
    "cn",
    "india",
    "in",
    "russia",
    "ru",
    "japan",
    "jp",
    "south korea",
    "kr",
    "mexico",
    "mx",
    "indonesia",
    "id",
    "nigeria",
    "ng",
    "egypt",
    "eg",
    "saudi arabia",
    "sa",
    "turkey",
    "tr",
    "iran",
    "ir",
]


states = [
    # United States
    "alabama",
    "al",
    "alaska",
    "ak",
    "arizona",
    "az",
    "arkansas",
    "ar",
    "california",
    "ca",
    "colorado",
    "co",
    "connecticut",
    "ct",
    "delaware",
    "de",
    "florida",
    "fl",
    "georgia",
    "ga",
    "hawaii",
    "hi",
    "idaho",
    "id",
    "illinois",
    "il",
    "indiana",
    "in",
    "iowa",
    "ia",
    "kansas",
    "ks",
    "kentucky",
    "ky",
    "louisiana",
    "la",
    "maine",
    "me",
    "maryland",
    "md",
    "massachusetts",
    "ma",
    "michigan",
    "mi",
    "minnesota",
    "mn",
    "mississippi",
    "ms",
    "missouri",
    "mo",
    "montana",
    "mt",
    "nebraska",
    "ne",
    "nevada",
    "nv",
    "new hampshire",
    "nh",
    "new jersey",
    "nj",
    "new mexico",
    "nm",
    "new york",
    "ny",
    "north carolina",
    "nc",
    "north dakota",
    "nd",
    "ohio",
    "oh",
    "oklahoma",
    "ok",
    "oregon",
    "or",
    "pennsylvania",
    "pa",
    "rhode island",
    "ri",
    "south carolina",
    "sc",
    "south dakota",
    "sd",
    "tennessee",
    "tn",
    "texas",
    "tx",
    "utah",
    "ut",
    "vermont",
    "vt",
    "virginia",
    "va",
    "washington",
    "wa",
    "west virginia",
    "wv",
    "wisconsin",
    "wi",
    "wyoming",
    "wy",
    # Canada
    "alberta",
    "ab",
    "british columbia",
    "bc",
    "manitoba",
    "mb",
    "new brunswick",
    "nb",
    "newfoundland and labrador",
    "nl",
    "nova scotia",
    "ns",
    "ontario",
    "on",
    "prince edward island",
    "pe",
    "quebec",
    "qc",
    "saskatchewan",
    "sk",
    "northwest territories",
    "nt",
    "nunavut",
    "nu",
    "yukon",
    "yt",
]

states_and_abbreviations = {
    "al": "alabama",
    "ak": "alaska",
    "az": "arizona",
    "ar": "arkansas",
    "ca": "california",
    "co": "colorado",
    "ct": "connecticut",
    "de": "delaware",
    "fl": "florida",
    "ga": "georgia",
    "hi": "hawaii",
    "id": "idaho",
    "il": "illinois",
    "in": "indiana",
    "ia": "iowa",
    "ks": "kansas",
    "ky": "kentucky",
    "la": "louisiana",
    "me": "maine",
    "md": "maryland",
    "ma": "massachusetts",
    "mi": "michigan",
    "mn": "minnesota",
    "ms": "mississippi",
    "mo": "missouri",
    "mt": "montana",
    "ne": "nebraska",
    "nv": "nevada",
    "nh": "new hampshire",
    "nj": "new jersey",
    "nm": "new mexico",
    "ny": "new york",
    "nc": "north carolina",
    "nd": "north dakota",
    "oh": "ohio",
    "ok": "oklahoma",
    "or": "oregon",
    "pa": "pennsylvania",
    "ri": "rhode island",
    "sc": "south carolina",
    "sd": "south dakota",
    "tn": "tennessee",
    "tx": "texas",
    "ut": "utah",
    "vt": "vermont",
    "va": "virginia",
    "wa": "washington",
    "wv": "west virginia",
    "wi": "wisconsin",
    "wy": "wyoming",
}


def extract_id_from_uri(uri: str) -> str:
    if "wikidata.org" in uri:
        return uri[uri.rfind("/") + 1 :]
    else:
        # e.g. can be a statement like "statement/Q66072077-D5E883AA-1A7A-4F7C-A4B7-2723229A4385"
        return uri


def spans(mention):
    spans = []
    tokens = mention.split()
    for length in range(1, len(tokens) + 1):
        for index in range(len(tokens) - length + 1):
            span = " ".join(tokens[index : index + length])
            spans.append(span)
    return spans


def search_span(span: str, limit: int = 5, return_full_results=False, type="item"):
    """
    type should be one of the following values: form, item, lexeme, property, sense
    """
    candidates = []
    params = {
        "action": "wbsearchentities",
        "search": span,
        "language": "en",
        "limit": limit,
        "format": "json",
        # "props": "description",
        "type": type,
    }

    data, _ = cached_requests(wikidata_url, params=tuple(params.items()))

    if "search" not in data:
        with open("parser_error.log", "a") as fd:
            fd.write(f"{span} threw error for search_span")
        return []
    
    # Accessing the results
    results = data["search"]
    if return_full_results:
        return results

    # Print the title of each result
    for result in results:
        candidates.append(result["id"])

    return candidates


def location_search(mention):
    """
    In some instances, when NED misses a necessary entity, the semantic parser is trained to predict that in natural language
    This function tries to map those to their QIDs by searching Wikidata.
    It includes some heuristics to deal with locations like wd:phoenix_az
    """
    mention = mention.replace("_", " ")
    tokenized = mention.split()
    tokens = []
    for token in tokenized:
        if token in states_and_abbreviations:
            tokens.append(states_and_abbreviations[token])
        else:
            tokens.append(token)

    mention = " ".join(tokens)
    candidates = search_span(mention)
    if len(candidates) > 0:
        return candidates[0]

    for state in states:
        if state in spans(mention):
            index = mention.index(state)
            candidates = search_span(mention[: index - 1] + ", " + mention[index:])
            if len(candidates) > 0:
                return candidates[0]
            candidates = search_span(mention[: index - 1])
            if len(candidates) > 0:
                return candidates[0]

    for country in countries:
        if country in spans(mention):
            index = mention.index(country)
            candidates = search_span(mention[: index - 1] + ", " + mention[index:])
            if len(candidates) > 0:
                return candidates[0]
            candidates = search_span(mention[: index - 1])
            if len(candidates) > 0:
                return candidates[0]


def remove_whitespace_string(string):
    return "".join(string.split())


class SparqlExecutionStatus(str, Enum):
    OK = "ok"
    SYNTAX_ERROR = "syntax_error"
    TIMED_OUT = "timed_out"
    OTHER_ERROR = "other_error"
    TOO_LARGE = "too_large"

    @staticmethod
    def from_http_status_code(http_status_code: int):
        if http_status_code == 400:
            return SparqlExecutionStatus.SYNTAX_ERROR
        elif http_status_code > 400:
            return SparqlExecutionStatus.OTHER_ERROR
        else:
            return SparqlExecutionStatus.OK

    def __init__(self, default_message):
        self.default_message = default_message
        self.custom_message = None

    def set_message(self, msg):
        self.custom_message = msg

    def get_message(self):
        return self.custom_message if self.custom_message else self.default_message


@tenacity.retry(
    retry=tenacity.retry_if_exception_type(requests.exceptions.RequestException),
    wait=tenacity.wait_random_exponential(min=20, exp_base=3),
    stop=tenacity.stop_after_attempt(7),
    after=tenacity.after_log(logger, logging.INFO),
)
def execute_sparql(
    sparql: str, return_status: bool = False
) -> bool | list | tuple[bool | list, SparqlExecutionStatus]:
    """
    For syntactically incorrect SPARQLs returns None
    """
    url = "https://query.wikidata.org/sparql"
    try:
        r, status_code = cached_requests(
            url, params=tuple({"format": "json", "query": sparql}.items())
        )
        if status_code in {500, 400, 431, 413, 414}:
            # These are unrecoverable errors
            # 500: internal server error, 400: bad request (can happen when the SPARQL query is syntactically incorrect)
            # 431 for Request Header Fields Too Large
            # 413 for Content Too Large
            # 414 for URI Too Long for url
            if return_status:
                return [], SparqlExecutionStatus.from_http_status_code(status_code)
            else:
                return []
        if status_code >= 400:
            # 429, too many tries, would be included in this case.
            raise  # Reraise the exception so that we can retry using tenacity

        if "boolean" in r:
            res = r["boolean"]
        else:
            res = r["results"]["bindings"]
            if res == [] or (len(res) == 1 and res[0] == {}):
                res = []

    except requests.exceptions.ReadTimeout:
        if return_status:
            return [], SparqlExecutionStatus.TIMED_OUT
        else:
            return []
    # except requests.exceptions.JSONDecodeError or json.decoder.JSONDecodeError:
    #     if return_status:
    #         return [], SparqlExecutionStatus.TIMED_OUT  # TODO it this always the case?
    #     else:
    #         return []
    except requests.exceptions.ConnectionError:
        if return_status:
            return [], SparqlExecutionStatus.OTHER_ERROR
        else:
            return []
    except SPARQLResultsTooLargeError:
        if return_status:
            return [], SparqlExecutionStatus.TOO_LARGE
        else:
            return []
    except SPARQLTimeoutError:
        if return_status:
            return [], SparqlExecutionStatus.TIMED_OUT
        else:
            return []
    except SPARQLSyntaxError as e:
        if return_status:
            status = SparqlExecutionStatus.SYNTAX_ERROR
            status.set_message(e.message)
            return [], status
        else:
            return []
    # except Exception as e:
    #     return res
    #     logger.exception(e)

    if return_status:
        return res, SparqlExecutionStatus.OK
    else:
        return res

def nl2sparql_execute_sparql_timeout(query):
    from pymongo import MongoClient
    mongo_client = MongoClient()
    sparql_results = mongo_client["sparql_results"]["sparql_results"]
    url = "https://query.wikidata.org/sparql"
    try:
        final_res = True
        if sparql_results.find_one({"sparql": query}):
            res = sparql_results.find_one({"sparql": query})["results"]
        else:
            r = requests.post(
                url,
                params={"format": "json", "query": query},
                timeout=70,
                headers={"User-Agent": "Stanford CS OVAL, Wikidata project"},
            )
            r.raise_for_status()

            if "boolean" in r.json():
                res = r.json()["boolean"]
            else:
                res = r.json()["results"]["bindings"]
        if res == []:
            final_res = False
        
        try:
            if not sparql_results.find_one({"sparql": query}):
                sparql_results.insert_one({"sparql": query, "results": res})
        except Exception:
            pass
        
        return final_res
    except requests.exceptions.HTTPError as err:
        # 431 for Request Header Fields Too Large
        # 414 for URI Too Long for url
        # 504 for Server Error: Gateway Timeout for url
        # if r.status_code == 500 or r.status_code == 400 or r.status_code == 431 or r.status_code == 414:
        if r.status_code == 429:
            time.sleep(60)
            res = nl2sparql_execute_sparql_timeout(query)
        else:
            res = str(err)
            # raise  # Reraise the exception if it's not a 500 error
    except requests.exceptions.ReadTimeout as err:
        res = str(err)
    except requests.exceptions.JSONDecodeError or json.decoder.JSONDecodeError as err:
        res = str(err)
    except requests.exceptions.ConnectionError as err:
        res = str(err)
    except KeyError as err:
        res = str(err)
    except Exception as err:
        res = str(err)
        
    return res



def execute_predicted_sparql(sparql):
    # first, let's replace the properties

    sparql = sparql.replace("wdt:instance_of/wdt:subclass_of", "wdt:P31/wdt:P279")

    url = "https://query.wikidata.org/sparql"
    extracted_property_names = [
        x[1]
        for x in re.findall(r"(wdt:|p:|ps:|pq:)([a-zA-Z_\(\)(\/_)]+)(?![1-9])", sparql)
    ]
    pid_replacements = {}
    for replaced_property_name in extracted_property_names:

        i = replaced_property_name.replace("_", " ").lower()
        pid_query = (
            """
            SELECT ?property ?propertyLabel WHERE {
            ?property rdf:type wikibase:Property .
            ?property rdfs:label "%s"@en .
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }"""
            % i
        )

        data = execute_sparql(pid_query)
        if (
            "results" in data
            and "bindings" in data["results"]
            and len(data["results"]["bindings"]) > 0
        ):
            # Extract the property ID from the response
            property_id = data["results"]["bindings"][0]["property"]["value"]
            property_id = property_id.replace("http://www.wikidata.org/entity/", "")
        else:
            # try querying f"{wikidata_url}?action=wbsearchentities&search=songwriter&language=en&limit=20&format=json&type=property"
            params = {
                "action": "wbsearchentities",
                "search": i,
                "language": "en",
                "limit": 20,
                "format": "json",
                "type": "property",
            }
            encoded_url = wikidata_url + "?" + urlencode(params)
            response = cached_requests(url=encoded_url)
            data = response[0]

            if "search" in data and len(data["search"]) > 0:
                property_id = data["search"][0]["id"]
            else:
                # logger.warning("CANNOT FIND PROPERTY: {} for SPARQL {}".format(replaced_property_name, sparql))
                return [], sparql

        pid_replacements[replaced_property_name] = property_id

    def sub_fcn(match):
        prefix = match.group(1)
        value = match.group(2)

        return prefix + pid_replacements[value]

    sparql = re.sub(
        r"(wdt:|p:|ps:|pq:)([a-zA-Z_\(\)(\/_)]+)(?![1-9])",
        lambda match: sub_fcn(match),
        sparql,
    )

    # next, we need to replace the entities
    extracted_entity_names = [
        x[1] for x in re.findall(r"(wd:)([a-zA-PR-Z_0-9-]+)", sparql)
    ]
    qid_replacements = {}
    for extracted_entity_name in extracted_entity_names:
        try_location = location_search(extracted_entity_name.replace("_", " "))
        if try_location is not None:
            try_location = "wd:" + try_location
            # print("inserting {} for {}".format(try_location, extracted_entity_name))
            qid_replacements[extracted_entity_name] = try_location
        else:
            # print("CANNOT FIND ENTITY: {} for SPARQL {}".format(extracted_entity_name, sparql))
            return [], sparql

    def sub_entity_fcn(match):
        value = match.group(2)
        return qid_replacements[value]

    sparql = re.sub(
        r"(wd:)([a-zA-PR-Z_0-9-]+)", lambda match: sub_entity_fcn(match), sparql
    )

    # finally, we can execute
    prediction_results = execute_sparql(sparql)
    return prediction_results, sparql


@lru_cache()
def get_name_from_qid(qid):
    if "wd:" in qid:
        qid = qid.replace("wd:", "")
    
    query = f"""
    SELECT ?label
    WHERE {{
    wd:{qid} rdfs:label ?label.
    FILTER(LANG(?label) = "en").
    }}
    """
    r = execute_sparql(query)

    if r == [] or r is None:
        return None

    name = r[0]["label"]["value"]

    return name


def format_date(match):
    date_string = match.group(0)
    if date_string == "0000-00-00T00:00:00Z":
        return "0"
    if date_string.startswith("0000-"):
        # year zero is not allowed in Wikidata dates, but we still encounter it rarely
        return datetime.strptime(date_string[5:], "%m-%dT%H:%M:%SZ").strftime("%-d %B")
    return datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%SZ").strftime("%-d %B %Y")



def convert_if_date(x: str) -> str:
    if x is None:
        return x
    if type(x) is bool:
        return x
    return re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", format_date, x)


def convert_each_result(result: str):
    # print("result = ", end=None)
    # pprint(result)
    if "x" in result:
        result = result["x"]
    elif "count" in result:
        result = result["count"]
    elif "number" in result:
        result = result["number"]
    else:
        # logger.warning("Did not find 'x' or 'count' in %s", str(i))
        found_good_key = False
        candidates = []
        for key in result:
            if "type" in result[key] and result[key]["type"] == "literal":
                candidates.append(result[key])
        if candidates:
            result = min(candidates, key=lambda x: len(x["value"]))
        found_good_key = len(candidates) > 0

        if not found_good_key:
            for key in result:
                if "type" in result[key] and result[key]["type"] == "uri":
                    found_good_key = True
                    result = result[key]
                    break
        if not found_good_key:
            logger.error(
                "Could not convert the result %s", str(result)
            )  # just to make sure we are not missing anything
            return ""

    if result["type"] == "uri":
        # extract QID part from results
        qid = extract_id_from_uri(result["value"])
        name = get_name_from_qid(qid)
        if name:
            return name
        # then it was most likely a url like https://de.wikipedia.org/wiki/Gambling,_Gods_and_LSD
        return result["value"]

    elif result["type"] == "literal":
        return result["value"]
    else:
        raise ValueError()


def get_actual_names_from_results(executed_result):
    # convert QIDs to their actual names
    if isinstance(executed_result, bool):
        return [executed_result]

    # if there are too many results, it would not pass anyways, so just choose the top 20
    # executed_result = executed_result[:20]

    ret = [convert_if_date(convert_each_result(r)) for r in executed_result]
    ret = [
        r for r in ret if r is not None
    ]  # some entities like Q24941172 don't have a human-readable label in Wikidata, ignore them
    return ret


def check_sparql_syntax(query):
    try:
        # Attempt to parse the SPARQL query
        parse_result = parseQuery(query)
        # If parsing succeeds, the query is syntactically correct
        return True
    except Exception as e:
        # If parsing fails, the query is syntactically incorrect
        return False


def get_property_examples(pid: str):
    """Get a list of examples (according to property P1855)
    for a given property

    Args:
        pid (str): e.g. "P155" or "wd:P155" or "wdt:155"

    Returns:
        a list of tuples, [(subject, pid_label, object)], where in WikiData
        there exists a tuple of the form:
        (subject, pid, object)
        and subject and object are already in their labels

        e.g., for P155, we will have:
        [..., ('April', 'follows', 'March'), ...]
        meaning that April follows March
    """

    if pid.startswith("wd:"):
        pid = pid.replace("wd:", "")
    if pid.startswith("wdt:"):
        pid = pid.replace("wdt:", "")

    sparql_query = f"""
SELECT DISTINCT ?subLabel ?objLabel WHERE {{
  wd:{pid} p:P1855 ?v.
  ?v ps:P1855 ?sub.
  ?v pq:{pid} ?obj.
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" . }}
}}
"""
    res = execute_sparql(sparql_query)
    try:
        res = [
            (
                i["subLabel"]["value"],
                get_property_or_entity_description(pid)["label"],
                i["objLabel"]["value"],
            )
            for i in res
        ]
    except Exception as e:
        logger.warning(
            "Property %s throw exception %s in get_property_examples()", pid, str(e)
        )
    if not res:
        logger.warning("Property %s has no examples", pid)
    return res


def get_property_or_entity_description(p_or_qid: str):
    """Get label and description for a given property or entity

    Args:
        p_or_qid (str): e.g. "P155" or "wd:P155" or "wdt:P155"

    Returns:
        {
            "label": property label in English,
            "description": property description in English
        }
    """

    if p_or_qid.startswith("wd:"):
        p_or_qid = p_or_qid.replace("wd:", "")
    if p_or_qid.startswith("wdt:"):
        p_or_qid = p_or_qid.replace("wdt:", "")

    sparql_query = f"""
SELECT ?propertyLabel ?propertyDesc
WHERE {{
  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "en" .
    wd:{p_or_qid} rdfs:label ?propertyLabel .
    wd:{p_or_qid} schema:description ?propertyDesc .
  }}
}}
"""
    res = execute_sparql(sparql_query)
    
    # not "propertyDesc" in res[0] would denote no description available
    if not res or not "propertyDesc" in res[0]:
        return None
    return {
        "label": res[0]["propertyLabel"]["value"],
        "description": res[0]["propertyDesc"]["value"],
    }


@lru_cache()
def get_outgoing_edges(qid: str, compact: bool):
    """
    QID example: Q679545
    compact: if True, will exclude the "Description" field from the output
    """

    property_sparql = f"""
    SELECT DISTINCT ?p ?v ?p2Label ?vLabel ?vDescription WHERE {{
    wd:{qid} ?p ?v.
    FILTER(STRSTARTS(STR(?p), str(wdt:))) . 
    BIND (IRI(replace(str(?p), str(wdt:), str(wd:)))  AS ?p2)
    ?p2 wikibase:propertyType ?type .
    FILTER (?type != wikibase:ExternalId) .
    OPTIONAL {{
        ?v schema:description ?vDescription .
        FILTER(LANG(?vDescription) = "en")
    }}
    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" . }}
    }}
    """

    qualifier_sparql = f"""
    SELECT DISTINCT ?p ?stmt ?p2Label ?qualifier ?q2Label ?d ?dLabel ?dDescription ?finalLabel ?final WHERE {{
    wd:{qid} ?p ?stmt.
    ?stmt ?qualifier ?final.
    FILTER(STRSTARTS(STR(?p), str(p:))) .
    FILTER(STRSTARTS(STR(?qualifier), str(pq:))) .
    BIND (IRI(replace(str(?p), str(p:), str(wd:)))  AS ?p2)
    BIND (IRI(replace(str(?p), str(p:), str(ps:)))  AS ?p3)
    BIND (IRI(replace(str(?qualifier), str(pq:), str(wd:))) AS ?q2)
    ?stmt ?p3 ?d .
    OPTIONAL {{
        ?d schema:description ?dDescription .
        FILTER(LANG(?vDescription) = "en")
    }}
    ?p2 wikibase:propertyType ?type .
    FILTER (?type != wikibase:ExternalId) .
    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" . }}
    }}
    """

    property_values: list = execute_sparql(property_sparql)
    qualifiers: list = execute_sparql(qualifier_sparql)

    all_properties = {}
    if qualifiers is not None:
        for q in qualifiers:
            pid = extract_id_from_uri(q["p"]["value"])
            stmt = q["stmt"]["value"]
            p_name = q["p2Label"]["value"]
            p_value_id = extract_id_from_uri(q["d"]["value"])
            p_value_label = convert_if_date(q["dLabel"]["value"])
            p_value_description = (
                q["dDescription"]["value"] if "dDescription" in q else None
            )
            qual_id = extract_id_from_uri(q["qualifier"]["value"])
            if qual_id.startswith("pqv:"):
                # skip those with pqv since they seem to get repeated with pq:
                continue
            qual_label = q["q2Label"]["value"]
            qual_value = convert_if_date(q["finalLabel"]["value"])
            if q["final"]["type"] == "uri":
                # Add the qualifier value's QID in parenthesis
                qual_value_id = extract_id_from_uri(q["final"]["value"])
                qual_value = f"{qual_value} ({qual_value_id})"

            property_key = f"{p_name} ({pid})"
            if p_value_id.startswith("Q"):
                property_value_key = f"{p_value_label} ({p_value_id})"
            else:
                property_value_key = f"{p_value_label}"
            if property_key not in all_properties:
                all_properties[property_key] = {}

            # if this property found before, update the "Qualifiers" dict:
            if (
                property_value_key in all_properties[property_key]
                and "Qualifiers" in all_properties[property_key][property_value_key]
                and stmt
                in all_properties[property_key][property_value_key]["Qualifiers"]
            ):
                all_properties[property_key][property_value_key]["Qualifiers"][
                    stmt
                ].update({f"{qual_label} ({qual_id})": f"{qual_value}"})
            # this means we are just missing this block
            elif (
                property_value_key in all_properties[property_key]
                and "Qualifiers" in all_properties[property_key][property_value_key]
            ):
                all_properties[property_key][property_value_key]["Qualifiers"][stmt] = {
                    f"{qual_label} ({qual_id})": f"{qual_value}"
                }
            # if neither found then we have to create one
            else:
                all_properties[property_key][property_value_key] = {}
                if p_value_description:
                    all_properties[property_key][property_value_key].update(
                        {
                            "Description": p_value_description,
                        }
                    )
                all_properties[property_key][property_value_key].update(
                    {
                        "Qualifiers": {
                            stmt: {f"{qual_label} ({qual_id})": f"{qual_value}"}
                        }
                    }
                )

    # delete the stmt keys
    for i in all_properties:
        for j in all_properties[i]:
            all_properties[i][j]["Qualifiers"] = list(
                all_properties[i][j]["Qualifiers"].values()
            )

    if property_values is not None:
        for pv in property_values:
            pid = extract_id_from_uri(pv["p"]["value"])
            p_name = pv["p2Label"]["value"]
            p_value_label = convert_if_date(pv["vLabel"]["value"])
            p_value_id = extract_id_from_uri(pv["v"]["value"])
            key = f"{p_name} ({pid})"
            if key not in all_properties:
                all_properties[key] = {}
            if p_value_id.startswith("Q"):
                key2 = f"{p_value_label} ({p_value_id})"
            else:
                key2 = f"{p_value_label}"
            if key2 in all_properties[key]:
                # this indicates that it was already added by the qualifiers above
                continue
            all_properties[key][key2] = {}
            if "vDescription" in pv:
                all_properties[key][key2]["Description"] = pv["vDescription"]["value"]

    # sort properties by their PID, because smaller PIDs are probably more common and useful, and bigger ones are generally more obscure
    all_properties = OrderedDict(
        sorted(
            all_properties.items(),
            key=lambda x: int(re.search(r"\(P(\d+)\)", x[0]).group(1)),
        )
    )

    # sort qualifiers that have "ordinal"
    def ordinal_sort_fn(pv: dict):
        # Example of pv:
        # "Hussein (Q27101961)": {
        #       "Description": "male given name (حسين)",
        #       "Qualifiers": [
        #           {
        #               "series ordinal (P1545)": "2"
        #           }
        #       ]
        #  }
        p, v = pv
        if "Qualifiers" in v:
            for q in v["Qualifiers"]:
                if "series ordinal (P1545)" in q:
                    # TODO we can convert strings like "second one" to a number too
                    try:
                        ret = int(q["series ordinal (P1545)"])
                    except ValueError:
                        ret = 0

                    return ret

        return -1

    # TODO if compact == True, and if series ordinal is the only qualifier in all values, we can convert the whole thing into a list
    for p in all_properties:
        all_properties[p] = OrderedDict(
            sorted(all_properties[p].items(), key=ordinal_sort_fn)
        )

    if compact:
        for p in all_properties:
            should_compact = True
            for v in all_properties[p]:
                if "Description" in all_properties[p][v]:
                    del all_properties[p][v]["Description"]
                if len(all_properties[p][v]) > 0:
                    should_compact = False
            if should_compact:
                all_properties[p] = [v for v in all_properties[p]]
            if len(all_properties[p]) == 1:
                if isinstance(all_properties[p], list):
                    all_properties[p] = all_properties[p][0]

    return all_properties


def get_p_or_q_id_from_name(
    name: str, 
    type="qid", 
    limit=3,
    return_list=False, # if enabled, return all results instead of just one
    return_with_label=False, # if enabled, each entry is a tuple with (q_or_p_id, label)
    no_off_topic=False # if enabled, do not return qid if requested pid, vise versa
) -> str:
    query = f"""
    SELECT ?entity ?entityLabel WHERE {{
    ?entity rdfs:label "{name}"@en.
    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    LIMIT {limit}
    """
    r = execute_sparql(query)

    if r == [] or r is None:
        if return_list:
            return []
        else:
            return None
    
    res = []
    for i in range(len(r)):
        p_q_id = extract_id_from_uri(r[i]["entity"]["value"])
        if type == "qid" and p_q_id.startswith("Q"):
            if return_list:
                res.append((p_q_id, r[i]["entityLabel"]["value"]) if return_with_label else p_q_id)
            else:
                return p_q_id
        elif type == "pid" and p_q_id.startswith("P"):
            if return_list:
                res.append((p_q_id, r[i]["entityLabel"]["value"]) if return_with_label else p_q_id)
            else:
                return p_q_id

    if no_off_topic:
        return res if return_list else None

    # Now return something even if it is QID and we wanted PID or vice versa
    for i in range(len(r)):
        p_q_id = extract_id_from_uri(r[i]["entity"]["value"])
        if p_q_id.startswith(("Q", "P")):
            if return_list:
                res.append((p_q_id, r[i]["entityLabel"]["value"]) if return_with_label else p_q_id)
            else:
                return p_q_id

    if not return_list:
        logger.warning('Could not find a %s for "%s" in Wikidata', type, name)
    return res if return_list else None


def normalize_result_string(result_string: str) -> str:
    """
    Helps when calculating EM or Superset metrics
    Sorts multiple answers alphabetically so that it is always consistent. Lowercases everything.
    """
    # Sometimes there are duplicates in the gold string. Remove them:
    results = list(set(result_string.split(";")))
    results = [r.strip() for r in results]
    return "; ".join(s.strip() for s in sorted(results)).lower()

if __name__ == "__main__":
    entities = "Santa Claus"
    
    print(search_span(entities))
    
    sparql="""
SELECT ?item ?itemLabel ?inventoryNumber ?creator ?creatorLabel ?birthPlace ?birthPlaceLabel WHERE {                                                                   
  ?item wdt:P195 wd:Q3329624.                                                                                           
  OPTIONAL { ?item wdt:P170 ?creator. }                                                                                 
  OPTIONAL { ?creator wdt:P19 ?birthPlace.                                                                              
             ?birthPlace wdt:P131* wd:Q12130. }                                                                         
  OPTIONAL { ?item wdt:P217 ?inventoryNumber. }                                                                         
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }                                    
}
    """
    print(execute_sparql(sparql, return_status=True))