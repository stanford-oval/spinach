import json
import operator
import re
from typing import Annotated, Optional, Sequence, TypedDict, List

import pandas as pd
import chainlit as cl

from parser_python_interface import prettify_sparql
from wikidata_utils import (
    convert_if_date,
    execute_sparql,
    extract_id_from_uri,
    get_actual_names_from_results,
    get_name_from_qid,
    normalize_result_string,
    try_to_optimize_query,
)

def _truncate_json(data: dict):
    """
        Helper function used by `json_to_panda_string` and `json_to_panda_markdown`
    """
    formatted_data = []
    for d in data:
        new_d = {}
        for k, v in d.items():
            new_d[k] = extract_id_from_uri(v["value"])
        formatted_data.append(new_d)

    df = pd.DataFrame(formatted_data).fillna("-")
    return df

def json_to_panda_string(data: dict) -> str:
    if type(data) == bool:
        return str(data)

    df = _truncate_json(data)
    return df.to_string(index=False, justify="left", max_rows=10)


def json_to_panda_markdown(data: dict, head = 10) -> str:
    if type(data) == bool:
        return str(data)

    df = _truncate_json(data)
    
    # Determine the number of rows to show at the top and bottom
    num_head = head // 2
    num_tail = head - num_head
    
    # Create the truncated DataFrame
    if len(df) > head:
        head_df = df.head(num_head)
        tail_df = df.tail(num_tail)
        
        # Number of omitted rows
        omitted_rows = len(df) - head
        
        # Create a custom row with '... (x omitted) | ... | ...'
        custom_row = pd.DataFrame({col: ['...'] for col in df.columns})
        custom_row[df.columns[0]] = f'... ({omitted_rows} omitted)'
        
        # Concatenate the head, custom row, and tail DataFrames
        truncated_df = pd.concat([head_df, custom_row, tail_df], ignore_index=True)
    else:
        truncated_df = df

    return truncated_df.to_markdown(index=False)


class SparqlQuery:

    def __init__(self, sparql: Optional[str] = None):
        if "PXXX" in sparql or "PYYY" in sparql or "PZZZ" in sparql:
            self.is_valid = False
        self.sparql = SparqlQuery.clean_sparql(sparql)
        self.preverbalized, self.label_to_id_map = SparqlQuery.preverbalize(self.sparql)
        for k, v in self.label_to_id_map.items():
            if not v:
                self.is_valid = False
                return
        self.is_valid = True
        
        self.execution_result = None
        
    @staticmethod
    def clean_sparql(sparql: str):
        if sparql is None:
            return sparql
        cleaned_sparql = sparql.strip()

        cleaned_sparql = re.sub(r"#.*", "", cleaned_sparql).strip()  # Remove comments
        cleaned_sparql = try_to_optimize_query(cleaned_sparql)
        # cleaned_sparql = re.sub(
        #     r"\s+", " ", cleaned_sparql
        # )  # Remove line breaks and other extra whitespaces
        cleaned_sparql = prettify_sparql(cleaned_sparql)

        return cleaned_sparql

    def execute(self):
        self.execution_result, self.execution_status = execute_sparql(
            self.sparql, return_status=True
        )

    def has_results(self) -> bool:
        return (type(self.execution_result) == bool) or bool(self.execution_result)

    def results_in_table_format(self) -> str:
        return json_to_panda_string(self.execution_result)

    def results_in_markdown_format(self) -> str:
        return json_to_panda_markdown(self.execution_result)

    def get_execution_result_set(self) -> set:
        """
        Used for evaluation
        """

        if self.execution_result is None:
            return set()

        if isinstance(self.execution_result, dict):
            raise ValueError(
                "Found dictionary result: %s in %s"
                % (str(self.execution_result), str(self))
            )

        execution_result = get_actual_names_from_results(self.execution_result)

        if type(execution_result) == bool:
            return set(["yes"]) if execution_result else set(["no"])

        results = set()
        for x in execution_result:
            if type(x) == dict:
                if x["name"]:
                    results.add(x["name"])
            elif x != None:
                x = convert_if_date(x)
                results.add(x)

        return results

    def get_normalized_execution_result(self):
        result_set = self.get_execution_result_set()
        if not result_set:
            return None

        return normalize_result_string("; ".join(result_set))

    @staticmethod
    def preverbalize(sparql) -> str:
        """
        Example sparql: ASK WHERE { wd:Q173144 wdt:P26 ?x. ?x wdt:P102 wd:Q18233. }
        """
        qid_list = set(
            [x[1] for x in re.findall(r"(wdt:|p:|ps:|pq:|wd:)(Q[0-9]+)", sparql)]
        )
        pid_list = set(
            [x[1] for x in re.findall(r"(wdt:|p:|ps:|pq:|wd:)(P[0-9]+)", sparql)]
        )

        qid_list_tuples = [(i, get_name_from_qid(i)) for i in qid_list]
        pid_list_tuples = [(i, get_name_from_qid(i)) for i in pid_list]
        all_ids = pid_list_tuples + qid_list_tuples
        label_to_id_map = {}
        for id_, label in all_ids:
            label_to_id_map[label] = id_

        # Replace starting from the longest one, so that we don't mistakenly replace P12 instead of P1
        for _id, name in sorted(all_ids, key=lambda x: len(x[0]), reverse=True):
            sparql = sparql.replace(_id, f"{_id}[{name}]")

        return sparql, label_to_id_map

    def __repr__(self):
        return f"SPARQL({self.sparql}, preverbalized={self.preverbalized})"

    def __hash__(self):
        return hash(self.sparql)


def merge_dictionaries(dictionary_1: dict, dictionary_2: dict) -> dict:
    """
    Merges two dictionaries, combining their key-value pairs.
    If a key exists in both dictionaries, the value from dictionary_2 will overwrite the value from dictionary_1.

    Parameters:
        dictionary_1 (dict): The first dictionary.
        dictionary_2 (dict): The second dictionary.

    Returns:
        dict: A new dictionary containing the merged key-value pairs.
    """
    merged_dict = dictionary_1.copy()  # Start with a copy of the first dictionary
    merged_dict.update(
        dictionary_2
    )  # Update with the second dictionary, overwriting any duplicates
    return merged_dict


def merge_sets(set_1: set, set_2: set) -> set:
    return set_1 | set_2


def add_item_to_list(_list: list, item) -> list:
    ret = _list.copy()
    # if item not in ret:
    ret.append(item)
    return ret


class BaseParserState(TypedDict):
    question: str
    conversation_history: List
    engine: str
    response: str
    generated_sparqls: Annotated[list[SparqlQuery], add_item_to_list]
    final_sparql: SparqlQuery
    action_counter: Annotated[int, operator.add]


class GraphExplorerParserState(BaseParserState):
    pids: Annotated[set[str], merge_sets]
    explored_entities: Annotated[dict, merge_dictionaries]


class Action:
    possible_actions = [
        "get_wikidata_entry",
        "search_wikidata",
        "execute_sparql",
        "get_property_examples",
        "stop",
    ]

    # All actions have a single input parameter for now
    def __init__(
        self, thought: str, action_name: str, action_argument: str
    ):
        self.thought = thought
        self.action_name = action_name
        self.action_argument = action_argument
        self.observation = None
        self.observation_markdown = None

        assert self.action_name in Action.possible_actions

    def to_jinja_string(self, include_observation: bool) -> str:
        if not self.observation:
            observation = "Did not find any results."
        else:
            observation = self.observation
        ret = f"Thought: {self.thought}\nAction: {self.action_name}({self.action_argument})\n"
        if include_observation:
            ret += f"Observation: {observation}\n"
        return ret
    
    async def print_chainlit(self):
        async with cl.Step(name="Thought", type="tool", disable_feedback=False) as step_thought:
            step_thought.output = self.thought
        
            async with cl.Step(name="Action", type="tool", language="sql", disable_feedback=False) as step_action:
                step_action.output = f"{self.action_name}({self.action_argument})"
                
            if not self.observation:
                observation = "Did not find any results."
            elif self.observation_markdown:
                # if markdown is available, this is from `execute_sparql`, use it
                observation = self.observation_markdown
            else:
                observation = self.observation
                
            observation_language = None
            if self.action_name == "get_wikidata_entry":
                observation_language = "python"
            
            async with cl.Step(name="Observation", type="tool", language=observation_language) as step_observation:
                step_observation.output = observation
        return step_thought, step_action, step_observation
        

    def __repr__(self) -> str:
        if not self.observation:
            observation = "Did not find any results."
        else:
            observation = self.observation
        return f"Thought: {self.thought}\nAction: {self.action_name}({self.action_argument})\nObservation: {observation}"

    def __eq__(self, other):
        if not isinstance(other, Action):
            return NotImplemented
        return (
            self.action_name == other.action_name
            and self.action_argument == other.action_argument
        )

    def __hash__(self):
        return hash((self.action_name, self.action_argument))


class PartToWholeParserState(BaseParserState):
    actions: Annotated[Sequence[Action], add_item_to_list]


def state_to_dict(state: BaseParserState):
    """
    This is defined outside of the DialogueState class because TypedDict classes cannot have methods
    """
    j = dict(state)  # copy
    if "pids" in j:
        j["pids"] = list(j["pids"])
    return j


def state_to_string(state: BaseParserState):
    return json.dumps(state_to_dict(state), indent=2, ensure_ascii=False, default=vars)
