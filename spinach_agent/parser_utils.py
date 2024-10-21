import asyncio
import warnings

from chainlite import chain, get_logger, llm_generation_chain, pprint_chain
from json_repair import repair_json
from langchain_core.runnables import Runnable

from spinach_agent.parser_state import SparqlQuery
from wikidata_utils import get_p_or_q_id_from_name

logger = get_logger(__name__)
warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)  # from ReFinED


class BaseParser:
    @classmethod
    def initialize(engine: str):
        raise NotImplementedError("Subclasses should implement this method")

    @classmethod
    def run_batch(cls, questions: list[str]):
        return asyncio.run(
            cls.runnable.with_config(
                {"recursion_limit": 60, "max_concurrency": 50}
            ).abatch(questions)
        )


def get_prune_edges_chain(engine: str) -> Runnable:
    return (
        llm_generation_chain(
            template_file="prune_outgoing_edges.prompt",
            engine=engine,
            max_tokens=1000,
        )
        | parse_string_to_json
    )


@chain
async def parse_string_to_json(output: str) -> dict:
    return repair_json(output, return_objects=True)


@chain
def process_pid_qid_output(output: dict) -> dict:
    if not isinstance(output, dict):
        return {}
    new_ret = {}
    for k, v in output.items():
        # Remove wd: etc.
        if ":" in k:
            k = k[k.find(":") + 1 :]
        new_ret[k] = v

    return new_ret


@chain
def search_wikidata_for_ids(_input):
    ret = {}
    for k, v in _input.items():
        _id = get_p_or_q_id_from_name(v, type="qid" if k.startswith("Q") else "pid")
        # print(f"{k}, {v} => {_id}")
        if _id:
            k = _id
            if k.startswith(("Q", "P")):
                ret[k] = v
    return ret


@chain
def extract_code_block_from_output(llm_output: str, code_block: str) -> str:
    code_block = code_block.lower()
    if f"```{code_block}" in llm_output.lower():
        start_idx = llm_output.lower().rfind(f"```{code_block}") + len(
            f"```{code_block}"
        )
        end_idx = llm_output.lower().rfind("```", start_idx)
        if end_idx < 0:
            # because llm_generation_chain does not include the stop token
            end_idx = len(llm_output)
        extracted_block = llm_output[start_idx:end_idx].strip()
        return extracted_block
    else:
        raise ValueError(f"Expected a code block, but llm output is {llm_output}")


@chain
def sparql_string_to_sparql_object(sparql: str) -> SparqlQuery:
    return SparqlQuery(sparql=sparql)


@chain
def execute_sparql_object(sparql: SparqlQuery):
    sparql.execute()
    return sparql
