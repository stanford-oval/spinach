<p align="center">
    <img src="./images/Wikidata-logo-en.svg" width="100px" alt="Wikidata" />
    <h1 align="center">
        <b>SPINACH: <u>SP</u>ARQL-Based <u>I</u>nformation <u>Na</u>vigation for <u>Ch</u>allenging Real-World Questions</b>
        <br>
        <a href="https://arxiv.org/abs/2407.11417">
            <img src="https://img.shields.io/badge/cs.CL-2407.11417-b31b1b" alt="arXiv">
        </a>
        <a href="https://github.com/stanford-oval/spinach/stargazers">
            <img src="https://img.shields.io/github/stars/stanford-oval/spinach?style=social" alt="Github Stars">
        </a>
    </h1>
</p>

<p align="center">
    Online Chatbot:
    <a href="https://spinach.genie.stanford.edu" target="_blank">
        https://spinach.genie.stanford.edu
    </a>
    <br>
</p>

# About

**The SPINACH dataset**: Current KBQA datasets lack real-world complexity. The SPINACH KBQA dataset, collected from Wikidata's Request a Query sites, is the first to cover both natural questions and complex SPARQLs

**The SPINACH agent**: The SPINACH agent is a new KBQA approach that mimics expert human SPARQL writing, achieving SOTA on many KBQA datasets. You can try it at https://spinach.genie.stanford.edu

For more details, check out this blog post on [Wikimedia Research Newsletter](https://meta.m.wikimedia.org/wiki/Research:Newsletter/2024/November).

# Folder Structure
`datasets/` contains all prior dataset files. Predictions for the SPINACH agent used in the paper can be found at:
- `datasets/qald_7_task4/spinach_output_test.json` for QALD-7
- `datasets/qald_9_plus/en/spinach_output_test.json` for QALD-9-plus
- `datasets/qald_10/en/spinach_output_test.json` for QALD-10 full set (the prediction for the ToG subset can be retrieved by uncommenting the portion using `get_tog_baseline_questions` in `evaluate_file.py`)
- `datasets/wikiwebquestions/spinach_output_dev.json` and `datasets/wikiwebquestions/spinach_output_test.json` for WikiWebQuestions

`spinach_dataset/` contains the dev and test set of the SPINACH dataset. The SPINACH agent's outputs are also stored in this directory.

`spinach_agent/` contains the implementation for the SPINACH agent.

`notebooks/` stores various Jupyter notebooks used to crawl the initial conversations and compute dataset complexity metrics.

`tasks/` stores the files declaring how to use the `invoke` command.

`tests/` contains all tests, which use `pytest`. You can run all tests by running `invoke tests`. `test_eval.py`, which stores test cases for the row-major F1 implementation, can be run via `python tests/test_eval.py`.

# Running the SPINACH agent and evaluating results

## Set up environment

Run `conda env create -f conda_env.yaml`.

Create a file called `API_KEYS` and write various API keys inside. The format is one key per line, for example `OPENAI_API_KEY=sk-...`

## Run SPINACH parser and evaluate

```
inv evaluate-parser --parser-type part_to_whole --subsample=-1 --engine=gpt-4o --dataset=datasets/qald_10/en/test.json --output-file=datasets/qald_10/en/spinach_output_test.json --regex-use-select-distinct-and-id-not-label --llm-extract-prediction-if-null
```

The two flags at the end are for:

- `llm-extract-prediction-if-null`: If a reasoning chain ended without any predicted SPARQL, asks a LLM to return a SPARQL. This part is implemented inside `extract_sparql.ainvoke`. This is helpful because for simple queries, LLMs could just use ``get_wikidata_entry'' to get results instead of ever writing a SPARQL. We enabled this flag for all datasets we evaluated on.
- `--regex-use-select-distinct-and-id-not-label`: Attempts to use regex to force use `SELECT DISTINCT` instead of `SELECT`, and try to always include the variable QID instead of the label (i.e., use `x` instead of `xLabel`). We enabled this for all datasets except the new SPINACH dataset that we evaluated on (The SPINACH dataset involves more complex predicted SPARQLs. The regex is not sophisticated enough to handle these cases.)

The script will also write a `.log` file with SPINACH's chain of reasonings and actions with the same file name as the `.json` output.

You can re-evaluate the output simply from the `.json` file:
```
python spinach_agent/evaluate_file.py --input datasets/qald_10/en/spinach_output_test.json
```

If you'd like to simply run the parser on a list of questions, use the following code from `evaluate_parser.py`:
```python
from spinach_agent.part_to_whole_parser import PartToWholeParser

semantic_parser_class = PartToWholeParser
semantic_parser_class.initialize(engine=args.engine) # e.g. "gpt-4o"

chain_output = semantic_parser_class.run_batch(
    questions, # this should be a dict of {"question": "...", "conversation_history": [...]}, conversation_history can be empty list if running on single-turn questions
)
```


# License

The code in this repo is released under Apache License, version 2.0. The SPINACH dataset, derived from the Wikidata Request a Query forum, is released under the CC BY-SA 4.0 license, the same license that covers the forum.

# Citation

```
@misc{liu2024spinachsparqlbasedinformationnavigation,
      title={SPINACH: SPARQL-Based Information Navigation for Challenging Real-World Questions}, 
      author={Shicheng Liu and Sina J. Semnani and Harold Triedman and Jialiang Xu and Isaac Dan Zhao and Monica S. Lam},
      year={2024},
      eprint={2407.11417},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.11417}, 
}
```
