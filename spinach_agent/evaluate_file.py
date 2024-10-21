import argparse
import json
import asyncio

from evaluate_parser import calculate_metrics

## ToG baseline specific functions
def get_tog_baseline_questions():
    with open("/home/oval/wikidata-dataset/datasets/qald_10/en/tog_results.json", "r") as fd:
        tog = json.load(fd)
        
    return [i["question"] for i in tog]

async def evaluate_file(
    filename,
):
    with open(filename, "r") as fd:
        results = json.load(fd)
    
    # uncomment this for getting the tog subset of qald-10  
    # results = [
    #     entry for entry in results if entry["question"] in get_tog_baseline_questions()
    # ]
            
    calculate_metrics(results, use_existing_predictions=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Name of the JSON file to evaluate",
    )
    
    args = parser.parse_args()
    asyncio.run(evaluate_file(
        args.input,
    ))