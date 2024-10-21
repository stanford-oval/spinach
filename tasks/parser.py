import os

import docker
from chainlite import get_logger
from invoke import task

from tasks.common import (
    DEFAULT_WORKDIR,
    check_if_docker_container_is_running,
    get_container_by_name,
    load_api_keys,
    start_redis,
    wait_for_container_to_be_ready,
)

logger = get_logger(__name__)


@task
def start_parser_docker(
    c,
    workdir=DEFAULT_WORKDIR,
    port=5004,
    parser_model="/data/wikisp_qald7_llama2",
):
    client = docker.from_env()
    is_running = check_if_docker_container_is_running(
        client, f"text-generation-inference"
    )
    if is_running:
        logger.info(
            "text-generation-inference docker container is already running.",
        )
        return

    container = get_container_by_name(f"text-generation-inference")
    if container:
        # container already exists, just stopped
        logger.info("Starting stopped docker container")

        container.start()
        return

    # Get the current working directory and construct the volume path
    current_directory = os.getcwd()

    # Convert the volume specification into a format the Docker SDK expects
    volumes = {
        os.path.join(current_directory, workdir): {
            "bind": "/data",
            "mode": "rw",
        },
    }

    try:

        container = client.containers.run(
            "ghcr.io/huggingface/text-generation-inference:2.0.3",
            detach=True,
            device_requests=[docker.types.DeviceRequest(capabilities=[["gpu"]])],
            ports={"80": port},
            volumes=volumes,
            shm_size="1g",
            name=f"text-generation-inference",
            command=[
                "--model-id",
                parser_model,
                "--max-batch-total-tokens",
                "10000000",
                "--hostname",
                "0.0.0.0",
                "--cuda-memory-fraction",
                "0.3",  # TODO
            ],
        )
        logger.info(
            "text-generation-inference docker container started, id=%s",
            container.id,
        )

        wait_for_container_to_be_ready(client, container)
    except Exception as e:
        logger.error(
            "Failed to start text-generation-inference container: %s",
            str(e),
        )
    client.close()


@task(pre=[start_redis, load_api_keys], aliases=["evaluate_file"])
def eval_file(c, input_file="eval_log.json", field="all", output_file="eval.json"):
    c.run(
        f"python spinach_agent/evaluate_file.py "
        f"--file_name {input_file} "
        f"--field {field} "
        f"--output_file {output_file} "
        f"--eval_EM "
        f"--eval_substring "
        f"--eval_GPT_judge"
    )


@task(pre=[start_redis, load_api_keys], aliases=["eval_parser"])
def evaluate_parser(
    c,
    parser_type,
    engine="gpt-4o",
    dataset="new_dataset/dev_cleaned.json",
    subsample=3,
    offset=0,
    output_file="output.json",
    regex_use_select_distinct_and_id_not_label=False,
    llm_extract_prediction_if_null=False,
):
    command = (
        f"python spinach_agent/evaluate_parser.py "
        f"--engine {engine} "
        f"--dataset {dataset} "
        f"--subsample {subsample} "
        f"--offset {offset} "
        f"--output_file {output_file} "
        f"--parser_type {parser_type} "
    )
    
    # Add optional flags based on conditions
    if regex_use_select_distinct_and_id_not_label:
        command += "--regex_use_select_distinct_and_id_not_label "
    if llm_extract_prediction_if_null:
        command += "--llm_extract_prediction_if_null "

    c.run(command)
