import os
from time import sleep

import docker
import redis
from chainlite import get_logger
from invoke import task

logger = get_logger(__name__)

DEFAULT_REDIS_PORT = 6379
DEFAULT_WORKDIR = "workdir"


@task
def load_api_keys(c):
    try:
        with open("API_KEYS") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = tuple(line.split("=", 1))
                    key, value = key.strip(), value.strip()
                    os.environ[key] = value
                    logger.debug("Loaded API key named %s", key)

    except Exception as e:
        logger.error(
            "Error while loading API keys from API_KEY. Make sure this file exists, and has the correct format. %s",
            str(e),
        )


@task()
def start_redis(c, redis_port=DEFAULT_REDIS_PORT):
    try:
        r = redis.Redis(host="localhost", port=redis_port)
        r.ping()
    except redis.exceptions.ConnectionError:
        logger.info("Redis server not found, starting it now...")
        c.run(f"redis-server --port {redis_port} --daemonize yes")
        return

    logger.debug("Redis server is aleady running.")


@task(pre=[load_api_keys, start_redis], aliases=["test"])
def tests(c):
    """Run tests using pytest"""
    c.run(
        "pytest "
        "-rP "
        "--color=yes "
        "--disable-warnings "
        "./tests/test_spinach_agent.py "
        "./tests/test_wikidata_utils.py "
        "./tests/test_python_interface.py ",
        pty=True,
    )


def get_container_by_name(container_name: str):
    client = docker.from_env()
    all_containers = client.containers.list(all=True)
    for container in all_containers:
        if container_name == container.name:
            client.close()
            return container

    client.close()
    return None


def check_if_docker_container_is_running(docker_client, container_name: str):
    # List all running containers
    running_containers = docker_client.containers.list()

    for container in running_containers:
        # Check if the specified container name matches any running container's name
        # Container names are stored in a list
        if container_name in container.name:
            return True

    # If no running container matched the specified name
    return False


def wait_for_container_to_be_ready(docker_client, container):
    timeout = 60
    stop_time = 3
    elapsed_time = 0
    logger.info("Waiting for the container to be ready...")

    def is_ready():
        return docker_client.containers.get(container.id).status == "running"

    while not is_ready() and elapsed_time < timeout:
        sleep(stop_time)
        logger.info(container.logs())
        elapsed_time += stop_time

    if not is_ready():
        logger.error("Docker container still not running after %d seconds.", timeout)


@task
def stop_container(c, container_name_prefix):
    """
    Stops a specified Docker container if it is running.

    This function checks if a Docker container whose given name starts with `container_name_prefix` is running. If the container is already
    stopped, it logs this information and takes no further action. If the container is running, this function
    proceeds to stop it. It requires the Docker SDK for Python (`docker` package) to interact with Docker.

    Args:
        c: The context parameter for Invoke tasks, automatically passed by Invoke.
        container_name_prefix (str): The name of the Docker container to stop starts with this string.

    Note:
        This function requires Docker to be installed and running on the host system.
    """
    client = docker.from_env()
    # List all running containers
    running_containers = client.containers.list()

    for container in running_containers:
        if container.name.startswith(container_name_prefix):
            container.stop()

    client.close()
