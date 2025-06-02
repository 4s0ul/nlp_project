import json
import requests
from loguru import logger
from tqdm import tqdm
import time
from typing import Dict, Any

# API base URL
API_BASE_URL = "http://localhost:8000"


def create_topic(topic_name: str) -> str:
    """
    Creates a topic via the /topics/ endpoint and returns its ID.

    Args:
        topic_name (str): Name of the topic.

    Returns:
        str: Topic ID.

    Raises:
        Exception: If topic creation fails.
    """
    url = f"{API_BASE_URL}/topics/"
    payload = {"name": topic_name, "info": f"Topic for {topic_name} terms"}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            topic_id = response.json()["id"]
            logger.info(f"Created topic: {topic_name} (ID: {topic_id})")
            return topic_id
        elif response.status_code == 409:
            # Topic already exists, fetch its ID
            response = requests.get(url)
            if response.status_code == 200:
                for topic in response.json():
                    if topic["name"] == topic_name:
                        logger.info(f"Topic already exists: {topic_name} (ID: {topic['id']})")
                        return topic["id"]
            raise Exception(f"Failed to find existing topic: {topic_name}")
        else:
            raise Exception(f"Failed to create topic: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Error creating topic {topic_name}: {e}")
        raise


def create_word(topic_id: str, raw_text: str, description_raw_text: str, language: str = "russian") -> Dict[str, Any]:
    """
    Creates a word via the /words/ endpoint, including description and triplets.

    Args:
        topic_id (str): ID of the topic.
        raw_text (str): Raw text of the word (Russian translation).
        description_raw_text (str): Raw text of the description (Russian definition).
        language (str): Language of the word (default: russian).

    Returns:
        Dict[str, Any]: API response containing word data.

    Raises:
        Exception: If word creation fails.
    """
    url = f"{API_BASE_URL}/words/"
    payload = {
        "topic_id": topic_id,
        "raw_text": raw_text,
        "description_raw_text": description_raw_text,
        "language": language,
        "info": f"Auto-generated from dataset"
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 201:
            logger.debug(f"Created word: {raw_text} (ID: {response.json()['id']})")
            return response.json()
        else:
            raise Exception(f"Failed to create word: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Error creating word {raw_text}: {e}")
        raise


def build_knowledge_graph(language: str = "russian") -> Dict[str, Any]:
    """
    Calls the /graph/ endpoint to build the knowledge graph.

    Args:
        language (str): Language for the graph (default: russian).

    Returns:
        Dict[str, Any]: Graph data (nodes, edges, counts).

    Raises:
        Exception: If graph building fails.
    """
    url = f"{API_BASE_URL}/graph/?language={language}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            graph_data = response.json()
            logger.info(f"Graph built: {graph_data['node_count']} nodes, {graph_data['edge_count']} edges")
            return graph_data
        else:
            raise Exception(f"Failed to build graph: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Error building graph: {e}")
        raise


def visualize_knowledge_graph(language: str = "russian") -> Dict[str, Any]:
    """
    Calls the /graph/visualize/ endpoint to visualize the knowledge graph.

    Args:
        language (str): Language for the graph (default: russian).

    Returns:
        Dict[str, Any]: Visualization response.

    Raises:
        Exception: If visualization fails.
    """
    url = f"{API_BASE_URL}/graph/visualize/?language={language}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            logger.info(f"Graph visualization completed: {response.json()['message']}")
            return response.json()
        else:
            raise Exception(f"Failed to visualize graph: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Error visualizing graph: {e}")
        raise


def populate_database_via_api(data_file: str, topic_name: str = "Science", max_retries: int = 3,
                              retry_delay: float = 1.0):
    """
    Populates the database using FastAPI endpoints and builds/visualizes the knowledge graph.

    Args:
        data_file (str): Path to JSON file containing term data.
        topic_name (str): Name of the topic to group terms.
        max_retries (int): Maximum number of retries for failed API calls.
        retry_delay (float): Delay between retries in seconds.
    """
    # Read JSON data
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            terms_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read JSON file: {e}")
        return

    if len(terms_data) != 3000:
        logger.warning(f"Expected 3000 terms, got {len(terms_data)}")

    # Create topic
    try:
        topic_id = create_topic(topic_name)
    except Exception as e:
        logger.error(f"Aborting due to topic creation failure: {e}")
        return

    # Process each term
    language = "russian"
    successful = 0
    for term_data in tqdm(terms_data, desc="Populating database via API"):
        raw_text = term_data.get('translation', '').strip()
        desc_raw_text = term_data.get('definition', '').strip()

        if not raw_text or not desc_raw_text:
            logger.warning(f"Skipping term due to empty fields: {term_data}")
            continue

        # Retry logic for API calls
        for attempt in range(max_retries):
            try:
                create_word(topic_id, raw_text, desc_raw_text, language)
                successful += 1
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Retry {attempt + 1}/{max_retries} for {raw_text}: {e}")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to create word after {max_retries} attempts: {raw_text}")
                    break

    logger.info(f"Successfully created {successful}/{len(terms_data)} words")

    # Build and visualize knowledge graph
    try:
        graph_data = build_knowledge_graph(language)
        visualize_knowledge_graph(language)
    except Exception as e:
        logger.error(f"Failed to build/visualize graph: {e}")


if __name__ == "__main__":
    # Example usage
    populate_database_via_api(
        data_file="unique_terms.json",
        topic_name="Химические Термины",
        max_retries=3,
        retry_delay=1.0
    )
    populate_database_via_api(
        data_file="terms_definitions.json",
        topic_name="Физические Термины",
        max_retries=3,
        retry_delay=1.0
    )
