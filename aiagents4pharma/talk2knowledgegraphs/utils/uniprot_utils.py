# Import necessary libraries
import time
import json
import zlib
import requests
from requests.adapters import HTTPAdapter, Retry
from urllib.parse import urlparse, parse_qs

# Define variables to perform UniProt ID mapping
# Adopted from https://www.uniprot.org/help/id_mapping
API_URL = "https://rest.uniprot.org"
POLLING_INTERVAL = 5
retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retries))

def submit_id_mapping(from_db, to_db, ids) -> str:
    """
    Function to submit a job to perform ID mapping.

    Args:
        from_db (str): The source database.
        to_db (str): The target database.
        ids (list): The list of IDs to map.

    Returns:
        str: The job ID.
    """
    request = requests.post(f"{API_URL}/idmapping/run",
                            data={"from": from_db,
                                  "to": to_db,
                                  "ids": ",".join(ids)},)
    try:
        request.raise_for_status()
    except requests.HTTPError:
        print(request.json())
        raise

    return request.json()["jobId"]

def check_id_mapping_results_ready(job_id):
    """
    Function to check if the ID mapping results are ready.

    Args:
        job_id (str): The job ID.

    Returns:
        bool: True if the results are ready, False otherwise.
    """
    while True:
        request = session.get(f"{API_URL}/idmapping/status/{job_id}")

        try:
            request.raise_for_status()
        except requests.HTTPError:
            print(request.json())
            raise

        j = request.json()
        if "jobStatus" in j:
            if j["jobStatus"] in ("NEW", "RUNNING"):
                print(f"Retrying in {POLLING_INTERVAL}s")
                time.sleep(POLLING_INTERVAL)
            else:
                raise Exception(j["jobStatus"])
        else:
            return bool(j["results"] or j["failedIds"])

def get_id_mapping_results_link(job_id):
    """
    Function to get the link to the ID mapping results.

    Args:
        job_id (str): The job ID.

    Returns:
        str: The link to the ID mapping results.
    """
    url = f"{API_URL}/idmapping/details/{job_id}"
    request = requests.Session().get(url)

    try:
        request.raise_for_status()
    except requests.HTTPError:
        print(request.json())
        raise

    return request.json()["redirectURL"]

def decode_results(response, file_format, compressed):
    """
    Function to decode the ID mapping results.

    Args:
        response (requests.Response): The response object.
        file_format (str): The file format of the results.
        compressed (bool): Whether the results are compressed.

    Returns:
        str: The ID mapping results
    """

    if compressed:
        decompressed = zlib.decompress(response.content, 16 + zlib.MAX_WBITS)
        if file_format == "json":
            j = json.loads(decompressed.decode("utf-8"))
            return j
        elif file_format == "tsv":
            return [line for line in decompressed.decode("utf-8").split("\n") if line]
        elif file_format == "xlsx":
            return [decompressed]
        elif file_format == "xml":
            return [decompressed.decode("utf-8")]
        else:
            return decompressed.decode("utf-8")
    elif file_format == "json":
        return response.json()
    elif file_format == "tsv":
        return [line for line in response.text.split("\n") if line]
    elif file_format == "xlsx":
        return [response.content]
    elif file_format == "xml":
        return [response.text]
    return response.text

def get_id_mapping_results_stream(url):
    """
    Function to get the ID mapping results from a stream.

    Args:
        url (str): The URL to the ID mapping results.

    Returns:
        str: The ID mapping results.
    """
    if "/stream/" not in url:
        url = url.replace("/results/", "/results/stream/")

    request = session.get(url)

    try:
        request.raise_for_status()
    except requests.HTTPError:
        print(request.json())
        raise

    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    file_format = query["format"][0] if "format" in query else "json"
    compressed = (
        query["compressed"][0].lower() == "true" if "compressed" in query else False
    )
    return decode_results(request, file_format, compressed)