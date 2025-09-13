import io
import os
import time
import json
import base64
import logging
import requests
import hydra
from typing import Any, Dict
from PIL import Image
from pdf2image import convert_from_path


# Set up logging with configurable level
log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, log_level))

# Load configuration and start logging
def load_hydra_config() -> Any:
    """
    Load the configuration using Hydra and return the configuration for the multimodal processor.
    """
    with hydra.initialize(version_base=None, config_path="../../../configs"):
        cfg = hydra.compose(
            config_name="config",
            overrides=["tools/multimodal_processor=default"],
        )
        config = cfg.tools.multimodal_processor
        logger.debug("Loaded Multimodal Processor configuration.")
        return config
config = load_hydra_config()
# Utility Functions
MAX_B64_SIZE = 180_000

def compress_image_to_target_size(image, max_bytes=MAX_B64_SIZE, min_quality=20, min_width=400):
    """
    Compress image to guaranteed < max_bytes base64 length.
    Returns (b64_string, success_flag).
    """
    quality = 85
    width, height = image.size

    while quality >= min_quality and width >= min_width:
        resized_image = image.resize((width, int(height * width / image.width)), Image.LANCZOS)
        buffer = io.BytesIO()
        resized_image.save(buffer, format="JPEG", quality=quality, optimize=True)
        b64_encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        if len(b64_encoded) <= max_bytes:
            return b64_encoded, True

        # Shrink dimensions and quality further
        width = int(width * 0.95)
        quality -= 5

    # Final check – if still too large, drop
    if len(b64_encoded) > max_bytes:
        logger.warning(f"Compression failed: final size {len(b64_encoded)} > {max_bytes}. Returning empty string.")
        return "", False

    return b64_encoded, True


def pdf_to_base64_compressed(pdf_path, max_b64_size=MAX_B64_SIZE, dpi=150):
    images = convert_from_path(pdf_path, dpi=dpi)
    results = []

    for i, image in enumerate(images):
        page_num = i + 1
        try:
            b64_string, under_limit = compress_image_to_target_size(image, max_bytes=max_b64_size)
            if not under_limit:
                b64_string = ""  # enforce empty if oversized
            logger.info(f"Page {page_num} | Under limit: {under_limit} | Size: {len(b64_string)} chars")
            results.append({"page": page_num, "base64": b64_string})
        except Exception as e:
            logger.error(f"Page {page_num} failed: {e}")
            results.append({"page": page_num, "base64": ""})

    return results


def crop_categorized_elements(categorized_data, base64_pages, max_bytes=MAX_B64_SIZE, min_quality=30, min_width=100):
    page_b64_map = {p["page"]: p["base64"] for p in base64_pages}

    def crop_regions(boxes):
        cropped_b64, metadata = [], []
        for b in boxes:
            page_num, coords = b.get("page"), b["box"]
            page_b64 = page_b64_map.get(page_num)
            if not page_b64:
                logger.warning(f"No image found for page {page_num}")
                continue
            try:
                img_data = base64.b64decode(page_b64)
                with Image.open(io.BytesIO(img_data)) as img:
                    w, h = img.size
                    crop = img.crop((
                        int(coords["x_min"] * w), int(coords["y_min"] * h),
                        int(coords["x_max"] * w), int(coords["y_max"] * h)
                    )).convert("RGB")

                    # Always compress to guaranteed safe size
                    b64, ok = compress_image_to_target_size(crop, max_bytes=max_bytes,
                                                            min_quality=min_quality, min_width=min_width)
                    if ok:
                        cropped_b64.append({"page": page_num, "b64": b64})
                        metadata.append({
                            "page": page_num, "box": coords,
                            "type": b.get("type"), "page_index": b.get("page_index")
                        })
                    else:
                        logger.warning(f"Region from page {page_num} could not be compressed under {max_bytes}. Skipped.")
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {e}")
        return cropped_b64, metadata

    results = {}
    for key in ["charts", "tables", "infographics"]:
        logger.info(f"Cropping {key}...")
        b64_list, meta_list = crop_regions(categorized_data.get(key, []))
        results[key] = {"base64": b64_list, "metadata": meta_list}
    return results


def detect_page_elements(pdf_base64_list):
    responses = []
    page_elements_url = config.page_elements_url
    headers = config.headers
    
    for item in pdf_base64_list:
        page, img_b64 = item.get("page"), item.get("base64")
        print("Processing page:", page)

        if not img_b64:
            logger.warning(f"Skipping page {page}: no base64 data.")
            responses.append(None)
            continue

        if len(img_b64) > MAX_B64_SIZE:
            logger.warning(f"Skipping page {page}: image too large ({len(img_b64)} chars).")
            responses.append(None)
            continue

        print(f"Using headers: {headers}")

        payload = {"input": [{"type": "image_url", "url": f"data:image/jpeg;base64,{img_b64}"}]}
        try:
            r = requests.post(page_elements_url, headers=headers, json=payload)
            r.raise_for_status()
            res_json = r.json()
            logger.info(f"Response for page {page}: {json.dumps(res_json)[:200]}...")
            time.sleep(1.5)
            responses.append({"page": page, "data": res_json})
        except Exception as e:
            logger.error(f"Request error on page {page}: {e}")
            responses.append(None)

    return responses


def categorize_page_elements(responses):
    categories = {"chart": [], "table": [], "infographic": []}
    pages_by_type = {"chart": set(), "table": set(), "infographic": set()}

    for r in responses:
        if not r or not r.get("data"):
            continue

        page_num = r.get("page")
        page_data_list = r["data"].get("data", [])

        for page_data in page_data_list:
            page_index = page_data.get("index", -1)
            bounding_boxes = page_data.get("bounding_boxes", {})

            for key in categories.keys():
                for box in bounding_boxes.get(key, []):
                    categories[key].append({
                        "page": page_num,
                        "page_index": page_index,
                        "type": key,
                        "box": box
                    })
                    pages_by_type[key].add(page_num)

    return {
        "charts": categories["chart"],
        "tables": categories["table"],
        "infographics": categories["infographic"],
        "pages": {
            "charts": sorted(pages_by_type["chart"]),
            "tables": sorted(pages_by_type["table"]),
            "infographics": sorted(pages_by_type["infographic"])
        }
    }


def crop_categorized_elements(categorized_data, base64_pages, max_bytes=180_000, min_quality=30, min_width=100):
    page_b64_map = {p["page"]: p["base64"] for p in base64_pages}

    def crop_regions(boxes):
        cropped_b64, metadata = [], []
        for b in boxes:
            page_num, coords = b.get("page"), b["box"]
            page_b64 = page_b64_map.get(page_num)
            if not page_b64:
                logger.warning(f"No image found for page {page_num}")
                continue
            try:
                img_data = base64.b64decode(page_b64)
                with Image.open(io.BytesIO(img_data)) as img:
                    w, h = img.size
                    crop = img.crop((int(coords["x_min"] * w), int(coords["y_min"] * h),
                                     int(coords["x_max"] * w), int(coords["y_max"] * h))).convert("RGB")
                    width, height, quality = crop.size[0], crop.size[1], 85
                    while True:
                        buf = io.BytesIO()
                        resized_crop = crop.resize((width, int(height * width / crop.width)), Image.LANCZOS)
                        resized_crop.save(buf, format="JPEG", quality=quality, optimize=True)
                        b64 = base64.b64encode(buf.getvalue()).decode()
                        if len(b64) <= max_bytes or width < min_width or quality < min_quality:
                            break
                        width, quality = int(width * 0.95), quality - 5
                    if len(b64) <= max_bytes:
                        cropped_b64.append({"page": page_num, "b64": b64})
                        metadata.append({"page": page_num, "box": coords, "type": b.get("type"), "page_index": b.get("page_index")})
                    else:
                        logger.warning(f"Region from page {page_num} too large even after compression.")
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {e}")
        return cropped_b64, metadata

    results = {}
    for key in ["charts", "tables", "infographics"]:
        logger.info(f"Cropping {key}...")
        b64_list, meta_list = crop_regions(categorized_data.get(key, []))
        results[key] = {"base64": b64_list, "metadata": meta_list}
    return results


def ocr_with_paddle(cropped_b64_list, metadata):
    logger.info("Running PaddleOCR...")
    ocr_results = []
    headers = config.headers
    for img_b64, meta in zip(cropped_b64_list, metadata):
        payload = {"input": [{"type": "image_url", "url": f"data:image/jpeg;base64,{img_b64}"}]}
        paddle_ocr_url = config.paddle_ocr_url
        try:
            r = requests.post(paddle_ocr_url, headers=headers, json=payload)
            r.raise_for_status()
            ocr_results.append({"page": meta.get("page", -1), "type": meta.get("type", ""), "ocr": r.json(), "base64_image": img_b64})
            logger.info(f"PaddleOCR processed {meta.get('type', '')} (page {meta.get('page', -1)})")
            time.sleep(1.5)
        except Exception as e:
            logger.error(f"PaddleOCR failed (page {meta.get('page', -1)}): {e}")
            ocr_results.append(None)
    return ocr_results


def analyze_with_nvidia(elements_list, api_url, element_type):
    logger.info(f"Analyzing {element_type}...")
    analyzed = []
    headers = config.headers
    for item in elements_list:
        payload = {"input": [{"type": "image_url", "url": f"data:image/jpeg;base64,{item['b64']}"}]}
        try:
            r = requests.post(api_url, headers=headers, json=payload)
            r.raise_for_status()
            analyzed.append({"page": item["page"], "b64": item["b64"], "analysis_result": r.json()})
            logger.info(f"NVIDIA {element_type} processed (page {item['page']})")
            time.sleep(2.0)
        except Exception as e:
            logger.error(f"NVIDIA {element_type} failed (page {item['page']}): {e}")
            analyzed.append(None)
    return analyzed


def process_all(cropped_results):
    final_results = {"charts": [], "tables": [], "infographics": []}
    chart_url = config.chart_url
    table_url = config.table_url

    # Process charts
    for item, meta in zip(
        cropped_results.get("charts", {}).get("base64", []),
        cropped_results.get("charts", {}).get("metadata", []),
    ):
        chart_analysis = analyze_with_nvidia([item], chart_url, "chart")
        chart_ocr = ocr_with_paddle([item["b64"]], [meta])
        final_results["charts"].append(
            {"nvidia": chart_analysis[0], "ocr": chart_ocr[0]}
        )
 # Process tables
    for item, meta in zip(
        cropped_results.get("tables", {}).get("base64", []),
        cropped_results.get("tables", {}).get("metadata", []),
    ):
        table_analysis = analyze_with_nvidia([item], table_url, "table")
        table_ocr = ocr_with_paddle([item["b64"]], [meta])
        final_results["tables"].append(
            {"nvidia": table_analysis[0], "ocr": table_ocr[0]}
        )

    # Process infographics (OCR only)
    for item, meta in zip(
        cropped_results.get("infographics", {}).get("base64", []),
        cropped_results.get("infographics", {}).get("metadata", []),
    ):
        infographic_ocr = ocr_with_paddle([item["b64"]], [meta])
        final_results["infographics"].append({"ocr": infographic_ocr[0]})

    return final_results

# OCR Post-Processing Helpers
def collect_ocr_results(final_results):
    """Flatten final_results into a list of raw OCR results."""
    logger.info("Collecting OCR results...")
    ocr_entries = []
    for category, items in final_results.items():
        for item in items:
            if item and "ocr" in item and item["ocr"]:
                ocr_entries.append(item["ocr"])
    return ocr_entries


def extract_text_lines(data):
    """
    Extract page, type, and text from PaddleOCR output.
    Works with both JSON string and Python dict/list.
    """
    logger.info("Extracting text lines from OCR data...")
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            raise ValueError("Input is a string but not valid JSON")

    if isinstance(data, dict):
        data = [data]

    all_lines = []

    for page in data:
        if not isinstance(page, dict):
            continue

        page_num = page.get("page", -1)
        page_type = page.get("type", "unknown")

        for item in page.get("ocr", {}).get("data", []):
            for detection in item.get("text_detections", []):
                text = detection.get("text_prediction", {}).get("text", "")
                if text:
                    all_lines.append({
                        "page": page_num,
                        "type": page_type,
                        "text": text
                    })

    return all_lines