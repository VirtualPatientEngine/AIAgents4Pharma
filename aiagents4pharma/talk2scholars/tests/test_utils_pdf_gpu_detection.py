"""gpu detection and index configuration tests."""

import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from aiagents4pharma.talk2scholars.tools.pdf.utils.gpu_detection import (
    detect_nvidia_gpu,
    get_optimal_index_config,
    log_index_configuration,
)

# === detect_nvidia_gpu ===


def test_detect_nvidia_gpu_force_cpu_from_config():
    """detect_nvidia_gpu should return False if force_cpu_mode is set."""
    config = SimpleNamespace(gpu_detection=SimpleNamespace(force_cpu_mode=True))
    assert detect_nvidia_gpu(config) is False


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.gpu_detection.subprocess.run")
def test_detect_nvidia_gpu_success(mock_run):
    """detect_nvidia_gpu should return True if NVIDIA GPUs are detected."""
    mock_run.return_value = MagicMock(returncode=0, stdout="NVIDIA A100\nNVIDIA RTX 3090")

    assert detect_nvidia_gpu() is True
    mock_run.assert_called_once()


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.gpu_detection.subprocess.run")
def test_detect_nvidia_gpu_no_output(mock_run):
    """detect_nvidia_gpu should return False if no GPUs are detected."""
    mock_run.return_value = MagicMock(returncode=0, stdout="")

    assert detect_nvidia_gpu() is False


# === get_optimal_index_config ===


def test_get_optimal_index_config_gpu():
    """get_optimal_index_config should return GPU_CAGRA for GPU setup."""
    index_params, search_params = get_optimal_index_config(has_gpu=True, embedding_dim=768)

    assert index_params["index_type"] == "GPU_CAGRA"
    assert "cache_dataset_on_device" in index_params["params"]
    assert search_params["params"]["search_width"] == 16


def test_get_optimal_index_config_cpu():
    """get_optimal_index_config should return IVF_FLAT for CPU setup."""
    index_params, search_params = get_optimal_index_config(has_gpu=False, embedding_dim=768)

    assert index_params["index_type"] == "IVF_FLAT"
    assert index_params["params"]["nlist"] == 96  # 768 / 8 = 96
    assert search_params["params"]["nprobe"] == 16


# === log_index_configuration ===


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.gpu_detection.logger")
def test_log_index_configuration_logs_all(mock_logger):
    """log_index_configuration should log all parameters correctly."""
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128},
    }
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}

    log_index_configuration(index_params, search_params)

    assert mock_logger.info.call_count >= 5


def test_get_optimal_index_config_gpu_without_cosine():
    """Ensure GPU config defaults to IP when use_cosine is False."""
    index_params, search_params = get_optimal_index_config(
        has_gpu=True, embedding_dim=768, use_cosine=False
    )

    assert index_params["index_type"] == "GPU_CAGRA"
    assert index_params["metric_type"] == "IP"
    assert search_params["metric_type"] == "IP"


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.gpu_detection.logger")
def test_log_index_configuration_logs_cosine_simulation_note(mock_logger):
    """Test GPU_CAGRA COSINE -> IP note is logged properly."""
    index_params = {
        "index_type": "GPU_CAGRA",
        "metric_type": "IP",
        "params": {"itopk_size": 128},
    }
    search_params = {
        "metric_type": "IP",
        "params": {"search_width": 16},
    }

    log_index_configuration(index_params, search_params, use_cosine=True)

    log_messages = [str(call.args[0]) for call in mock_logger.info.call_args_list]
    assert any("simulate COSINE for GPU" in msg for msg in log_messages)


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.gpu_detection.logger")
@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.gpu_detection.subprocess.run")
def test_detect_nvidia_gpu_timeout_raises_false(mock_run, mock_logger):
    """detect_nvidia_gpu should return False and log info on subprocess.TimeoutExpired."""
    # Simulate a timeout
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10)

    result = detect_nvidia_gpu()
    assert result is False
    mock_logger.info.assert_called_with("NVIDIA GPU detection failed: %s", mock_run.side_effect)


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.gpu_detection.logger")
@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.gpu_detection.subprocess.run")
def test_detect_nvidia_gpu_file_not_found_raises_false(mock_run, mock_logger):
    """detect_nvidia_gpu should return False and log info on FileNotFoundError."""
    # Simulate nvidia-smi not installed
    mock_run.side_effect = FileNotFoundError("nvidia-smi not found")

    result = detect_nvidia_gpu()
    assert result is False
    mock_logger.info.assert_called_with("NVIDIA GPU detection failed: %s", mock_run.side_effect)
