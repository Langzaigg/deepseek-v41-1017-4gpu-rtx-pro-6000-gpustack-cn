"""GPUStack entrypoint: local checkpoint only; no model or package download.

Serving flags and SM120/Engram adapter originate from the pinned upstream
recipe in PROVENANCE.txt. This wrapper has not been tested on GPUs.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import sys

GIB = 2**30


def int_env(name, default, minimum, maximum):
    value = int(os.environ.get(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def check_checkpoint(root):
    if not root.is_dir():
        raise ValueError(f"Model directory is not mounted: {root}")
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != "deepseek_v41":
        raise ValueError("Expected official DeepSeek-V4.1 checkpoint")
    quant = config.get("quantization_config", {})
    if quant.get("quant_method") != "fp8" or quant.get("expert_dtype") != "fp4":
        raise ValueError("This image expects official mixed FP8/FP4 weights")
    if not (root / "tokenizer.json").is_file():
        raise ValueError("tokenizer.json is missing")
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    for name in set(index.values()):
        shard = (root / name).resolve()
        if not shard.is_relative_to(root) or not shard.is_file() or shard.stat().st_size < 8:
            raise ValueError(f"Missing/invalid weight shard: {name}")
    table_files = {
        index[f"layers.{layer}.engram.embed.{kind}"]
        for layer in (1, 14) for kind in ("weight", "scale")
    }
    return sum((root / name).stat().st_size for name in table_files)


def check_host_memory(table_bytes):
    needed = table_bytes + 24 * GIB
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(mem["MemAvailable"].split()[0]) * 1024
    if available < needed:
        raise RuntimeError(f"RAM mode needs >= {needed/GIB:.1f} GiB currently available; found {available/GIB:.1f}")
    # cgroup v2: reject a container budget that is tighter than host MemAvailable.
    maximum = Path("/sys/fs/cgroup/memory.max")
    current = Path("/sys/fs/cgroup/memory.current")
    if maximum.exists() and current.exists():
        cap = maximum.read_text().strip()
        if cap != "max" and int(cap) - int(current.read_text()) < needed:
            raise RuntimeError("Container cgroup memory headroom is too small for RAM mode")
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if soft != resource.RLIM_INFINITY and soft < table_bytes:
        if hard == resource.RLIM_INFINITY or hard >= table_bytes:
            resource.setrlimit(resource.RLIMIT_MEMLOCK, (hard, hard))
        else:
            status = Path("/proc/self/status").read_text().splitlines()
            caps = next(line.split()[1] for line in status if line.startswith("CapEff:"))
            if not int(caps, 16) & (1 << 14):  # CAP_IPC_LOCK
                raise RuntimeError("RAM mode requires CAP_IPC_LOCK or sufficient memlock ulimit")


def make_command(args, root):
    context = int_env("CONTEXT_LENGTH", 524288, 1024, 1048576)
    concurrency = int_env("MAX_RUNNING_REQUESTS", 8, 1, 32)
    total = int_env("MAX_TOTAL_TOKENS", 4200000, context, 16777216)
    chunk = int_env("CHUNKED_PREFILL_SIZE", 2048, 128, context)
    if chunk % 128:
        raise ValueError("CHUNKED_PREFILL_SIZE must be divisible by 128")
    fraction = float(os.environ.get("MEMORY_FRACTION", "0.95"))
    if not 0.5 <= fraction <= 0.95:
        raise ValueError("MEMORY_FRACTION must be in [0.5, 0.95]")
    spec = os.environ.get("ENABLE_DSPARK", "1")
    if spec not in ("0", "1"):
        raise ValueError("ENABLE_DSPARK must be 0 or 1")
    command = [sys.executable, "-m", "sglang.launch_server",
        "--model-path", str(root), "--served-model-name", args.served_model_name,
        "--host", args.host, "--port", str(args.port),
        "--trust-remote-code", "--load-format", "safetensors", "--tp", "4", "--ep-size", "4",
        "--attention-backend", "dsv4", "--moe-runner-backend", "flashinfer_mxfp4",
        "--mem-fraction-static", str(fraction), "--chunked-prefill-size", str(chunk),
        "--context-length", str(context), "--max-running-requests", str(concurrency),
        "--max-total-tokens", str(total), "--cuda-graph-max-bs-decode", str(concurrency),
        "--min-free-slots-delay", "1", "--random-seed", "0",
        "--enable-decoder-swa-bounded-replay", "--tool-call-parser", "deepseekv41",
        "--reasoning-parser", "deepseek-v41"]
    if spec == "1":
        command += ["--speculative-algorithm", "DSPARK", "--speculative-dspark-block-size", "5"]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--served-model-name", default="deepseek-v4.1-flash-1017")
    parser.add_argument("--preflight", action="store_true", help="Validate without loading model weights")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be 1..65535")
    root = Path(args.model_path).resolve()
    command = make_command(args, root)
    mode = os.environ.get("OFFLOAD_MODE", "ram")
    if mode not in ("ram", "nvme"):
        raise ValueError("OFFLOAD_MODE must be ram or nvme")
    if os.environ.get("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE", "0") not in ("0", "", "false", "False"):
        raise ValueError("Do not mix the upstream host-table switch with this image's Engram adapter")
    table_bytes = check_checkpoint(root)
    if mode == "ram":
        check_host_memory(table_bytes)
    import torch
    if torch.cuda.device_count() != 4:
        raise RuntimeError(f"Expected exactly 4 CUDA-visible GPUs, found {torch.cuda.device_count()}")
    devices = []
    for i in range(4):
        prop = torch.cuda.get_device_properties(i)
        if (prop.major, prop.minor) != (12, 0) or prop.total_memory < 90 * GIB:
            raise RuntimeError(f"GPU {i} must be an SM120 96GB-class GPU; found {prop}")
        devices.append({"index": i, "name": prop.name, "memory_gib": round(prop.total_memory/GIB, 1)})
    environment = dict(os.environ)
    environment.update(DSV41_SOURCE=str(root), OFFLOAD_MODE=mode,
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    old_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = "/opt/dsv41/adapter" + (":" + old_path if old_path else "")
    print(json.dumps({"mode": mode, "gpus": devices, "command": command,
        "engram_backing_gib": round(table_bytes/GIB, 2)}, ensure_ascii=False), flush=True)
    if args.preflight:
        print("PREFLIGHT PASSED: model files/memory/device checks only; inference is not yet validated.", flush=True)
        return
    # A fresh interpreter installs sitecustomize after DSV41_SOURCE is populated.
    # No inner --api-key: GPUStack is the authenticated frontend.
    os.execvpe(sys.executable, command, environment)


if __name__ == "__main__":
    main()
