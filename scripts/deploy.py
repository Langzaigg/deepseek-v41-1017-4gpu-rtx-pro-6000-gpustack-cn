#!/usr/bin/env python3
"""GPUStack 2.2.3 Docker Worker deployment; Python 3.10+, standard library only."""
from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANAGER = "deepseek-v41-1017-gpustack-cn"
MARKER = f"managed-by:{MANAGER}"
VERSION = "1.0.0-custom"
MODEL = "deepseek-v4.1-flash-1017-cn"
IMAGE = "dsv41-1017-gpustack-cn:1.0.0"
ENTRYPOINT = "python3 -u /opt/dsv41/launch.py"
COMMAND = ('--model-path "{{model_path}}" --host {{worker_ip}} '
           '--port {{port}} --served-model-name "{{model_name}}"')
GIB = 2**30
SECRETS: list[str] = []


class DeployError(RuntimeError):
    pass


def log(message):
    for secret in SECRETS:
        if secret:
            message = str(message).replace(secret, "[REDACTED]")
    print(message, flush=True)


def run(args, *, capture=True, check=True):
    try:
        result = subprocess.run(args, check=False, text=True, encoding="utf-8",
                                errors="replace", capture_output=capture)
    except FileNotFoundError as exc:
        raise DeployError(f"找不到命令：{args[0]}") from exc
    if check and result.returncode:
        raise DeployError(f"命令失败：{args[0]} {args[1]}\n{(result.stderr or '')[-2500:]}")
    return result


def load_profile(path=None):
    data = json.loads(Path(path or ROOT / "profiles/ram-4gpu.json").read_text(encoding="utf-8"))
    required = set(json.loads((ROOT / "profiles/ram-4gpu.json").read_text(encoding="utf-8")))
    if set(data) != required or not all(isinstance(v, str) for v in data.values()):
        raise DeployError("配置必须保留默认 profile 的全部键，值必须是字符串。")
    if data["OFFLOAD_MODE"] != "ram" or data["DSV41_CACHE_GIB"] != "0":
        raise DeployError("本工具固定使用 RAM 全量 Engram，DSV41_CACHE_GIB=0。")
    if any(data[k] != "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise DeployError("推理容器必须离线读取 GPUStack 已下载的权重。")
    context = int(data["CONTEXT_LENGTH"])
    total = int(data["MAX_TOTAL_TOKENS"])
    chunk = int(data["CHUNKED_PREFILL_SIZE"])
    if not 204800 <= context <= 1048576 or not context <= total <= 16777216:
        raise DeployError("上下文必须在 204800..1048576，token 池必须 >= 上下文且 <=16777216。")
    if not 1 <= int(data["MAX_RUNNING_REQUESTS"]) <= 32:
        raise DeployError("并发必须在 1..32。")
    if not 128 <= chunk <= context or chunk % 128:
        raise DeployError("prefill chunk 必须是 128 的倍数且不超过上下文。")
    if not 0.5 <= float(data["MEMORY_FRACTION"]) <= 0.95:
        raise DeployError("显存比例必须在 0.5..0.95。")
    if data["ENABLE_DSPARK"] not in ("0", "1") or not 1 <= int(data["GPUSTACK_MODEL_RUNTIME_SHM_SIZE_GIB"]) <= 128:
        raise DeployError("DSPARK 必须为 0/1，SHM 必须在 1..128 GiB。")
    return data


def vendor_check():
    lock = json.loads((ROOT / "vendor-lock.json").read_text(encoding="utf-8"))
    for relative, expected in lock["files"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected["sha256"]:
            raise DeployError(f"上游文件校验失败：{relative}")
    return lock


def recipe_hash():
    lock = vendor_check()
    digest = hashlib.sha256()
    for name in sorted(["Dockerfile", ".dockerignore", "launch.py", "vendor-lock.json", *lock["files"]]):
        digest.update(name.encode() + b"\0" + (ROOT / name).read_bytes() + b"\0")
    return digest.hexdigest()


def default_base():
    return "m.daocloud.io/docker.io/" + vendor_check()["base_image"]


def inspect_image(image):
    result = run(["docker", "image", "inspect", image], check=False)
    return json.loads(result.stdout)[0] if result.returncode == 0 else None


def build_image(image, base, rebuild=False):
    digest = vendor_check()["base_image"].split("@", 1)[1]
    if not base.endswith("@" + digest) or "/" not in base:
        raise DeployError("BASE_IMAGE 必须是包含锁定 digest 的完整上游/镜像站地址。")
    expected = recipe_hash()
    existing = inspect_image(image)
    if existing and (existing.get("Config", {}).get("Labels") or {}).get("deployment.bundle.sha256") == expected and not rebuild:
        log(f"复用已校验镜像：{image}")
        return False
    if existing and not rebuild:
        raise DeployError(f"同名镜像 {image} 与本项目源码不一致；指定新 --image，或明确 --rebuild。")
    log(f"构建镜像：{image}\n基础镜像：{base}")
    run(["docker", "build", "--pull", "--network=none", "--build-arg", f"BASE_IMAGE={base}",
         "--build-arg", f"BUNDLE_SHA256={expected}", "-t", image, str(ROOT)], capture=False)
    built = inspect_image(image)
    if not built or (built.get("Config", {}).get("Labels") or {}).get("deployment.bundle.sha256") != expected:
        raise DeployError("构建后的镜像标识不匹配。")
    return True


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class API:
    def __init__(self, url, key=""):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise DeployError("地址必须是 http(s)://主机[:端口][/前缀]，不得包含凭据、查询或片段。")
        self.url, self.key = url.rstrip("/"), key
        # Worker 内网 API 不经过 shell 的 Clash/HTTP_PROXY；不转发凭据到重定向地址。
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, method, path, payload=None, *, timeout=30, raw=False):
        headers = {"Accept": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        body = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload).encode()
        req = urllib.request.Request(self.url + path, data=body, headers=headers, method=method)
        try:
            with self.opener.open(req, timeout=timeout) as response:
                content = response.read()
                return content if raw else (json.loads(content) if content else {})
        except urllib.error.HTTPError as exc:
            detail = exc.read(8192).decode("utf-8", "replace")
            raise DeployError(f"{method} {path}: HTTP {exc.code} {detail[:2000]}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DeployError(f"{method} {path}: {exc}") from exc

    def items(self, path):
        rows = []
        for page in range(1, 10001):
            data = self.request("GET", f"{path}{'&' if '?' in path else '?'}page={page}&perPage=100")
            batch = data.get("items")
            if not isinstance(batch, list):
                raise DeployError(f"{path} 未返回 items 列表；请核对 GPUStack API 版本。")
            rows.extend(batch)
            pagination = data.get("pagination") or {}
            total = pagination.get("total")
            pages = pagination.get("totalPage") or pagination.get("total_page")
            if not batch or (total is not None and len(rows) >= total) or (pages is not None and page >= pages):
                return rows
            if total is None and pages is None and len(batch) < 100:
                return rows
        raise DeployError("分页数量异常，已停止读取。")


def backend_payload(name, image, env, owner=None):
    result = {
        "backend_name": name + "-custom", "description": MARKER,
        "backend_source": "custom", "enabled": True,
        "default_version": VERSION, "health_check_path": "/health",
        "default_backend_param": [], "default_entrypoint": ENTRYPOINT,
        "default_run_command": COMMAND, "default_env": env,
        "version_configs": {VERSION: {"image_name": image, "custom_framework": "cuda"}},
    }
    if owner is not None:
        result["owner_principal_id"] = owner
    return result


def model_payload(name, worker, indexes, env):
    return {
        "name": name, "description": "4 x SM120 / RAM Engram / ModelScope",
        "meta": {"managed_by": MANAGER}, "source": "model_scope",
        "model_scope_model_id": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "replicas": 1, "categories": ["llm"], "placement_strategy": "spread",
        "cpu_offloading": True, "distributed_inference_across_workers": False,
        "distributable": False, "worker_selector": {},
        "gpu_selector": {"gpu_ids": [f'{worker["name"]}:cuda:{i}' for i in indexes], "gpus_per_replica": 4},
        "backend": name + "-custom", "backend_version": VERSION, "backend_parameters": [],
        "image_name": None, "run_command": None, "env": env,
        "restart_on_error": False, "cluster_id": worker["cluster_id"],
        "speculative_config": None, "extended_kv_cache": None,
    }


def yaml_text(data, level=0):
    """Emit the small JSON-compatible subset needed for GPUStack backend import."""
    lines = []
    for key, value in data.items():
        prefix = "  " * level + json.dumps(str(key), ensure_ascii=False) + ":"
        if isinstance(value, dict) and value:
            lines.extend([prefix, yaml_text(value, level + 1).rstrip("\n")])
        else:
            lines.append(prefix + " " + json.dumps(value, ensure_ascii=False))
    return "\n".join(lines) + "\n"


def export_config(directory, backend, model):
    directory.mkdir(parents=True, exist_ok=True)
    for filename, data in (("backend.json", backend), ("model.json", {**model, "enable_model_route": True})):
        (directory / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (directory / "backend.yaml").write_text(yaml_text(backend), encoding="utf-8")
    (directory / "model.env").write_text("".join(f"{k}={v}\n" for k, v in model["env"].items()), encoding="utf-8")


def parse_indexes(value):
    try:
        indexes = [int(x.strip()) for x in value.split(",")]
    except ValueError as exc:
        raise DeployError("GPU 编号格式：2,3,4,5") from exc
    if len(indexes) != 4 or len(set(indexes)) != 4 or min(indexes) < 0:
        raise DeployError("必须提供四个不同的非负 GPU 编号。")
    return sorted(indexes)


def local_gpus():
    result = run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used", "--format=csv,noheader,nounits"])
    return [{"index": int(row[0]), "uuid": row[1].strip(), "name": row[2].strip(),
             "total_mib": float(row[3]), "used_mib": float(row[4])}
            for row in csv.reader(io.StringIO(result.stdout)) if row]


def choose_worker(workers, local, requested):
    uuids = {g["uuid"] for g in local}
    candidates = [w for w in workers if (not requested or w["name"] == requested)
                  and any(g.get("uuid") in uuids for g in (w.get("status") or {}).get("gpu_devices") or [])]
    if len(candidates) != 1:
        raise DeployError("无法唯一匹配当前主机的 Worker；请用 --worker 指定名称，确认 UUID 上报正常且只有一个注册记录。")
    worker = candidates[0]
    if worker.get("state") != "ready" or worker.get("unreachable") or (worker.get("maintenance") or {}).get("enabled"):
        raise DeployError("目标 Worker 必须 ready、可达且不在维护模式。")
    if not worker.get("cluster_id"):
        raise DeployError("Worker 缺少 cluster_id；本工具面向 GPUStack 2.2.3 API。")
    return worker


def map_gpus(worker, local, indexes):
    local_by_index = {g["index"]: g for g in local}
    remote_by_uuid = {g["uuid"]: g for g in worker["status"]["gpu_devices"] if g.get("uuid")}
    selected = []
    for index in indexes:
        gpu = local_by_index.get(index)
        if not gpu or gpu["total_mib"] < 90 * 1024 or "RTX PRO 6000" not in gpu["name"] or "Blackwell" not in gpu["name"]:
            raise DeployError(f"本机 GPU {index} 不是所需的 RTX PRO 6000 Blackwell 96GB。")
        remote = remote_by_uuid.get(gpu["uuid"])
        if not remote or remote.get("type") != "cuda":
            raise DeployError(f"GPU {index} 的 UUID 未在目标 Worker 的 CUDA 设备中找到。")
        selected.append({**gpu, "api_index": remote["index"], "allocated": (remote.get("memory") or {}).get("allocated") or 0})
    return selected


def check_free(selected, instances, worker_id, owned_id=None):
    selected_indexes = {g["api_index"] for g in selected}
    for instance in instances:
        if instance.get("worker_id") == worker_id and instance.get("model_id") != owned_id:
            if selected_indexes.intersection(instance.get("gpu_indexes") or []):
                raise DeployError(f'所选卡已分配给其他实例：{instance.get("name")}。请选择其他四张卡。')
    pids = run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"]).stdout
    busy_uuids = {row[0].strip() for row in csv.reader(io.StringIO(pids)) if len(row) >= 2}
    for gpu in selected:
        # Re-query local memory to avoid relying on stale API utilization.
        if gpu["uuid"] in busy_uuids or gpu["used_mib"] > 1024 or (owned_id is None and gpu["allocated"]):
            raise DeployError(f'GPU {gpu["index"]} 有进程、显存占用或调度分配；请手动释放，工具不会终止其他业务。')


def host_preflight():
    if sys.platform != "linux":
        raise DeployError("deploy/build/verify 请在目标 Linux Docker Worker 上执行；render 可在 Windows 使用。")
    run(["docker", "info", "--format", "{{.ServerVersion}}"])
    # Prevent accidentally building on another Docker daemon while checking local GPUs.
    context = run(["docker", "context", "inspect"]).stdout
    host = os.environ.get("DOCKER_HOST") or json.loads(context)[0]["Endpoints"]["docker"]["Host"]
    if not host.startswith("unix://"):
        raise DeployError("请使用本机 Unix socket 的 Docker daemon，与 GPUStack Worker 共用镜像存储。")


def memory_preflight():
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(memory["MemAvailable"].split()[0]) * 1024
    if available < 224 * GIB:
        raise DeployError(f"RAM 模式预检要求当前可用内存 >=224 GiB，目前 {available/GIB:.1f} GiB。")
    log(f"当前可用主机内存：{available/GIB:.0f} GiB")


def find_owned(rows, field, value, kind, require_owned=True):
    matches = [row for row in rows if row.get(field) == value]
    if len(matches) > 1:
        raise DeployError(f"存在多个同名{kind}，请使用唯一 --name 或有明确组织范围的 API Key。")
    if not matches:
        return None
    row = matches[0]
    owned = ((row.get("meta") or {}).get("managed_by") == MANAGER if kind == "模型"
             else row.get("description") == MARKER and not row.get("is_built_in"))
    if require_owned and not owned:
        raise DeployError(f"同名{kind}不是本工具创建，拒绝覆盖；请更换 --name。")
    return row


def differs(existing, desired):
    if existing is None:
        return True
    def contains(actual, wanted):
        if isinstance(wanted, dict):
            return isinstance(actual, dict) and all(contains(actual.get(k), v) for k, v in wanted.items())
        return actual == wanted
    if "env" in desired and existing.get("env") != desired["env"]:
        return True
    return not contains(existing, desired)


def update_model(api, existing, desired):
    # Preserve tenancy/access while replacing only our managed model specification.
    payload = dict(desired)
    for key in ("owner_principal_id", "access_policy"):
        if key in existing:
            payload[key] = existing[key]
    return api.request("PUT", f'/v2/models/{existing["id"]}', payload)


def wait_stopped(api, model_id, timeout=300):
    deadline = time.monotonic() + timeout
    while api.items(f"/v2/models/{model_id}/instances"):
        if time.monotonic() > deadline:
            raise DeployError("旧实例停止超时；未启动新实例。请在 GPUStack 检查后重新运行。")
        time.sleep(5)


def wait_running(api, model_id, timeout):
    deadline, last_log = time.monotonic() + timeout, 0
    while time.monotonic() < deadline:
        instances = api.items(f"/v2/models/{model_id}/instances")
        for instance in instances:
            if instance.get("state") in ("error", "unreachable"):
                raise DeployError(f'实例 {instance["name"]}: {instance.get("state_message") or instance["state"]}')
        if len(instances) == 1 and instances[0].get("state") == "running":
            return instances[0]
        if time.monotonic() - last_log >= 30:
            log("等待模型下载/初始化/预热：" + "; ".join(
                f'{x.get("name")}: {x.get("state")} 下载 {x.get("download_progress")}%' for x in instances))
            last_log = time.monotonic()
        time.sleep(5)
    raise DeployError("等待启动超时。资源已保留；检查 GPUStack 实例日志，重跑可继续等待，--redeploy 会重建本工具的实例。")


def check_server_info(data, env):
    args = data.get("server_args", data)
    expected = {
        "context_length": int(env["CONTEXT_LENGTH"]), "tp_size": 4, "ep_size": 4,
        "max_running_requests": int(env["MAX_RUNNING_REQUESTS"]),
        "max_total_tokens": int(env["MAX_TOTAL_TOKENS"]),
        "mem_fraction_static": float(env["MEMORY_FRACTION"]),
        "chunked_prefill_size": int(env["CHUNKED_PREFILL_SIZE"]),
    }
    if env["ENABLE_DSPARK"] == "1":
        expected.update(speculative_algorithm="DSPARK", speculative_dspark_block_size=5)
    else:
        expected["speculative_algorithm"] = None
    mismatches = {k: {"expected": v, "actual": args.get(k)} for k, v in expected.items() if k not in args or args[k] != v}
    if mismatches:
        raise DeployError("运行参数未通过核验：" + json.dumps(mismatches, ensure_ascii=False))
    return expected


def verify_runtime(api, instance, model, worker, env, image, inference_key):
    if instance.get("worker_id") != worker["id"]:
        raise DeployError("实例运行在另一个 Worker，拒绝报告部署成功。")
    expected_indexes = {int(x.rsplit(":", 1)[1]) for x in model["gpu_selector"]["gpu_ids"]}
    if set(instance.get("gpu_indexes") or []) != expected_indexes:
        raise DeployError("实例实际 GPU 分配与指定的四张卡不一致。")
    ids = run(["docker", "ps", "-q", "--filter", "name=" + instance["name"]]).stdout.split()
    expected_image = inspect_image(image)
    if not expected_image:
        raise DeployError("本机镜像不存在，无法核验实际运行版本。")
    matches = []
    for cid in ids:
        info = json.loads(run(["docker", "inspect", cid]).stdout)[0]
        if info.get("Image") == expected_image["Id"]:
            matches.append(info)
    if len(matches) != 1:
        raise DeployError("本机 Docker 未唯一找到推理容器；确认 GPUStack 使用同一 daemon/镜像及 IfNotPresent 策略。")
    container = matches[0]
    actual_env = dict(item.split("=", 1) for item in container["Config"].get("Env", []) if "=" in item)
    for key, value in env.items():
        if actual_env.get(key) != value:
            raise DeployError(f"容器环境变量未生效：{key} 应为 {value}，实际 {actual_env.get(key)}")
    if container["HostConfig"].get("ShmSize", 0) < int(env["GPUSTACK_MODEL_RUNTIME_SHM_SIZE_GIB"]) * GIB:
        raise DeployError("容器 /dev/shm 小于配置要求。")
    labels = container["Config"].get("Labels") or {}
    if labels.get("deployment.bundle.sha256") != recipe_hash():
        raise DeployError("运行容器的镜像源码标识与当前仓库不一致。")
    host = worker["ip"]
    if ":" in host:
        host = f"[{host}]"
    backend = API(f'http://{host}:{instance["port"]}')
    deadline = time.monotonic() + 120
    while True:
        try:
            backend.request("GET", "/health", raw=True)
            break
        except DeployError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(5)
    effective = check_server_info(backend.request("GET", "/server_info"), env)
    inference = API(api.url, inference_key)
    response = inference.request("POST", "/v1/chat/completions", {
        "model": model["name"], "messages": [{"role": "user", "content": "19+23等于多少？仅输出数字。"}],
        "max_tokens": 64, "temperature": 0, "stream": False,
        "chat_template_kwargs": {"thinking": False},
    }, timeout=300)
    choices = response.get("choices") or []
    answer = (choices[0].get("message", {}).get("content") or "").strip() if choices else ""
    if not re.search(r"(?<!\d)42(?!\d)", answer):
        raise DeployError("推理请求未返回预期答案 42；请检查模型路由、推理权限和服务日志。")
    return {"status": "smoke_passed", "model": model["name"], "instance": instance["name"],
            "worker": worker["name"], "container": container["Id"][:12],
            "effective_args": effective, "env": env, "answer": answer,
            "validated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "validation_scope": "configuration + health + short generation; no long-context or throughput benchmark"}


def prompt_value(value, message):
    if value:
        return value
    if not sys.stdin.isatty():
        raise DeployError(f"非交互模式缺少参数：{message}")
    return input(message).strip()


def deploy(args):
    host_preflight()
    env = load_profile(args.profile)
    url = prompt_value(args.url or os.environ.get("GPUSTACK_URL"), "GPUStack 地址（例如 http://10.0.0.10:80）：")
    key = os.environ.get("GPUSTACK_API_KEY", "")
    if not key:
        if not sys.stdin.isatty():
            raise DeployError("请设置 GPUSTACK_API_KEY；不要把 Key 放进命令行参数。")
        key = getpass.getpass("GPUStack API Key（需要管理和推理权限，不保存）：").strip()
    if not key:
        raise DeployError("API Key 不能为空。")
    inference_key = os.environ.get("GPUSTACK_INFERENCE_API_KEY") or key
    SECRETS.extend([key, inference_key])
    api = API(url, key)
    local = local_gpus()
    worker = choose_worker(api.items("/v2/workers"), local, args.worker)
    log(f'目标 Worker：{worker["name"]} / cluster {worker["cluster_id"]}')
    for gpu in local:
        log(f'  GPU {gpu["index"]}: {gpu["name"]}, 已用 {gpu["used_mib"]:.0f} MiB, {gpu["uuid"]}')
    indexes = parse_indexes(prompt_value(args.gpus, "指定本机四张空闲 GPU 编号（例如 2,3,4,5）："))
    selected = map_gpus(worker, local, indexes)
    desired_backend = backend_payload(args.name, args.image, env, worker.get("owner_principal_id"))
    desired_model = model_payload(args.name, worker, [g["api_index"] for g in selected], env)
    backend = find_owned(api.items("/v2/inference-backends"), "backend_name", desired_backend["backend_name"], "后端", args.action != "verify")
    model = find_owned(api.items("/v2/models"), "name", args.name, "模型", args.action != "verify")
    if model and model.get("cluster_id") != worker["cluster_id"]:
        raise DeployError("同名受管模型属于另一集群，请使用新 --name。")
    output = ROOT / ".state" / args.name
    export_config(output, desired_backend, desired_model)
    log(f"部署配置已生成：{output}")
    changes = differs(backend, desired_backend) or differs(model, desired_model)
    if args.plan:
        log(json.dumps({"worker": worker["name"], "gpu_mapping": selected,
                        "will_update": changes, "will_redeploy": bool(model and (changes or args.redeploy or args.rebuild))}, ensure_ascii=False, indent=2))
        log("预览结束：未构建镜像、未写 GPUStack。")
        return
    # A failed new attempt must not leave a previous success looking current.
    (output / "acceptance.json").unlink(missing_ok=True)
    if args.action == "verify":
        if not model:
            raise DeployError("未找到指定名称的模型。")
        for key in ("source", "model_scope_model_id", "env", "backend", "backend_version", "backend_parameters"):
            actual = (model.get(key) or []) if key == "backend_parameters" else model.get(key)
            if actual != desired_model[key]:
                raise DeployError(f"GPUStack 模型配置 {key} 与预期不一致。")
    else:
        if not model:
            check_free(selected, api.items("/v2/model-instances"), worker["id"])
            memory_preflight()
        rebuilt = build_image(args.image, args.base_image or default_base(), args.rebuild)
        recreate = bool(model and (changes or args.redeploy or rebuilt))
        if recreate:
            log("停止本工具管理的旧实例，等待释放后使用新配置创建。")
            update_model(api, model, {**desired_model, "replicas": 0})
            wait_stopped(api, model["id"])
        if not model or recreate:
            # Repeat checks after a potentially long build; never take a busy card.
            current_worker = next(w for w in api.items("/v2/workers") if w["id"] == worker["id"])
            selected = map_gpus(current_worker, local_gpus(), indexes)
            check_free(selected, api.items("/v2/model-instances"), worker["id"], model["id"] if model else None)
            memory_preflight()
        if differs(backend, desired_backend):
            path = f'/v2/inference-backends/{backend["id"]}' if backend else "/v2/inference-backends"
            api.request("PUT" if backend else "POST", path, desired_backend)
        if not model:
            model = api.request("POST", "/v2/models", {**desired_model, "enable_model_route": True})
        elif recreate:
            model = update_model(api, model, desired_model)
        else:
            log("配置相同，保留现有实例并继续验收；--redeploy 可主动重建。")
    instance = wait_running(api, model["id"], args.timeout)
    report = verify_runtime(api, instance, desired_model, worker, env, args.image, inference_key)
    (output / "acceptance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f'部署验收通过：{args.name}\nOpenAI API：{api.url}/v1\n报告：{output / "acceptance.json"}\n已核对参数并完成短问答；长上下文与吞吐量需按实际请求另测。')


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("deploy", "verify", "build", "render"), nargs="?", default="deploy")
    parser.add_argument("--url", help="GPUStack 服务根地址，不含 /v2 或 /v1")
    parser.add_argument("--worker", help="Worker 精确名称；deploy 默认按 UUID 匹配")
    parser.add_argument("--gpus", help="四张本机 nvidia-smi GPU 编号；render 时使用 GPUStack 编号")
    parser.add_argument("--cluster-id", type=int, help="render 时必填")
    parser.add_argument("--name", default=MODEL)
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--profile", help="自定义 JSON profile，默认 profiles/ram-4gpu.json")
    parser.add_argument("--base-image", help="完整镜像地址及锁定 digest；默认 DaoCloud")
    parser.add_argument("--rebuild", action="store_true", help="重建本地推理镜像并重建受管实例")
    parser.add_argument("--redeploy", action="store_true", help="重建受管模型实例，不影响其他模型")
    parser.add_argument("--plan", action="store_true", help="只读 API 并导出部署计划，不构建/部署")
    parser.add_argument("--timeout", type=int, default=21600, help="等待实例 running 的秒数，默认 6 小时")
    parser.add_argument("--output", type=Path, default=ROOT / "generated", help="render 输出目录")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,79}", args.name):
        parser.error("--name 仅允许小写字母、数字、点和连字符，最长 80 字符。")
    if args.timeout < 1:
        parser.error("--timeout 必须大于 0。")
    if args.action == "verify" and (args.rebuild or args.redeploy):
        parser.error("verify 不接受 --rebuild/--redeploy。")
    try:
        if args.action == "render":
            if not args.worker or not args.gpus or not args.cluster_id:
                parser.error("render 需要 --worker、--gpus、--cluster-id。")
            env = load_profile(args.profile)
            worker = {"name": args.worker, "cluster_id": args.cluster_id}
            export_config(args.output, backend_payload(args.name, args.image, env),
                          model_payload(args.name, worker, parse_indexes(args.gpus), env))
            log(f"配置已生成：{args.output.resolve()}")
        elif args.action == "build":
            host_preflight()
            build_image(args.image, args.base_image or default_base(), args.rebuild)
        else:
            deploy(args)
    except (DeployError, ValueError, KeyError, OSError) as exc:
        log(f"失败：{exc}")
        return 1
    except KeyboardInterrupt:
        log("已中断。已创建的模型/下载缓存保留；相同命令可继续。")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
