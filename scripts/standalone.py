#!/usr/bin/env python3
"""Direct Docker deployment. No GPUStack server, worker or token is required."""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import shlex
import sys
import time

import deploy as core

ROOT = Path(__file__).resolve().parents[1]
OWNER = "deepseek-v41-1017-standalone-v1"
OWNER_LABEL = "deployment.standalone.manager"
CONFIG_LABEL = "deployment.standalone.config"


def inspect_container(name):
    result = core.run(["docker", "container", "inspect", name], check=False)
    if result.returncode:
        if "No such" in result.stderr:
            return None
        raise core.DeployError(f"无法读取容器 {name}: {result.stderr[-1000:]}")
    return json.loads(result.stdout)[0]


def require_owned(container):
    if container and (container.get("Config", {}).get("Labels") or {}).get(OWNER_LABEL) != OWNER:
        raise core.DeployError("同名容器不属于本工具，拒绝修改；请更换 --name。")


def selected_gpus(indexes):
    local = {g["index"]: g for g in core.local_gpus()}
    selected = []
    for index in indexes:
        gpu = local.get(index)
        if not gpu or gpu["total_mib"] < 90 * 1024 or "RTX PRO 6000" not in gpu["name"] or "Blackwell" not in gpu["name"]:
            raise core.DeployError(f"GPU {index} 必须为 RTX PRO 6000 Blackwell 96GB。")
        selected.append({**gpu, "api_index": index, "allocated": 0})
    return selected


def validate_checkpoint(path):
    # Reuse the same local-file validation as the image without importing torch.
    sys.path.insert(0, str(ROOT))
    from launch import check_checkpoint
    try:
        check_checkpoint(path)
    except (ValueError, KeyError, OSError) as exc:
        raise core.DeployError(f"权重目录不完整或不匹配：{exc}；先从 ModelScope 下载完整模型。") from exc


def launch_args(model):
    return ["--model-path", "/model", "--host", "0.0.0.0", "--port", "30000", "--served-model-name", model]


def desired_spec(args, env, selected):
    return {"model_dir": str(args.model_dir), "cache_dir": str(ROOT / ".state/docker" / args.name / "cache"),
            "gpus": [g["uuid"] for g in selected], "env": env,
            "image": args.image, "recipe": core.recipe_hash(),
            "bind": args.bind, "port": args.port, "model": args.served_model_name}


def config_hash(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def docker_command(name, spec, *, preflight=False):
    command = ["docker", "run", "--pull=never"]
    if preflight:
        command += ["--rm"]
    else:
        command += ["-d", "--name", name, "--restart=no",
                    "--label", f"{OWNER_LABEL}={OWNER}",
                    "--label", f"{CONFIG_LABEL}={config_hash(spec)}",
                    "--publish", f'{spec["bind"]}:{spec["port"]}:30000']
    # Docker's --gpus option parses CSV: the inner quotes are part of this argv value.
    command += ["--gpus", '"device=' + ",".join(spec["gpus"]) + '"',
                "--shm-size", spec["env"]["GPUSTACK_MODEL_RUNTIME_SHM_SIZE_GIB"] + "g",
                "--ulimit", "memlock=-1:-1", "--cap-add", "IPC_LOCK",
                "--mount", f'type=bind,source={spec["model_dir"]},target=/model,readonly',
                "--mount", f'type=bind,source={spec["cache_dir"]},target=/root/.cache']
    for key, value in spec["env"].items():
        command += ["--env", f"{key}={value}"]
    command += [spec["image"], *launch_args(spec["model"])]
    if preflight:
        command += ["--preflight"]
    return command


def validate_container(container, spec, image_id):
    require_owned(container)
    if container["Image"] != image_id:
        raise core.DeployError("实际容器使用的镜像与所选本地镜像不同。")
    config = container["Config"]
    if (config.get("Labels") or {}).get(CONFIG_LABEL) != config_hash(spec):
        raise core.DeployError("容器配置标识不一致；使用相同参数，或 --recreate 重建。")
    actual_env = dict(item.split("=", 1) for item in config.get("Env", []) if "=" in item)
    if any(actual_env.get(k) != v for k, v in spec["env"].items()):
        raise core.DeployError("容器实际环境变量与 profile 不一致。")
    if config.get("Cmd") != launch_args(spec["model"]):
        raise core.DeployError("容器启动参数不一致。")
    host = container["HostConfig"]
    devices = [device for request in (host.get("DeviceRequests") or []) for device in request.get("DeviceIDs") or []]
    if sorted(devices) != sorted(spec["gpus"]):
        raise core.DeployError("容器实际 GPU UUID 与指定四卡不一致。")
    if host.get("ShmSize", 0) < int(spec["env"]["GPUSTACK_MODEL_RUNTIME_SHM_SIZE_GIB"]) * core.GIB:
        raise core.DeployError("容器共享内存小于要求。")
    bindings = (host.get("PortBindings") or {}).get("30000/tcp")
    if bindings != [{"HostIp": spec["bind"], "HostPort": str(spec["port"])}]:
        raise core.DeployError("容器实际端口映射不一致。")
    models = [m for m in container.get("Mounts", []) if m.get("Destination") == "/model"]
    if len(models) != 1 or models[0].get("Source") != spec["model_dir"] or models[0].get("RW"):
        raise core.DeployError("模型目录没有按要求只读挂载。")


def wait_and_verify(args, spec, image_id):
    # Wildcard bind is reached via loopback on the deploying host.
    host = "127.0.0.1" if spec["bind"] == "0.0.0.0" else spec["bind"]
    api = core.API(f'http://{host}:{spec["port"]}')
    deadline, next_log = time.monotonic() + args.timeout, 0
    while True:
        container = inspect_container(args.name)
        if not container or not container.get("State", {}).get("Running"):
            raise core.DeployError(f"容器未运行或已退出。请执行 docker logs --tail 200 {args.name} 查看原因。")
        validate_container(container, spec, image_id)
        try:
            api.request("GET", "/health", timeout=5, raw=True)
            break
        except core.DeployError:
            if time.monotonic() >= deadline:
                raise core.DeployError("初始化/健康检查超时；保留容器和日志，重跑可继续验收。")
            if time.monotonic() >= next_log:
                core.log(f"等待权重加载和预热，日志：docker logs -f {args.name}")
                next_log = time.monotonic() + 30
            time.sleep(5)
    effective = core.check_server_info(api.request("GET", "/server_info"), spec["env"])
    response = api.request("POST", "/v1/chat/completions", {
        "model": spec["model"], "messages": [{"role": "user", "content": "19+23等于多少？仅输出数字。"}],
        "max_tokens": 64, "temperature": 0, "stream": False,
        "chat_template_kwargs": {"thinking": False},
    }, timeout=300)
    choices = response.get("choices") or []
    answer = (choices[0].get("message", {}).get("content") or "").strip() if choices else ""
    if not re.search(r"(?<!\d)42(?!\d)", answer):
        raise core.DeployError("短问答未返回预期答案 42，未通过验收。")
    return {"status": "smoke_passed", "container": args.name, "base_url": api.url + "/v1",
            "model": spec["model"], "effective_args": effective, "answer": answer,
            "validated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "validation_scope": "configuration + health + short generation; no throughput/long-context benchmark"}


def deploy(args):
    core.host_preflight()
    env = core.load_profile(args.profile)
    indexes = core.parse_indexes(args.gpus)
    selected = selected_gpus(indexes)
    spec = desired_spec(args, env, selected)
    existing = inspect_container(args.name)
    require_owned(existing)
    if args.plan:
        core.log("预览：" + shlex.join(docker_command(args.name, spec)))
        core.log("未下载、构建或修改 Docker 容器。")
        return
    state = ROOT / ".state/docker" / args.name
    state.mkdir(parents=True, exist_ok=True)
    (state / "acceptance.json").unlink(missing_ok=True)
    if args.verify_only:
        image = core.inspect_image(args.image)
        if not existing or not image:
            raise core.DeployError("验收需要现有容器和本地镜像。")
    else:
        if not existing:
            core.check_free(selected, [], None)
            core.memory_preflight()
        if args.download:
            if existing:
                raise core.DeployError("已有容器时不覆盖下载挂载中的权重；复用现有权重请去掉 --download。")
            core.run(["bash", str(ROOT / "download-model.sh"), str(args.model_dir), args.revision], capture=False)
        validate_checkpoint(args.model_dir)
        core.build_image(args.image, args.base_image or core.default_base(), args.rebuild)
        image = core.inspect_image(args.image)
        if not image:
            raise core.DeployError("构建后未找到本地镜像。")
        if existing:
            unchanged = (existing["Image"] == image["Id"] and
                         (existing["Config"].get("Labels") or {}).get(CONFIG_LABEL) == config_hash(spec))
            if not args.recreate:
                if not unchanged or not existing.get("State", {}).get("Running"):
                    raise core.DeployError("现有容器已停止或配置有变化；使用 --recreate 重建此容器。")
                core.log("复用现有容器，继续验收。")
            else:
                core.run(["docker", "stop", "--time", "60", args.name])
                core.run(["docker", "rm", args.name])
                existing = None
        if not existing:
            core.check_free(selected_gpus(indexes), [], None)
            core.memory_preflight()
            Path(spec["cache_dir"]).mkdir(parents=True, exist_ok=True)
            core.run(docker_command(args.name, spec, preflight=True), capture=False)
            core.run(docker_command(args.name, spec), capture=False)
    if (image.get("Config", {}).get("Labels") or {}).get("deployment.bundle.sha256") != core.recipe_hash():
        raise core.DeployError("本地镜像源码标识不一致，请使用 --rebuild --recreate。")
    report = wait_and_verify(args, spec, image["Id"])
    (state / "acceptance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    core.log(f'验收通过：{report["base_url"]}，模型 {spec["model"]}\n报告：{state / "acceptance.json"}\n本次仅验证参数、健康状态和短问答。')


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path, help="完整本地 checkpoint 目录")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--name", default="dsv41-1017-docker")
    parser.add_argument("--served-model-name", default=core.MODEL)
    parser.add_argument("--image", default=core.IMAGE)
    parser.add_argument("--profile")
    parser.add_argument("--bind", default="127.0.0.1", help="宿主机 IPv4 监听地址，默认仅本机")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--download", action="store_true", help="先通过独立 venv 从 ModelScope 下载完整权重")
    parser.add_argument("--revision", default="master", help="ModelScope 权重版本，仅用于 --download")
    parser.add_argument("--base-image", help="基础镜像完整地址和锁定 digest")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--recreate", action="store_true", help="停止并删除同名受管容器，重新创建；保留权重和缓存")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,79}", args.name):
        parser.error("--name 仅允许小写字母、数字、点和连字符，最长 80 字符。")
    if not 1 <= args.port <= 65535 or args.timeout < 1:
        parser.error("端口必须在 1..65535，超时必须 >0。")
    if args.verify_only and (args.download or args.rebuild or args.recreate):
        parser.error("--verify-only 不允许下载、构建或重建。")
    try:
        ipaddress.IPv4Address(args.bind)
        args.model_dir = args.model_dir.expanduser().resolve()
        if args.model_dir == Path(args.model_dir.anchor) or any(c in str(args.model_dir) + str(ROOT) for c in (",", "\n", "\r")):
            raise core.DeployError("项目/模型路径不能含逗号或换行，模型不能使用根目录。")
        deploy(args)
    except (core.DeployError, ValueError, KeyError, OSError) as exc:
        core.log(f"失败：{exc}")
        return 1
    except KeyboardInterrupt:
        core.log("已中断；容器、已下载权重和缓存保留。")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
