# DeepSeek V4.1 Flash 1017 · 四卡 RTX PRO 6000 · 独立 Docker 部署包

无需安装 GPUStack，直接在 Linux 服务器上启动兼容 OpenAI API 的推理容器。默认 GPU `0,1,2,3`，复用四卡 RAM Engram、DSPARK block 5、524288 tokens 上下文、并发 8 配置。

本包包含 Dockerfile、锁定的 SM120/Engram 适配代码、ModelScope 下载脚本、启动及验收脚本和参数配置；**不包含模型权重、Docker 镜像层或 Python 下载依赖**，首次使用需联网。代码和启动流程已做离线测试，目标 GPU 上的加载、长上下文和吞吐仍需现场验收。

## 最低要求与推荐配置

| 项目 | 最低要求 / 启动检查条件 | 推荐配置 |
| --- | --- | --- |
| GPU | 4 × RTX PRO 6000 Blackwell 96GB，SM120；每卡 CUDA 可见显存至少 90GiB，所选卡空闲 | 四卡专用于此实例，PCIe 拓扑和散热供电满足持续负载 |
| 主机内存 | 启动前 `MemAvailable ≥224GiB`；容器还会检查实际 Engram 分片大小 +24GiB | 物理内存 512GiB 及以上，为权重加载和系统保留余量；较大并发可规划 1TiB |
| 模型盘 | 本地 NVMe，完整模型目录；容量规划至少预留 700GB | 1TB 及以上可用空间，Docker 数据目录另留镜像及构建临时层空间 |
| 系统 | Linux x86_64、Bash、Python 3.10+、Docker、NVIDIA Container Toolkit、可工作的 `nvidia-smi` | 使用维护中的系统/驱动版本，驱动须兼容锁定基础镜像中的 CUDA |
| 网络 | 可访问 ModelScope，以及 DaoCloud 镜像站或经 Docker 代理访问 Docker Hub | 缓存权重和镜像，避免重复下载 |

这些是软件检查门槛及容量规划建议，不是已实测的最低硬件承诺。加载峰值、可用显存和请求形态仍会影响能否启动及实际吞吐；32GiB `/dev/shm` 来自主机内存，不是额外的显卡内存。

## 快速开始

解压部署包后进入项目目录。已有完整 ModelScope 权重时，直接指定其**实际目录**：

```bash
bash deploy-docker.sh \
  --model-dir /srv/models/DeepSeek-V4.1-Flash-1017 \
  --gpus 0,1,2,3
```

没有权重时，增加 `--download` 即可先下载再构建、部署和验收：

```bash
bash deploy-docker.sh \
  --model-dir /srv/models/DeepSeek-V4.1-Flash-1017 \
  --gpus 0,1,2,3 --download
```

模型目录及仓库目录应为当前用户可写的专用路径，示例 `/srv/models/...` 可替换成已准备的 NVMe 路径。需要对本机 Docker socket 的访问权限。镜像默认经 DaoCloud 获取；不安装或重启 Docker、不终止其他业务、不修改系统 Python。

默认容器名 `dsv41-1017-docker`，API 地址 `http://127.0.0.1:30000/v1`，服务模型名 `deepseek-v4.1-flash-1017-cn`。`--port` 可改宿主机端口，`--served-model-name` 可改 API 中的模型名。推理容器本身不设置 API Key，默认只映射宿主机回环地址；`--bind 0.0.0.0` 可开放到其他接口，仅应用于可信内网或已有鉴权的反向代理。

## 分步部署

### 1. 从 ModelScope 下载完整权重

```bash
bash download-model.sh /srv/models/DeepSeek-V4.1-Flash-1017
```

下载器使用独立 `.venv-modelscope`，安装固定版本 `modelscope-hub==0.4.0`。pip 默认使用清华镜像；已有 `PIP_INDEX_URL` 时使用该地址。下载入口固定 `https://modelscope.cn`，不回退 Hugging Face。默认下载整个仓库，包含 tokenizer、索引、分片和 DSPARK 权重。

中断后重跑相同命令，由 ModelScope 客户端复用下载结果。需要固定权重 revision 时，用 `bash download-model.sh /srv/models/DeepSeek-V4.1-Flash-1017 COMMIT_OR_TAG`，或给一键命令传 `--revision COMMIT_OR_TAG`。私有/受限仓库的登录应通过下载器的交互式 `ms-hub login` 完成，不把令牌写入仓库或命令行。

也可复用已经下载完成的模型目录。该目录应直接包含 `config.json`、`tokenizer.json` 和 `model.safetensors.index.json`；脚本会检查索引中的分片存在且不越出挂载目录。已有容器时应直接复用权重，不再加 `--download` 覆盖正在使用的文件。

### 2. 构建镜像

```bash
python3 scripts/deploy.py build
```

默认基础镜像：

```text
m.daocloud.io/docker.io/lmsysorg/sglang@sha256:c4ca651192e57e91989b5176c3665148131b9a171e53861dee87f5e57cef25b5
```

如果 Docker daemon 已配置 Clash HTTP/mixed 代理，可改用同一 digest 的 Docker Hub 地址：

```bash
python3 scripts/deploy.py build \
  --base-image docker.io/lmsysorg/sglang@sha256:c4ca651192e57e91989b5176c3665148131b9a171e53861dee87f5e57cef25b5
```

`deploy-docker.sh` 也接受同样的 `--base-image`。终端里的 `HTTP_PROXY`/`HTTPS_PROXY` 不等同于 Docker daemon 代理。默认 DaoCloud 方案无需修改系统代理；需要新配 Docker 代理时参照 Docker 官方文档，在维护窗口操作，避免影响已有容器。构建阶段没有 pip/apt 安装，`RUN` 禁止联网；拉取镜像层仍然需要网络。

### 3. 启动或预览

```bash
# 打印实际 docker run 命令，不下载、构建或启动容器
bash deploy-docker.sh --model-dir /srv/models/DeepSeek-V4.1-Flash-1017 --gpus 0,1,2,3 --plan

# 启动；已存在且配置相同的容器会被复用
bash deploy-docker.sh --model-dir /srv/models/DeepSeek-V4.1-Flash-1017 --gpus 0,1,2,3
```

启动参数包括：GPU UUID 精确选择、`--shm-size 32g`、`--ulimit memlock=-1:-1`、`--cap-add IPC_LOCK`。模型只读挂载为 `/model`；编译缓存放在 `.state/docker/<容器名>/cache/`。无需 `--privileged`，也无需手工追加 TP/EP 参数。默认关闭错误自动重启，初始化失败时保留日志。

`profiles/ram-4gpu.json` 是唯一参数来源：

| 配置 | 默认值 |
| --- | --- |
| `OFFLOAD_MODE` | `ram` |
| `DSV41_CACHE_GIB` | `0`；RAM 模式使用全量 Engram，此字段不控制 RAM 占用 |
| `CONTEXT_LENGTH` | `524288`，单请求输入与输出的总预算 |
| `MAX_RUNNING_REQUESTS` | `8` |
| `MAX_TOTAL_TOKENS` | `4200000`，多请求共享池的配置上限 |
| `MEMORY_FRACTION` | `0.95` |
| `CHUNKED_PREFILL_SIZE` | `2048` |
| `ENABLE_DSPARK` | `1`，使用 checkpoint 自带的 DSPARK，block size 5 |
| `GPUSTACK_MODEL_RUNTIME_SHM_SIZE_GIB` | `32`；独立脚本将该兼容字段转换为 Docker `--shm-size` |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | `1` / `1`；推理阶段只读本地权重 |

配置 512K 不代表八路请求都能同时跑满该长度。若资源不足，可复制 profile，保留 524288 上下文，将并发降为 `4`、共享 token 池降为 `1048576`、显存比例降为 `0.90`，然后使用 `--profile /path/to/local-profile.json --recreate`。这些同样是待现场验收的候选值。

### 4. 验收与 API 调用

启动脚本会先在临时容器内检查权重、主机内存、锁页条件和四张 SM120 GPU，再启动服务。只有容器参数、`/health`、`/server_info` 和短问答全部通过才写入 `.state/docker/<容器名>/acceptance.json`。默认等待两小时，可用 `--timeout 14400` 延长。

```bash
# 对已有容器只读验收
bash deploy-docker.sh --model-dir /srv/models/DeepSeek-V4.1-Flash-1017 --gpus 0,1,2,3 --verify-only

curl --fail http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash-1017-cn","messages":[{"role":"user","content":"你好"}],"max_tokens":128,"chat_template_kwargs":{"thinking":false}}'
```

验收只覆盖配置和短问答。200K+ 输入、512K 上下文边界及吞吐需用实际业务负载另行测试，输入长度还应计入 chat template、工具定义和输出预留。

## 日常操作

```bash
docker logs -f --tail 100 dsv41-1017-docker
docker stop dsv41-1017-docker
docker start dsv41-1017-docker

# 修改 profile、目录、端口等参数后，重建本工具管理的容器
bash deploy-docker.sh --model-dir /srv/models/DeepSeek-V4.1-Flash-1017 --gpus 0,1,2,3 --recreate

# 镜像源码变化后同时重建镜像和容器
bash deploy-docker.sh --model-dir /srv/models/DeepSeek-V4.1-Flash-1017 --gpus 0,1,2,3 --rebuild --recreate
```

`--recreate` 只停止并移除同名、带本工具管理标识的容器，保留模型和编译缓存。无关同名容器会被拒绝覆盖。服务容器失败后保留容器及日志；临时预检容器自动删除，其错误输出显示在终端。不执行全局 prune，不更改其他服务。

## 来源与许可

下载命令依据 [ModelScope Hub 官方 CLI](https://github.com/modelscope/modelscope_hub#download)。容器选项依据 [Docker GPU 文档](https://docs.docker.com/engine/containers/gpu/)、[Docker daemon 代理文档](https://docs.docker.com/engine/daemon/proxy/)。适配代码来源、固定 commit 和许可记录于 `vendor-lock.json`、`PROVENANCE.txt` 和 `licenses/`；包内 SHA256 清单可用 `sha256sum -c SHA256SUMS.txt` 校验。
