# DeepSeek V4.1 Flash 1017 · 4 × RTX PRO 6000 · GPUStack 国内部署

仓库名：`deepseek-v41-1017-4gpu-rtx-pro-6000-gpustack-cn`

GitHub description：面向 4 张 RTX PRO 6000 Blackwell 96GB 的 DeepSeek V4.1 Flash 1017 GPUStack 部署工具，提供一键部署与半手动教程，适配 ModelScope、国内镜像及 Clash 代理，包含 RAM Engram、DSPARK 和 512K 上下文配置。

在已有 GPUStack Docker Worker 上，使用 **4 张 RTX PRO 6000 Blackwell 96GB + RAM Engram + DSPARK** 部署官方 `deepseek-ai/DeepSeek-V4.1-Flash-1017`。模型由 GPUStack 从 **ModelScope** 下载，推理镜像从 **DaoCloud** 拉取基础层，也可使用已配置的 Clash Docker 代理。

提供两条路径：[一键部署](#一键部署)和[半手动部署](#半手动部署)。两者共用 `profiles/ram-4gpu.json`，默认配置为 **524288 tokens 上下文、并发 8、DSPARK block 5**。上下文是单请求的输入与输出总长度；`MAX_TOTAL_TOKENS=4200000` 是服务端共享 token 池上限，不是单请求上下文，也不保证八路请求都能跑满 512K。

**验证范围：**工具的配置生成、API 调用流程、重复执行、GPU 占用检查、参数核验已做离线测试。本仓库未在目标四卡服务器上完成端到端或性能测试；默认值是待现场验收的性能配置，不宣称“吞吐量最大”或“512K 长请求已实测”。工具只有通过实际容器检查、健康检查及短问答后才报告成功。

## 前提

- 在目标 **Linux x86_64 Worker 宿主机**运行；已有可用的 Docker、NVIDIA 驱动/Container Toolkit、`nvidia-smi`、Python **3.10+**、Bash。部署工具只用 Python 标准库，无需 pip。
- GPUStack 服务和 Worker 已注册、可联网下载 ModelScope 模型，Worker 状态为 Ready。本项目按 **GPUStack 2.2.3 的 `/v2` 管理 API**实现；其他版本需要核对 API/自定义后端行为。
- 同机至少四张空闲的 RTX PRO 6000 Blackwell 96GB。8 卡机器只指定其中四卡，其他卡可继续运行原有业务。工具按 UUID 对齐本机与 GPUStack 编号。
- RAM 使用全量 Engram 映射和锁页；2TB 主机适用，预检要求**当前可用内存至少 224 GiB**，加载时还会按实际权重大小复核。NVMe 上的 GPUStack 模型缓存建议至少预留 **700GB**，Docker 数据目录还须容纳基础镜像和构建临时层。
- 脚本和 GPUStack Worker 使用**同一 Docker daemon / 镜像存储**，Worker 允许使用本地镜像（通常为 `IfNotPresent`）。本项目不推送镜像到远程仓库。
- GPUStack API Key 需要管理与推理权限，推荐使用目标组织的 Key；若权限分离，另设 `GPUSTACK_INFERENCE_API_KEY`。Worker 的推理端口应只在可信内网可达，对外使用 GPUStack 的鉴权入口。

## 一键部署

将仓库复制/解压到目标 Worker，进入目录后执行：

```bash
bash deploy.sh
```

按提示填写 GPUStack **服务根地址**、API Key 和**本机**四张空闲卡编号。例如 `0,1,2,3` 仅为示例，请以本机 `nvidia-smi` 为准。API Key 隐藏输入且不保存。

工具自动完成：校验上游文件 → 匹配 Worker/GPU UUID → 校验四卡空闲 → 构建镜像 → 创建独立自定义后端 → 创建 ModelScope 模型和 API 路由 → 等待下载及初始化 → 核对真实容器环境变量和 `/server_info` → 执行短问答。

也可以预设参数，Key 仍使用隐藏输入：

```bash
bash deploy.sh --url http://10.0.0.10:80 --worker YOUR_WORKER --gpus 0,1,2,3
```

非交互执行时，用当前 shell 的 `GPUSTACK_URL`、`GPUSTACK_API_KEY` 环境变量提供地址和密钥；不要将 Key 放进脚本或 Git。

```bash
# 只读检查、生成计划，不构建镜像、不写 GPUStack
bash deploy.sh --url http://10.0.0.10:80 --worker YOUR_WORKER --gpus 0,1,2,3 --plan

# 修改 profile 后，重建本工具创建的模型实例
bash deploy.sh --url http://10.0.0.10:80 --worker YOUR_WORKER --gpus 0,1,2,3 --redeploy
```

默认模型名 `deepseek-v4.1-flash-1017-cn`，后端名 `deepseek-v4.1-flash-1017-cn-custom`，镜像 `dsv41-1017-gpustack-cn:1.0.0`。可用 `--name` 和 `--image` 修改。

重复运行会继续等待或验收配置相同的实例；配置改变时，先将**本工具管理的模型**缩容为 0，等旧实例退出再启动新实例。不会接管同名的其他部署、停止其他业务、重启 Docker 或清理缓存。构建源码发生变化时，使用 `--rebuild` 重建镜像。下载/初始化默认等待 6 小时，可通过 `--timeout 43200` 延长；失败后保留模型和已下载缓存供检查。

配置和验收结果存放于 `.state/<模型名>/`，不含 Key：

- `backend.yaml` / `backend.json`：实际后端配置。
- `model.json` / `model.env`：实际模型参数和环境变量。
- `acceptance.json`：仅在当前验收成功后写入，记录容器、有效参数、短问答和时间。

## 半手动部署

这条路径使用 GPUStack 页面下载模型、创建后端和部署模型，适合逐步操作。所有命令仍在目标 Docker Worker 的仓库目录内执行。

### 1. 构建推理容器镜像

默认走 DaoCloud，构建脚本会校验锁定的上游文件和镜像标识：

```bash
python3 scripts/deploy.py build
docker image inspect dsv41-1017-gpustack-cn:1.0.0 --format '{{.Id}}'
```

基础镜像固定为下列 digest，构建的 `RUN` 阶段不访问网络；拉取及导出镜像层仍需要网络：

```text
m.daocloud.io/docker.io/lmsysorg/sglang@sha256:c4ca651192e57e91989b5176c3665148131b9a171e53861dee87f5e57cef25b5
```

**已有 Clash Docker 代理时**，可直接从 Docker Hub 拉取同一个 digest：

```bash
python3 scripts/deploy.py build \
  --base-image docker.io/lmsysorg/sglang@sha256:c4ca651192e57e91989b5176c3665148131b9a171e53861dee87f5e57cef25b5
```

一键模式同样支持 `bash deploy.sh --base-image ...`。仅在终端 `export HTTPS_PROXY=...` 通常不能替 Docker daemon 的拉取配置代理。若需要新配 Clash，可手动创建 Docker systemd drop-in（主机上的 Clash HTTP/mixed 端口示例为 7890）：

```ini
# /etc/systemd/system/docker.service.d/http-proxy.conf
[Service]
Environment="HTTP_PROXY=http://127.0.0.1:7890"
Environment="HTTPS_PROXY=http://127.0.0.1:7890"
Environment="NO_PROXY=localhost,127.0.0.1,::1,10.0.0.10,YOUR_WORKER_IP"
```

填入实际内网服务地址后，维护窗口内执行 `sudo systemctl daemon-reload`、`sudo systemctl restart docker`。**重启 Docker 可能影响本机其他容器**，8 卡机器已有业务时优先使用默认 DaoCloud 路径或已生效的代理；部署工具不执行这些系统操作。DaoCloud 是镜像拉取服务，第三方服务的可用性与限速以其当时状态为准。

### 2. 创建独立自定义后端

在 GPUStack 的“推理后端 / Inference Backends”管理页面创建自定义后端，导入或粘贴仓库中的 **[`gpustack/backend.yaml`](gpustack/backend.yaml)**。默认字段为：

| 字段 | 值 |
| --- | --- |
| 后端名称 | `deepseek-v4.1-flash-1017-cn-custom` |
| 版本 | `1.0.0-custom` |
| 镜像 | `dsv41-1017-gpustack-cn:1.0.0` |
| 健康检查 | `/health` |
| Entrypoint | `python3 -u /opt/dsv41/launch.py` |
| 额外后端参数 | 留空 |

启动参数原样保留模板占位符：

```text
--model-path "{{model_path}}" --host {{worker_ip}} --port {{port}} --served-model-name "{{model_name}}"
```

请创建一个**独立的自定义后端**，然后在模型中选它。模型运行所需的 SGLang 参数由镜像内的 `launch.py` 生成；页面上的额外后端参数保持空列表，容器入口无需额外追加 `--tp-size` 等选项。

若修改模型名、镜像名或 profile，可先重新生成对应文件：

```bash
python3 scripts/deploy.py render \
  --worker YOUR_WORKER --cluster-id 1 --gpus 0,1,2,3 \
  --name deepseek-v4.1-flash-1017-cn \
  --image dsv41-1017-gpustack-cn:1.0.0
```

`render` 中填写 **GPUStack 页面上的 Worker 名称、集群 ID 和 GPU 编号**，输出位于 `generated/`。它只生成文件，不访问服务器。`generated/backend.yaml` 可在后端页面导入；`generated/model.json` 是 API 请求体示例，不是模型页面的 YAML 导入格式。

### 3. 在 GPUStack 下载并部署 ModelScope 模型

在 GPUStack 新建模型部署，填写：

| 字段 | 值 |
| --- | --- |
| 模型来源 | ModelScope |
| 仓库 ID | `deepseek-ai/DeepSeek-V4.1-Flash-1017` |
| 文件筛选 | 留空，下载完整官方 checkpoint |
| 模型/服务名称 | `deepseek-v4.1-flash-1017-cn` |
| 推理后端 / 版本 | 上一步的自定义后端 / `1.0.0-custom` |
| 副本 | `1` |
| GPU 选择 | 手动指定同一 Worker 的四张空闲卡 |
| 跨 Worker 分布式 | 关闭 |
| CPU offloading | 开启（实际 Engram 路径由下面的 `OFFLOAD_MODE` 控制） |
| 自动错误重启 | 首次验收阶段关闭 |
| 额外后端参数 | 留空 |
| 页面独立的 speculative/draft model 配置 | 留空，由本镜像的 DSPARK 配置控制 |
| 模型路由 / API 访问入口 | 启用，使用上述服务名称 |

在这个**模型部署本身的环境变量区域**，完整填写以下内容。支持批量输入时可复制 [`gpustack/model.env`](gpustack/model.env)，否则逐条录入：

```dotenv
OFFLOAD_MODE=ram
DSV41_CACHE_GIB=0
CONTEXT_LENGTH=524288
MAX_RUNNING_REQUESTS=8
MAX_TOTAL_TOKENS=4200000
MEMORY_FRACTION=0.95
CHUNKED_PREFILL_SIZE=2048
ENABLE_DSPARK=1
GPUSTACK_MODEL_RUNTIME_SHM_SIZE_GIB=32
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

RAM 模式会将完整 Engram 表映射并锁在主机内存；`DSV41_CACHE_GIB` 只控制 NVMe 模式的行缓存，在本 RAM 配置中设为 `0`。继续调大这个值不会增加 RAM 模式吞吐。`ENABLE_DSPARK=1` 会启用 checkpoint 自带的 DSPARK，block size 为 5，无需另下 draft 模型。

由 GPUStack 下载/复用 ModelScope 缓存并挂入容器。推理容器中的两个 OFFLINE 开关仅限制容器自身，不替代 GPUStack 的模型下载。不要再手动设置 `CUDA_VISIBLE_DEVICES`，GPUStack 负责分配四张卡。

保存并部署。修改已有模型的环境变量或后端后，在 GPUStack **重新创建实例**，使新容器使用新配置。首次下载、权重加载和内核编译可能较久，等待实例 Running 和健康检查通过。

### 4. 验收实际运行状态

半手动部署也可用只读验收命令；它不会修改 GPUStack 模型，API Key 仍隐藏输入：

```bash
python3 scripts/deploy.py verify \
  --url http://10.0.0.10:80 --worker YOUR_WORKER --gpus 0,1,2,3
```

这里 `--gpus` 和一键模式一样填写**本机 nvidia-smi 编号**，工具会通过 UUID 映射。若使用自定义名称/镜像/profile，传入相同的 `--name`、`--image`、`--profile`。

工具检查四卡分配、容器中的完整环境变量、`/dev/shm`、镜像源码标识、`/health`，再通过 `/server_info` 核对上下文、TP/EP、并发、token 池及 DSPARK，最后通过 GPUStack `/v1/chat/completions` 请求 `19+23`。成功报告写入 `.state/<模型名>/acceptance.json`。

也可手动检查实际容器（`CONTAINER_ID` 替换为推理容器 ID）：

```bash
docker ps --format 'table {{.ID}}\t{{.Image}}\t{{.Names}}'
docker logs --tail 100 CONTAINER_ID
docker inspect CONTAINER_ID --format '{{range .Config.Env}}{{println .}}{{end}}'
docker inspect CONTAINER_ID --format '{{.HostConfig.ShmSize}}'
```

启动器打印的 JSON 应包含 `mode: ram`、四张 GPU，以及 `--context-length 524288`、`--max-running-requests 8`、`--max-total-tokens 4200000`、`--speculative-algorithm DSPARK`。默认 `/dev/shm` 应为 `34359738368` 字节。

客户端使用 `http://GPUSTACK_HOST:PORT/v1` 和模型名 `deepseek-v4.1-flash-1017-cn`。短问答验收通过后，再按实际提示词测试 200K+ 输入、输出预留和目标并发；可用长度必须包含 chat template、工具定义和输出预算。生产并发/长文本性能须依据目标机器测量。

## 调整与维护

默认 profile 偏向吞吐，实际可用显存会影响启动。需要调整时复制 `profiles/ram-4gpu.json` 到自己的 JSON，用 `--profile /path/to/local-profile.json` 指定；例如保持 524288 上下文，降低 `MAX_RUNNING_REQUESTS` 至 `4`、`MAX_TOTAL_TOKENS` 至 `1048576`、`MEMORY_FRACTION` 至 `0.90`。这只是降低资源需求的候选配置，仍需现场验收。半手动流程同步更新模型环境变量并重新创建实例。

```bash
python3 -m unittest discover -s tests -v
bash -n deploy.sh
```

仓库自带 `AGENTS.md`、测试和 GitHub Actions。测试不下载模型、不调用真实 GPUStack，不需要 GPU。`.state/`、`generated/`、密钥、模型权重和运行日志均不纳入 Git。

## 来源与许可

SM120 / Engram 适配代码来自 [sybil-solutions 上游项目](https://github.com/sybil-solutions/deepseek-v4.1-flash-4x-rtx-pro-6000/tree/ce69105cb5812638e5fdba7ffecfb23945e11948)，固定 commit `ce69105cb5812638e5fdba7ffecfb23945e11948`，文件校验见 [`vendor-lock.json`](vendor-lock.json)。本项目新增部署工具使用 MIT，保留上游 MIT 和 SGLang Apache-2.0 许可；模型权重遵循其官方仓库许可。

接口依据：[GPUStack 2.2.3 源码](https://github.com/gpustack/gpustack/tree/v2.2.3)、[GPUStack 模型管理文档](https://docs.gpustack.ai/latest/user-guide/model-deployment-management/)、[DaoCloud 镜像服务说明](https://github.com/DaoCloud/public-image-mirror)。固定版本与文件来源详见 [`PROVENANCE.txt`](PROVENANCE.txt)。
