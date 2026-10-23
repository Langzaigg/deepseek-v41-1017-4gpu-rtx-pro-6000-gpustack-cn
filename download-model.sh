#!/usr/bin/env bash
# Full ModelScope snapshot; no Hugging Face fallback, no system pip changes.
set -euo pipefail
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
if [[ $# -lt 1 || $# -gt 2 ]]; then
  printf '%s\n' '用法：bash download-model.sh /absolute/model/directory [revision]'
  exit 2
fi
model_dir=$1
revision=${2:-master}
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "$model_dir" != /* || "$model_dir" == / ]]; then
  printf '%s\n' '请提供模型专用的绝对目录，不能使用根目录。' >&2
  exit 2
fi
venv_dir="$project_dir/.venv-modelscope"
if [[ ! -x "$venv_dir/bin/python" ]]; then
  python3 -m venv "$venv_dir"
fi
if ! "$venv_dir/bin/python" -c 'from importlib.metadata import version; assert version("modelscope-hub") == "0.4.0"' 2>/dev/null; then
  "$venv_dir/bin/python" -m pip install \
    --index-url "${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
    'modelscope-hub==0.4.0'
fi
mkdir -p -- "$model_dir"
exec "$venv_dir/bin/ms-hub" --endpoint https://modelscope.cn download \
  deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --local-dir "$model_dir" --revision "$revision"
