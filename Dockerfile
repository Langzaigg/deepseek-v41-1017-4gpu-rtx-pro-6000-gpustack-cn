ARG BASE_IMAGE=m.daocloud.io/docker.io/lmsysorg/sglang@sha256:c4ca651192e57e91989b5176c3665148131b9a171e53861dee87f5e57cef25b5
FROM ${BASE_IMAGE}
ARG BUNDLE_SHA256=manual-unverified
WORKDIR /opt/dsv41
COPY adapter /opt/dsv41/adapter
RUN g++ -O2 -Wall -Wextra -Werror -std=c++17 -shared -fPIC -pthread \
    adapter/row_store.cpp -o adapter/librow_store.so
RUN test -f /sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120.py
COPY runtime/flash_mla_sm120.py /sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120.py
COPY launch.py /opt/dsv41/launch.py
COPY licenses /opt/dsv41/licenses
ENV PYTHONPATH=/opt/dsv41/adapter \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    OFFLOAD_MODE=ram \
    DSV41_CACHE_GIB=0 \
    CONTEXT_LENGTH=524288 \
    MAX_RUNNING_REQUESTS=8 \
    MAX_TOTAL_TOKENS=4200000 \
    MEMORY_FRACTION=0.95 \
    CHUNKED_PREFILL_SIZE=2048 \
    ENABLE_DSPARK=1 \
    GPUSTACK_MODEL_RUNTIME_SHM_SIZE_GIB=32
LABEL org.opencontainers.image.title="DeepSeek V4.1 Flash 1017 - GPUStack CN" \
    deployment.bundle.sha256="${BUNDLE_SHA256}" \
    deployment.recipe.revision="ce69105cb5812638e5fdba7ffecfb23945e11948" \
    deployment.validation="Static checks only; GPU acceptance required"
HEALTHCHECK NONE
ENTRYPOINT ["python3", "-u", "/opt/dsv41/launch.py"]
CMD []
