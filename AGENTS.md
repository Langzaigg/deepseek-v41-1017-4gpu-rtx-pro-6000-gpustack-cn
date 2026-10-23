# Project context

GPUStack 2.2.3 deployment for DeepSeek V4.1 Flash 1017 on one Linux Docker Worker with four RTX PRO 6000 Blackwell 96GB cards. ModelScope supplies weights; DaoCloud or an existing Docker proxy supplies image layers. UTF-8 / LF files; development may run on Windows.

- The README contains both the one-command deployment and the semi-manual GPUStack UI procedure. Keep both paths consistent with `profiles/ram-4gpu.json` and generated `gpustack/` examples.
- `scripts/deploy.py` uses Python 3.10+ standard library. Do not add a mandatory package install to the deployment path.
- Preserve the pinned upstream files and licenses. Changes require an intentional vendor-lock update with source evidence.
- Never claim GPU throughput or long-context validation from unit tests. Only a real target run can create an acceptance report, which covers configuration and short generation.
- GPUStack API writes belong only to models/backends created by this tool. Do not kill other processes, restart Docker, remove weights, or prune caches.
- Keys must not enter Git, command-line arguments or reports. Do not log entire Docker inspection objects.
- Validation: `python -m unittest discover -s tests -v`; `bash -n deploy.sh`. Update generated examples after changing payloads or the profile.
