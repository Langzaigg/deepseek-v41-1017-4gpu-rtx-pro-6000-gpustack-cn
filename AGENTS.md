# Project context

Repository name: `deepseek-v41-1017-4gpu-rtx-pro-6000-gpustack-cn`. GitHub description is in `description.txt` and the README. Keep the existing runtime ownership marker `deepseek-v41-1017-gpustack-cn` stable so renaming the repository does not orphan existing managed deployments.

GPUStack 2.2.3 and standalone Docker deployment for DeepSeek V4.1 Flash 1017 on Linux with four RTX PRO 6000 Blackwell 96GB cards. ModelScope supplies weights; DaoCloud or an existing Docker proxy supplies image layers. UTF-8 / LF files; development may run on Windows.

- The README contains both the one-command deployment and the semi-manual GPUStack UI procedure. Keep both paths consistent with `profiles/ram-4gpu.json` and generated `gpustack/` examples.
- Standalone Docker uses `deploy-docker.sh` and `scripts/standalone.py`. Keep its tutorial, root README and shared profile consistent. Document generic minimum/recommended requirements; do not add any individual user's host inventory, addresses or paths.
- `download-model.sh` installs the pinned ModelScope Hub client in its own venv only. Never install downloader dependencies into the inference image or system Python.
- Rebuild the allowlisted deployment ZIP with `python scripts/make_bundle.py` after changes to bundled files. Commit `packages/` ZIP and checksum; never bundle local state, models, secrets or the venv.
- `scripts/deploy.py` uses Python 3.10+ standard library. Do not add a mandatory package install to the deployment path.
- Preserve the pinned upstream files and licenses. Changes require an intentional vendor-lock update with source evidence.
- Never claim GPU throughput or long-context validation from unit tests. Only a real target run can create an acceptance report, which covers configuration and short generation.
- GPUStack API writes belong only to models/backends created by this tool. Do not kill other processes, restart Docker, remove weights, or prune caches.
- Keys must not enter Git, command-line arguments or reports. Do not log entire Docker inspection objects.
- Validation: `python -m unittest discover -s tests -v`; `bash -n deploy.sh`; `bash -n deploy-docker.sh`; `bash -n download-model.sh`. Update generated examples after changing payloads or the profile.
