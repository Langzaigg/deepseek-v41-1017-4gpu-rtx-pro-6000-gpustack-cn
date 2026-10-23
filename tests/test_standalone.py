"""Standalone Docker and distributable-package checks; no Docker or GPUs needed."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import deploy as core
import standalone as s
import make_bundle as bundle


def options(**overrides):
    args = argparse.Namespace(model_dir=Path("/model-data"), gpus="0,1,2,3", name="dsv41-1017-docker",
                              served_model_name=core.MODEL, image=core.IMAGE, profile=None, bind="127.0.0.1",
                              port=30000, timeout=1, download=False, revision="master", base_image="test-base",
                              rebuild=False, recreate=False, plan=False, verify_only=False)
    vars(args).update(overrides)
    return args


def gpus():
    return [{"index": i, "api_index": i, "uuid": f"GPU-test-{i}", "allocated": 0,
             "name": "RTX PRO 6000 Blackwell", "total_mib": 97280, "used_mib": 0} for i in range(4)]


def container(spec):
    return {"Image": "sha256:image", "State": {"Running": True},
            "Config": {"Labels": {s.OWNER_LABEL: s.OWNER, s.CONFIG_LABEL: s.config_hash(spec)},
                       "Env": [f"{k}={v}" for k, v in spec["env"].items()], "Cmd": s.launch_args(spec["model"])},
            "HostConfig": {"DeviceRequests": [{"DeviceIDs": spec["gpus"]}], "ShmSize": 32 * core.GIB,
                           "PortBindings": {"30000/tcp": [{"HostIp": spec["bind"], "HostPort": str(spec["port"])}]}},
            "Mounts": [{"Destination": "/model", "Source": spec["model_dir"], "RW": False}]}


class DockerContractTests(unittest.TestCase):
    def setUp(self):
        self.spec = s.desired_spec(options(), core.load_profile(), gpus())

    def test_gpu_csv_loopback_and_mount_permissions(self):
        cmd = s.docker_command("test", self.spec)
        self.assertEqual(cmd[cmd.index("--gpus") + 1], '"device=GPU-test-0,GPU-test-1,GPU-test-2,GPU-test-3"')
        self.assertEqual(cmd[cmd.index("--publish") + 1], "127.0.0.1:30000:30000")
        self.assertIn(f'type=bind,source={self.spec["model_dir"]},target=/model,readonly', cmd)
        self.assertIn("IPC_LOCK", cmd)
        self.assertIn("memlock=-1:-1", cmd)
        self.assertNotIn("--privileged", cmd)
        self.assertIn("--pull=never", cmd)

    def test_preflight_does_not_bind_ports_and_is_removed(self):
        cmd = s.docker_command("test", self.spec, preflight=True)
        self.assertIn("--rm", cmd)
        self.assertEqual(cmd[-1], "--preflight")
        self.assertNotIn("--publish", cmd)
        self.assertNotIn("--name", cmd)

    def test_actual_devices_env_and_mounts_are_checked(self):
        good = container(self.spec)
        s.validate_container(good, self.spec, "sha256:image")
        for mutate in (
            lambda x: x["HostConfig"]["DeviceRequests"][0].update(DeviceIDs=["GPU-wrong"]),
            lambda x: x["Mounts"][0].update(RW=True),
            lambda x: x["Config"].update(Env=[]),
            lambda x: x.update(Image="sha256:old"),
            lambda x: x["HostConfig"]["PortBindings"].update({"30000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "30000"}]}),
        ):
            bad = copy.deepcopy(good)
            mutate(bad)
            with self.assertRaises(core.DeployError):
                s.validate_container(bad, self.spec, "sha256:image")

    def test_unmanaged_container_refused(self):
        bad = container(self.spec)
        bad["Config"]["Labels"] = {}
        with self.assertRaisesRegex(core.DeployError, "不属于"):
            s.require_owned(bad)

    def test_exited_container_is_not_ready(self):
        bad = container(self.spec)
        bad["State"]["Running"] = False
        with mock.patch.object(s, "inspect_container", return_value=bad):
            with self.assertRaisesRegex(core.DeployError, "退出"):
                s.wait_and_verify(options(), self.spec, "sha256:image")

    def test_health_503_is_waited_and_wrong_answer_fails(self):
        api = mock.Mock()
        engine = {"context_length": 524288, "tp_size": 4, "ep_size": 4, "max_running_requests": 8,
                  "max_total_tokens": 4200000, "mem_fraction_static": .95, "chunked_prefill_size": 2048,
                  "speculative_algorithm": "DSPARK", "speculative_dspark_block_size": 5}
        api.request.side_effect = [core.DeployError("HTTP 503"), b"", engine,
                                   {"choices": [{"message": {"content": "41"}}]}]
        with mock.patch.object(s, "inspect_container", return_value=container(self.spec)), \
             mock.patch.object(core, "API", return_value=api), mock.patch.object(s.time, "sleep") as sleep, \
             mock.patch.object(core, "log"):
            with self.assertRaisesRegex(core.DeployError, "42"):
                s.wait_and_verify(options(timeout=60), self.spec, "sha256:image")
            sleep.assert_called_once_with(5)


class StandaloneFlowTests(unittest.TestCase):
    def invoke(self, *, existing=False, **kwargs):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(s, "ROOT", Path(tmp)):
            args = options(**kwargs)
            spec = s.desired_spec(args, core.load_profile(), gpus())
            old = container(spec) if existing else None
            with mock.patch.object(core, "host_preflight"), mock.patch.object(core, "memory_preflight"), \
                 mock.patch.object(core, "check_free") as free, mock.patch.object(s, "selected_gpus", return_value=gpus()), \
                 mock.patch.object(s, "validate_checkpoint"), mock.patch.object(s, "inspect_container", return_value=old), \
                 mock.patch.object(core, "build_image") as build, mock.patch.object(core, "inspect_image", return_value={
                     "Id": "sha256:image", "Config": {"Labels": {"deployment.bundle.sha256": core.recipe_hash()}}}), \
                 mock.patch.object(core, "run") as run, mock.patch.object(s, "wait_and_verify", return_value={
                     "base_url": "http://127.0.0.1:30000/v1", "status": "smoke_passed"}) as verify, \
                 mock.patch.object(core, "log"):
                s.deploy(args)
                return run.call_args_list, build.call_count, free.call_count, verify.call_count

    def test_new_deployment_preflight_before_background_container(self):
        calls, builds, free, verify = self.invoke()
        self.assertEqual((builds, free, verify), (1, 2, 1))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].args[0][-1], "--preflight")
        self.assertIn("-d", calls[1].args[0])

    def test_existing_equal_config_is_not_restarted(self):
        calls, _, free, verify = self.invoke(existing=True)
        self.assertEqual((calls, free, verify), ([], 0, 1))

    def test_recreate_only_removes_owned_named_container(self):
        calls, _, _, _ = self.invoke(existing=True, recreate=True)
        self.assertEqual(calls[0].args[0], ["docker", "stop", "--time", "60", "dsv41-1017-docker"])
        self.assertEqual(calls[1].args[0], ["docker", "rm", "dsv41-1017-docker"])
        self.assertNotIn("--force", calls[1].args[0])

    def test_plan_and_verify_do_not_build_or_run_containers(self):
        self.assertEqual(self.invoke(plan=True), ([], 0, 0, 0))
        self.assertEqual(self.invoke(existing=True, verify_only=True), ([], 0, 0, 1))

    def test_download_does_not_replace_live_model(self):
        with self.assertRaisesRegex(core.DeployError, "覆盖"):
            self.invoke(existing=True, download=True)


class BundleTests(unittest.TestCase):
    def test_bundle_is_current_and_contains_no_local_state(self):
        expected = bundle.payloads()
        archive_path = ROOT / "packages" / f"{bundle.NAME}.zip"
        with zipfile.ZipFile(archive_path) as archive:
            actual = {name.split("/", 1)[1]: archive.read(name) for name in archive.namelist()}
        self.assertEqual(actual, expected)
        self.assertFalse(any(part in name for name in actual for part in (".state/", ".venv", "__pycache__", "models/", ".git/")))
        digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        self.assertEqual((ROOT / "packages/SHA256SUMS.txt").read_text().split()[0], digest)
        for line in actual["SHA256SUMS.txt"].decode().splitlines():
            expected_hash, path = line.split("  ", 1)
            self.assertEqual(hashlib.sha256(actual[path]).hexdigest(), expected_hash)

    def test_extracted_package_has_working_entrypoint_without_gpustack(self):
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(ROOT / "packages" / f"{bundle.NAME}.zip") as archive:
                archive.extractall(tmp)
            root = Path(tmp) / bundle.NAME
            result = subprocess.run([sys.executable, str(root / "scripts/standalone.py"), "--help"],
                                    capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--model-dir", result.stdout)
            self.assertIn("--verify-only", result.stdout)

    def test_requirements_consistent(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        standalone = (ROOT / "standalone/README.md").read_text(encoding="utf-8")
        rows = [line for line in standalone.splitlines() if line.startswith(("| GPU |", "| 主机内存 |", "| 模型盘 |", "| 系统 |", "| 网络 |"))]
        for row in rows:
            self.assertIn(row, readme)


if __name__ == "__main__":
    unittest.main()
