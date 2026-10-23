"""Offline contract tests. These do not validate GPU performance or model loading."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import deploy as d


def worker():
    return {"id": 3, "name": "test-worker", "cluster_id": 7, "owner_principal_id": 2,
            "state": "ready", "ip": "127.0.0.1", "status": {"gpu_devices": [
                {"index": i + 2, "uuid": f"GPU-test-{i}", "type": "cuda", "memory": {"allocated": 0}}
                for i in range(4)]}}


def gpus():
    return [{"index": i, "uuid": f"GPU-test-{i}", "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
             "total_mib": 97280, "used_mib": 0} for i in range(4)]


class ProfileTests(unittest.TestCase):
    def test_cli_render_accepts_default_name_with_dot(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(ROOT / "scripts/deploy.py"), "render",
                                     "--worker", "worker", "--cluster-id", "1", "--gpus", "0,1,2,3",
                                     "--output", tmp], capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((Path(tmp) / "model.json").read_text())["name"], d.MODEL)

    def test_vendor_integrity(self):
        self.assertEqual(len(d.vendor_check()["files"]), 7)
        self.assertEqual(len(d.recipe_hash()), 64)

    def test_profile_and_launch_agree(self):
        spec = importlib.util.spec_from_file_location("deployment_launch", ROOT / "launch.py")
        launch = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"resource": types.ModuleType("resource")}):
            spec.loader.exec_module(launch)
        args = argparse.Namespace(served_model_name=d.MODEL, host="127.0.0.1", port=40000)
        env = d.load_profile()
        with mock.patch.dict(os.environ, env, clear=True):
            command = launch.make_command(args, Path("/model"))
        self.assertEqual(command[command.index("--context-length") + 1], "524288")
        self.assertEqual(command[command.index("--max-total-tokens") + 1], "4200000")
        self.assertEqual(command[command.index("--speculative-algorithm") + 1], "DSPARK")
        self.assertEqual(command[command.index("--tp") + 1], "4")
        self.assertEqual(command[command.index("--ep-size") + 1], "4")
        with mock.patch.dict(os.environ, {**env, "ENABLE_DSPARK": "0"}, clear=True):
            self.assertNotIn("--speculative-algorithm", launch.make_command(args, Path("/model")))

    def test_invalid_profile_rejected(self):
        for change in ({"MAX_TOTAL_TOKENS": "200000"}, {"CONTEXT_LENGTH": "8192"},
                       {"OFFLOAD_MODE": "nvme"}, {"CHUNKED_PREFILL_SIZE": "2049"},
                       {"ENABLE_DSPARK": "true"}, {"MEMORY_FRACTION": "0.99"}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "invalid.json"
                path.write_text(json.dumps({**d.load_profile(), **change}), encoding="utf-8")
                with self.assertRaises(d.DeployError):
                    d.load_profile(path)

    def test_generated_examples_current(self):
        backend = d.backend_payload(d.MODEL, d.IMAGE, d.load_profile())
        self.assertEqual((ROOT / "gpustack/backend.yaml").read_text(encoding="utf-8"), d.yaml_text(backend))
        self.assertEqual((ROOT / "gpustack/model.env").read_text(encoding="utf-8"),
                         "".join(f"{k}={v}\n" for k, v in d.load_profile().items()))


class SelectionTests(unittest.TestCase):
    def test_uuid_mapping_does_not_assume_same_gpu_indexes(self):
        result = d.map_gpus(worker(), gpus(), [0, 1, 2, 3])
        self.assertEqual([g["api_index"] for g in result], [2, 3, 4, 5])
        self.assertEqual(d.choose_worker([worker()], gpus(), None)["id"], 3)

    def test_duplicate_and_nonlocal_selection_refused(self):
        with self.assertRaises(d.DeployError):
            d.parse_indexes("0,1,1,3")
        with self.assertRaises(d.DeployError):
            d.choose_worker([worker(), worker()], gpus(), None)
        with self.assertRaises(d.DeployError):
            d.map_gpus(worker(), gpus(), [0, 1, 2, 7])

    def test_occupied_cards_and_other_models_refused(self):
        selected = d.map_gpus(worker(), gpus(), [0, 1, 2, 3])
        with mock.patch.object(d, "run", return_value=subprocess.CompletedProcess([], 0, "GPU-test-0, 1234\n", "")):
            with self.assertRaisesRegex(d.DeployError, "进程"):
                d.check_free(selected, [], 3)
        with self.assertRaisesRegex(d.DeployError, "其他实例"):
            d.check_free(selected, [{"worker_id": 3, "model_id": 99, "gpu_indexes": [2], "name": "other"}], 3)

    def test_unmanaged_collision_is_not_adopted(self):
        model = {"name": d.MODEL, "id": 1, "meta": {}}
        with self.assertRaises(d.DeployError):
            d.find_owned([model], "name", d.MODEL, "模型")
        self.assertEqual(d.find_owned([model], "name", d.MODEL, "模型", False), model)

    def test_engine_report_required_not_only_http_200(self):
        env = d.load_profile()
        info = {"context_length": 524288, "tp_size": 4, "ep_size": 4,
                "max_running_requests": 8, "max_total_tokens": 4200000,
                "mem_fraction_static": .95, "chunked_prefill_size": 2048,
                "speculative_algorithm": "DSPARK", "speculative_dspark_block_size": 5}
        self.assertEqual(d.check_server_info({"server_args": info}, env), info)
        for broken in ({}, {**info, "context_length": 8192}, {**info, "speculative_algorithm": None}):
            with self.assertRaises(d.DeployError):
                d.check_server_info(broken, env)

    def test_extra_environment_triggers_recreation(self):
        desired = d.model_payload(d.MODEL, worker(), [2, 3, 4, 5], d.load_profile())
        existing = copy.deepcopy(desired)
        existing["env"]["CUDA_VISIBLE_DEVICES"] = "0,1"
        self.assertTrue(d.differs(existing, desired))


class FakeAPI:
    def __init__(self, model=None, backend=None):
        self.url = "http://127.0.0.1:9999"
        self.model, self.backend = model, backend
        self.writes = []

    def items(self, path):
        if path == "/v2/workers":
            return [worker()]
        if path == "/v2/models":
            return [self.model] if self.model else []
        if path == "/v2/inference-backends":
            return [self.backend] if self.backend else []
        if path == "/v2/model-instances" or path.endswith("/instances"):
            if not self.model or self.model["replicas"] == 0:
                return []
            return [{"id": 10, "name": "test-instance", "model_id": self.model["id"], "worker_id": 3,
                     "state": "running", "gpu_indexes": [2, 3, 4, 5], "port": 40000}]
        raise AssertionError(path)

    def request(self, method, path, payload=None, **kwargs):
        self.writes.append((method, path, copy.deepcopy(payload)))
        if "inference-backends" in path:
            self.backend = {**payload, "id": 9}
            return self.backend
        if "models" in path:
            self.model = {**payload, "id": 8, "owner_principal_id": 2}
            return self.model
        raise AssertionError(path)


class DeploymentFlowTests(unittest.TestCase):
    def invoke(self, api, *, plan=False, rebuild=False, redeploy=False, action="deploy"):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "profiles").mkdir()
            (root / "profiles/ram-4gpu.json").write_bytes((ROOT / "profiles/ram-4gpu.json").read_bytes())
            args = argparse.Namespace(action=action, profile=None, url=api.url, worker="test-worker", gpus="0,1,2,3",
                                      name=d.MODEL, image=d.IMAGE, plan=plan, rebuild=rebuild,
                                      redeploy=redeploy, timeout=1, base_image="test")
            with mock.patch.object(d, "ROOT", root), mock.patch.object(d, "API", return_value=api), \
                 mock.patch.dict(os.environ, {"GPUSTACK_API_KEY": "test-key"}), \
                 mock.patch.object(d, "host_preflight"), mock.patch.object(d, "memory_preflight"), \
                 mock.patch.object(d, "local_gpus", side_effect=gpus), mock.patch.object(d, "check_free") as free, \
                 mock.patch.object(d, "build_image", return_value=rebuild) as build, \
                 mock.patch.object(d, "verify_runtime", return_value={"status": "smoke_passed"}) as verify, \
                 mock.patch.object(d, "log"):
                d.deploy(args)
                return free.call_count, build.call_count, verify.call_count

    def existing(self):
        model = {**d.model_payload(d.MODEL, worker(), [2, 3, 4, 5], d.load_profile()), "id": 8}
        backend = {**d.backend_payload(d.MODEL, d.IMAGE, d.load_profile(), 2), "id": 9}
        return FakeAPI(model, backend)

    def test_new_deployment_writes_env_on_model_and_route(self):
        api = FakeAPI()
        free, build, verify = self.invoke(api)
        self.assertEqual((free, build, verify), (2, 1, 1))
        self.assertEqual([w[:2] for w in api.writes], [("POST", "/v2/inference-backends"), ("POST", "/v2/models")])
        self.assertEqual(api.model["env"], d.load_profile())
        self.assertTrue(api.model["enable_model_route"])
        self.assertEqual(api.model["gpu_selector"]["gpu_ids"], [f"test-worker:cuda:{i}" for i in [2, 3, 4, 5]])

    def test_repeat_does_not_restart(self):
        api = self.existing()
        self.invoke(api)
        self.assertEqual(api.writes, [])

    def test_changed_env_stops_then_recreates(self):
        api = self.existing()
        api.model["env"]["CONTEXT_LENGTH"] = "8192"
        self.invoke(api)
        self.assertEqual([w[2]["replicas"] for w in api.writes], [0, 1])
        self.assertEqual(api.model["env"]["CONTEXT_LENGTH"], "524288")

    def test_rebuild_forces_recreation(self):
        api = self.existing()
        self.invoke(api, rebuild=True)
        self.assertEqual([w[2]["replicas"] for w in api.writes], [0, 1])

    def test_plan_and_verify_never_write_or_build(self):
        api = self.existing()
        self.assertEqual(self.invoke(api, plan=True), (0, 0, 0))
        self.assertEqual(api.writes, [])
        api.model["meta"] = {}  # UI-created deployment can still be verified read-only.
        self.assertEqual(self.invoke(api, action="verify"), (0, 0, 1))
        self.assertEqual(api.writes, [])

    def test_wait_error_and_timeout_are_not_success(self):
        api = mock.Mock()
        api.items.return_value = [{"name": "test", "state": "error", "state_message": "OOM"}]
        with self.assertRaisesRegex(d.DeployError, "OOM"):
            d.wait_running(api, 8, 1)
        api.items.return_value = []
        with mock.patch.object(d.time, "monotonic", side_effect=[0, 2]):
            with self.assertRaisesRegex(d.DeployError, "超时"):
                d.wait_running(api, 8, 1)


class HTTPTests(unittest.TestCase):
    def test_real_http_pagination_auth_and_redirect_rejection(self):
        hits = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                hits.append((self.path, self.headers.get("Authorization")))
                if self.path == "/redirect":
                    self.send_response(302)
                    self.send_header("Location", "/must-not-follow")
                    self.end_headers()
                    return
                self.send_response(200)
                self.end_headers()
                page = 2 if "page=2&" in self.path else 1
                self.wfile.write(json.dumps({"items": [{"id": page}], "pagination": {"total": 2, "totalPage": 2}}).encode())

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            api = d.API(f"http://127.0.0.1:{server.server_port}", "secret")
            self.assertEqual(api.items("/v2/workers"), [{"id": 1}, {"id": 2}])
            self.assertTrue(all(auth == "Bearer secret" for _, auth in hits))
            with self.assertRaisesRegex(d.DeployError, "HTTP 302"):
                api.request("GET", "/redirect")
            self.assertNotIn("/must-not-follow", [path for path, _ in hits])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_credentials_in_url_rejected(self):
        with self.assertRaises(d.DeployError):
            d.API("http://username:password@example.test")


class RuntimeVerificationTests(unittest.TestCase):
    def verify(self, *, changed_env=False, answer="42"):
        env = d.load_profile()
        engine = {"context_length": 524288, "tp_size": 4, "ep_size": 4,
                  "max_running_requests": 8, "max_total_tokens": 4200000,
                  "mem_fraction_static": .95, "chunked_prefill_size": 2048,
                  "speculative_algorithm": "DSPARK", "speculative_dspark_block_size": 5}
        actual_env = {**env, "CONTEXT_LENGTH": "8192"} if changed_env else env
        container = {"Id": "a" * 64, "Image": "sha256:abc", "HostConfig": {"ShmSize": 32 * d.GIB},
                     "Config": {"Image": "docker.io/library/" + d.IMAGE,
                                "Env": [f"{k}={v}" for k, v in actual_env.items()],
                                "Labels": {"deployment.bundle.sha256": "source-hash"}}}
        model = d.model_payload(d.MODEL, worker(), [2, 3, 4, 5], env)
        instance = {"worker_id": 3, "name": "test-instance", "gpu_indexes": [2, 3, 4, 5], "port": 40000}
        backend, frontend = mock.Mock(), mock.Mock()
        backend.request.side_effect = [b"", engine]
        frontend.request.return_value = {"choices": [{"message": {"content": answer}}]}
        with mock.patch.object(d, "run", side_effect=[
            subprocess.CompletedProcess([], 0, "container-id\n", ""),
            subprocess.CompletedProcess([], 0, json.dumps([container]), "")]), \
             mock.patch.object(d, "inspect_image", return_value={"Id": "sha256:abc"}), \
             mock.patch.object(d, "recipe_hash", return_value="source-hash"), \
             mock.patch.object(d, "API", side_effect=[backend, frontend]) as factory:
            report = d.verify_runtime(types.SimpleNamespace(url="http://front.test"), instance, model,
                                      worker(), env, d.IMAGE, "inference-key")
            self.assertEqual(factory.call_args_list[0], mock.call("http://127.0.0.1:40000"))
            self.assertEqual(factory.call_args_list[1], mock.call("http://front.test", "inference-key"))
            self.assertNotIn("inference-key", json.dumps(report))
            return report

    def test_actual_container_and_engine_and_generation_are_required(self):
        self.assertEqual(self.verify()["status"], "smoke_passed")
        with self.assertRaisesRegex(d.DeployError, "环境变量"):
            self.verify(changed_env=True)
        with self.assertRaisesRegex(d.DeployError, "预期答案"):
            self.verify(answer="421")


if __name__ == "__main__":
    unittest.main()
