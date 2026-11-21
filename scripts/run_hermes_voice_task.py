"""Run a voice-submitted Puppy change with local Hermes and a restricted worktree."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
GIT = "/Library/Developer/CommandLineTools/usr/bin/git"
sys.path.insert(0, str(ROOT))
from companion.hermes_voice import write_record


def clean_environment(home, temporary):
    return {
        "HOME": str(home), "TMPDIR": str(temporary),
        "PATH": str(ROOT / ".venv/bin") + ":/opt/homebrew/bin:/usr/bin:/bin",
        "LANG": "en_US.UTF-8", "TERM": "dumb", "NO_COLOR": "1",
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PUPPY_DATA_DIR": str(temporary / "test-data"),
        "PUPPY_DEVICE_TOKEN": "isolated-test-device",
        "PUPPY_PARENT_TOKEN": "isolated-test-parent",
        "PUPPY_OPUS_LIBRARY": "/opt/homebrew/opt/opus/lib/libopus.dylib",
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "Puppy voice sandbox",
        "DEVELOPER_DIR": "/Library/Developer/CommandLineTools",
        "GIT_AUTHOR_EMAIL": "puppy@localhost", "GIT_COMMITTER_NAME": "Puppy voice sandbox",
        "GIT_COMMITTER_EMAIL": "puppy@localhost",
    }


def run_command(command, *, cwd, env, timeout=90, output=None):
    return subprocess.run(command, cwd=cwd, env=env, timeout=timeout, check=True,
                          stdout=output or subprocess.PIPE, stderr=subprocess.STDOUT)


def snapshot(destination):
    for directory, children, files in os.walk(ROOT / "companion", followlinks=False):
        children[:] = [name for name in children
                       if name != "__pycache__" and not (Path(directory) / name).is_symlink()]
        for name in files:
            source = Path(directory) / name
            if source.is_symlink() or source.suffix not in {".py", ".html", ".js", ".css", ".json", ".swift"}:
                continue
            target = destination / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    runner = destination / "scripts" / Path(__file__).name
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_bytes(Path(__file__).read_bytes())


def sandbox_policy(repo, home, temporary, profile, runtime, port):
    def quoted(path):
        return json.dumps(str(path))
    readable = ["/System", "/usr", "/bin", "/sbin", "/opt/homebrew",
                "/Library/Developer/CommandLineTools", "/private/etc",
                ROOT / ".venv", runtime, repo, home, temporary, profile]
    writable = [repo, home, temporary, profile]
    return "\n".join([
        '(version 1)', '(deny default)', '(import "system.sb")',
        '(allow process-exec process-fork sysctl-read mach-lookup)',
        '(allow signal (target self))', '(allow file-read-metadata)',
        '(allow file-read* ' + " ".join(f"(subpath {quoted(path)})" for path in readable)
        + ' (literal "/private/var/run/resolv.conf") (literal "/dev/null")'
          ' (literal "/dev/urandom") (literal "/dev/random") (literal "/dev/tty"))',
        f'(deny file-read* (subpath {quoted(runtime / ".git")})'
        f' (literal {quoted(runtime / ".env")}))',
        '(allow file-write* ' + " ".join(f"(subpath {quoted(path)})" for path in writable)
        + ' (literal "/dev/null") (literal "/dev/tty"))',
        f'(allow network-outbound (remote ip "localhost:{port}"))',
    ]) + "\n"


def stop_process(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def isolated_command(command, policy, *, repo, env, log, timeout):
    with log.open("wb") as output:
        process = subprocess.Popen(
            ["/usr/bin/sandbox-exec", "-f", str(policy), *command],
            cwd=repo, env=env, stdin=subprocess.DEVNULL, stdout=output,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
            if code:
                raise RuntimeError(f"{log.name} exited with {code}")
        finally:
            stop_process(process)


def wait_model(port, process):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("local_model_exited")
        try:
            with opener.open(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(.5)
    raise TimeoutError("local_model_start_timeout")


def validate_reply_change(before, after, original_reply="我刚才没看清，请再试一次。"):
    original_tree = ast.parse(before)
    candidate_tree = ast.parse(after)
    original_constants = [node for node in ast.walk(original_tree) if isinstance(node, ast.Constant)]
    candidate_constants = [node for node in ast.walk(candidate_tree) if isinstance(node, ast.Constant)]
    changes = [(old, new) for old, new in zip(original_constants, candidate_constants)
               if old.value != new.value]
    if (len(original_constants) != len(candidate_constants) or len(changes) != 1
            or changes[0][0].value != original_reply
            or not isinstance(changes[0][1].value, str)):
        raise ValueError("Change only the general camera-unavailable reply, preserving the parent-permission reply.")
    reply = changes[0][1].value
    changes[0][1].value = changes[0][0].value
    if ast.dump(candidate_tree) != ast.dump(original_tree):
        raise ValueError("Preserve all code behavior outside the general camera-unavailable reply.")
    return reply


def perform(task):
    record_path = task / "task.json"
    record = json.loads(record_path.read_text())
    def update(state, **details):
        record.update(state=state, updated_at=time.time(), **details)
        write_record(record_path, record)
    model_process = None
    try:
        repo, home, temporary, profile = [task / name for name in ("repo", "home", "tmp", "profile")]
        for directory in (repo, home, temporary, profile):
            directory.mkdir(mode=0o700)
        env = clean_environment(home, temporary)
        env["HERMES_HOME"] = str(profile)
        env["XDG_CACHE_HOME"] = str(temporary / "cache")
        snapshot(repo)
        before_source = (repo / "companion/server.py").read_text()
        update("baseline")
        for arguments in (["init", "-q"], ["add", "companion", "scripts"],
                          ["commit", "-q", "-m", "Deployed Puppy source before voice task"]):
            run_command([GIT, *arguments], cwd=repo, env=env)
        baseline = run_command([GIT, "rev-parse", "HEAD"], cwd=repo, env=env).stdout.decode().strip()
        update("baseline", baseline=baseline)
        runtime = Path.home() / ".hermes/hermes-agent"
        model = ROOT / "models/Qwen3.5-9B-4bit"
        python = ROOT / ".venv/bin/python"
        with socket.socket() as available:
            available.bind(("127.0.0.1", 0))
            port = available.getsockname()[1]
        policy = task / "hermes.sb"
        policy.write_text(sandbox_policy(repo, home, temporary, profile, runtime, port))
        tests = [str(python), "-B", "-m", "unittest", "discover", "-s", "companion", "-t", "."]
        with (task / "baseline-tests.log").open("wb") as log:
            run_command(tests, cwd=repo, env=env, output=log, timeout=120)
        config = (
            f"model:\n  default: {json.dumps(str(model))}\n  provider: custom\n"
            f"  base_url: \"http://127.0.0.1:{port}/v1\"\n"
            "  api_key: loopback-task\n  context_length: 65536\n"
            f"terminal:\n  backend: local\n  cwd: {json.dumps(str(repo))}\n  timeout: 60\n"
            "agent:\n  max_turns: 8\n"
        )
        (profile / "config.yaml").write_text(config)
        cli = profile / "voice_cli.py"
        cli.write_text(
            "import toolsets\n"
            "toolsets.TOOLSETS['puppy-voice'] = {'description': 'Patch and test the isolated Puppy code', "
            "'tools': ['patch', 'terminal', 'read_file'], 'includes': []}\n"
            "from hermes_cli.main import main\nmain()\n"
        )
        # Only task text is handed to Hermes. Audio and production credentials stay outside its sandbox.
        excerpt = before_source
        start = excerpt.index('        if visual and visual["observation"].get("available") is False:')
        end = excerpt.index('        request = ChatRequest(', start)
        original_reply_line = next(line for line in reversed(excerpt[start:end].splitlines())
                                   if line.lstrip().startswith('return {"reply":'))
        original_reply = ast.literal_eval(original_reply_line.strip().removeprefix("return "))["reply"]
        prompt = (
            f"Current working directory: {repo}\n"
            "All source paths below are relative to this working directory.\n"
            "A development request transcribed from the actual Puppy microphone follows.\n"
            + record["instruction"]
            + "\nWork only in this isolated repo. Produce a minimal real source change. "
            "The deployed voice camera-failure branch in companion/server.py is:\n"
            + excerpt[start:end]
            + "\nYour first action should be patch with mode=replace, path=companion/server.py, "
            + "old_string exactly " + repr(original_reply_line) + ", "
            "and new_string consisting of one reply line reflecting the voice request. "
            'Preserve the parent-permission line \'return {"reply": "要家长允许看图，我才能帮你看。"}\'. '
            "Then use your terminal tool to run exactly "
            "python -B -m unittest companion.test_camera_dialogue"
            "\nFinish with the changed behavior and actual test result. "
            "No deployment is available in this sandbox.\n"
        )
        (task / "task-prompt.txt").write_text(prompt)
        update("model_starting")
        with (task / "model.log").open("wb") as log:
            model_process = subprocess.Popen(
                [str(python), "-m", "mlx_vlm.server", "--host", "127.0.0.1",
                 "--port", str(port), "--model", str(model)],
                cwd=repo, env=env, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
        wait_model(port, model_process)
        update("hermes_running")
        for attempt in range(2):
            log_name = "hermes.log" if attempt == 0 else "hermes-repair.log"
            isolated_command(
                [str(runtime / "venv/bin/python"), str(cli), "chat", "-t", "puppy-voice",
                 "--max-turns", "8", "-q", prompt],
                policy, repo=repo, env=env, log=task / log_name, timeout=1200,
            )
            candidate_source = (repo / "companion/server.py").read_text()
            try:
                candidate_reply = validate_reply_change(before_source, candidate_source, original_reply)
                break
            except ValueError as error:
                if attempt:
                    raise
                update("hermes_repairing", validation_error=str(error))
                current_start = candidate_source.index('        if visual and visual["observation"].get("available") is False:')
                current_end = candidate_source.index("        request = ChatRequest(", current_start)
                prompt += ("\nIndependent validation requires a correction: " + str(error)
                           + "\nThe current candidate branch is:\n"
                           + candidate_source[current_start:current_end]
                           + "\nUse patch to correct this candidate, then run the same camera tests.")
        update("testing")
        git = ["/usr/bin/sandbox-exec", "-f", str(policy), GIT,
               "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null"]
        changed = run_command([*git, "diff", "--no-ext-diff", "--no-textconv", "--name-only", baseline],
                              cwd=repo, env=env).stdout.decode().splitlines()
        unknown = run_command([*git, "ls-files", "--others", "--exclude-standard"],
                              cwd=repo, env=env).stdout.decode().splitlines()
        if unknown or any(path != "companion/server.py" for path in changed):
            raise RuntimeError("candidate_outside_voice_reply_scope")
        if not changed:
            update("no_change")
            return
        run_command([*git, "diff", "--no-ext-diff", "--no-textconv", "--check", baseline], cwd=repo, env=env)
        patch = run_command([*git, "diff", "--no-ext-diff", "--no-textconv", "--binary", baseline],
                            cwd=repo, env=env).stdout
        (task / "candidate.patch").write_bytes(patch)
        # The trusted verifier runs the original regression suite only after the
        # syntax tree proves this pilot changed a single reply string.
        with (task / "candidate-tests.log").open("wb") as log:
            run_command(tests, cwd=repo, env=env, output=log, timeout=120)
        update("review_ready", changed_files=changed,
               candidate_reply=candidate_reply,
               patch_sha256=hashlib.sha256(patch).hexdigest(),
               review_artifacts=["candidate.patch", "baseline-tests.log", "candidate-tests.log", "hermes.log"])
    except Exception as error:
        update("failed", error=type(error).__name__, error_detail=str(error)[:200])
        raise
    finally:
        stop_process(model_process)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=Path)
    parser.add_argument("--lease-fd", type=int, required=True)
    args = parser.parse_args()
    os.fstat(args.lease_fd)
    perform(args.task.resolve(strict=True))
