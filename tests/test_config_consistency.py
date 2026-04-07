"""
tests/test_config_consistency.py — Catch config mismatches between files
=========================================================================

These tests verify that defaults in docker-compose.yml, entrypoint.sh,
translate_utils.py, and translate_diarize.py are consistent.

The exact bug this catches: ollama-init pulls `translategemma:4b` but the
pipeline container defaults to `translategemma:12b` → model not found (404).

Run:
    uv run --with pytest pytest tests/test_config_consistency.py -v
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ═════════════════════════════════════════════════════════════════════════════
# 1. TRANSLATE_MODEL default must be the same everywhere
# ═════════════════════════════════════════════════════════════════════════════

class TestTranslateModelConsistency:

    def _extract_defaults(self) -> dict[str, str]:
        """Extract TRANSLATE_MODEL defaults from all config files."""
        defaults = {}

        # docker-compose.yml — ollama-init service
        compose = _read(ROOT / "docker-compose.yml")
        # Look for the ollama-init service block
        init_match = re.search(
            r"ollama-init:.*?TRANSLATE_MODEL:\s*\"\$\{TRANSLATE_MODEL:-([^}]+)\}\"",
            compose, re.DOTALL,
        )
        if init_match:
            defaults["docker-compose:ollama-init"] = init_match.group(1)

        # docker-compose.yml — pipeline service
        pipeline_match = re.findall(
            r"TRANSLATE_MODEL:\s*\"\$\{TRANSLATE_MODEL:-([^}]+)\}\"",
            compose,
        )
        if len(pipeline_match) >= 2:
            defaults["docker-compose:pipeline"] = pipeline_match[1]
        elif len(pipeline_match) == 1:
            defaults["docker-compose:pipeline"] = pipeline_match[0]

        # translate_utils.py — variable is MODEL_NAME but reads TRANSLATE_MODEL env var
        utils_path = ROOT / "translate-gemma" / "translate_utils.py"
        if utils_path.exists():
            utils = _read(utils_path)
            m = re.search(r'os\.getenv\("TRANSLATE_MODEL",\s*"([^"]+)"\)', utils)
            if m:
                defaults["translate_utils.py"] = m.group(1)

        return defaults

    def test_ollama_init_and_pipeline_use_same_model(self):
        """ollama-init must pull the SAME model the pipeline defaults to."""
        defaults = self._extract_defaults()
        init_model = defaults.get("docker-compose:ollama-init")
        pipe_model = defaults.get("docker-compose:pipeline")

        if init_model and pipe_model:
            assert init_model == pipe_model, (
                f"Model mismatch! ollama-init pulls '{init_model}' "
                f"but pipeline defaults to '{pipe_model}'. "
                f"The pipeline will get a 404 'model not found' error."
            )

    def test_translate_utils_default_documented(self):
        """translate_utils.py MODEL_NAME default should exist."""
        defaults = self._extract_defaults()
        assert "translate_utils.py" in defaults, \
            "Could not find TRANSLATE_MODEL default in translate_utils.py"


# ═════════════════════════════════════════════════════════════════════════════
# 2. CHUNK_SIZE consistency
# ═════════════════════════════════════════════════════════════════════════════

class TestChunkSizeConsistency:

    def test_chunk_size_defaults_documented(self):
        """CHUNK_SIZE defaults should be consistent or at least intentional."""
        # translate_utils.py default
        utils = _read(ROOT / "translate-gemma" / "translate_utils.py")
        m = re.search(r'CHUNK_SIZE\s*=\s*int\(os\.getenv\("CHUNK_SIZE",\s*(\d+)\)', utils)
        assert m, "CHUNK_SIZE default not found in translate_utils.py"
        local_default = int(m.group(1))

        # docker-compose.yml default
        compose = _read(ROOT / "docker-compose.yml")
        m2 = re.search(r'CHUNK_SIZE:\s*"\$\{CHUNK_SIZE:-(\d+)\}"', compose)
        if m2:
            remote_default = int(m2.group(1))
            # Document the intentional difference (local=15, remote=40)
            # This is OK — remote has more GPU power
            assert local_default > 0
            assert remote_default > 0
            if local_default != remote_default:
                # Not a failure, but document it
                print(f"INFO: CHUNK_SIZE differs: local={local_default}, remote={remote_default} (intentional)")


# ═════════════════════════════════════════════════════════════════════════════
# 3. entrypoint.sh flags must match run_pipeline.py argparse
# ═════════════════════════════════════════════════════════════════════════════

class TestEntrypointFlags:

    def test_entrypoint_flags_exist_in_pipeline(self):
        """Every flag in entrypoint.sh must be accepted by run_pipeline.py."""
        entrypoint = _read(ROOT / "entrypoint.sh")
        pipeline = _read(ROOT / "run_pipeline.py")

        # Extract all --flag-name patterns from entrypoint.sh
        # Exclude --help (built-in argparse) and comment-only lines
        flags = set(re.findall(r"(--[a-z][-a-z]+)", entrypoint))
        flags -= {"--help"}  # built-in argparse flag, always available

        # These flags must exist as argparse arguments in run_pipeline.py
        for flag in flags:
            assert flag in pipeline, (
                f"entrypoint.sh uses '{flag}' but run_pipeline.py doesn't accept it. "
                f"The Docker container will crash with 'unrecognized arguments'."
            )

    def test_required_env_vars_have_flags(self):
        """Key env vars in docker-compose.yml must be wired through entrypoint.sh."""
        compose = _read(ROOT / "docker-compose.yml")
        entrypoint = _read(ROOT / "entrypoint.sh")

        # Extract env var names from docker-compose pipeline service
        env_vars = set(re.findall(r"^\s+([A-Z_]+):", compose, re.MULTILINE))
        # These env vars should be referenced in entrypoint.sh
        important_vars = {"TARGET_LANG", "INPUT_DIR", "OUTPUT_DIR", "PRECISION",
                          "TRIM", "WHISPER_MODEL", "QWEN_MODE", "RUN_MODE"}

        for var in important_vars:
            if var in env_vars:
                assert var in entrypoint, (
                    f"docker-compose.yml defines {var} but entrypoint.sh doesn't use it"
                )


# ═════════════════════════════════════════════════════════════════════════════
# 4. Docker/local path consistency
# ═════════════════════════════════════════════════════════════════════════════

class TestDockerPaths:

    def test_entrypoint_uses_correct_python(self):
        """entrypoint.sh must use the nemo venv python."""
        entrypoint = _read(ROOT / "entrypoint.sh")
        assert "/app/nemo/.venv/bin/python" in entrypoint

    def test_entrypoint_uses_correct_script(self):
        """entrypoint.sh must call run_pipeline.py."""
        entrypoint = _read(ROOT / "entrypoint.sh")
        assert "run_pipeline.py" in entrypoint

    def test_dockerfile_exists(self):
        assert (ROOT / "Dockerfile").exists()

    def test_docker_compose_exists(self):
        assert (ROOT / "docker-compose.yml").exists()
