"""Settings come only from CORPUS_DIR, PORT and GEMINI_API_KEY, with clear errors."""

import os
from pathlib import Path

import pytest

from corpusqa.config import ASK_DEADLINE_S, LANGFUSE_DEFAULT_HOST, ConfigError, Settings


def env(corpus: Path | str, port: str = "8080", **extra: str) -> dict[str, str]:
    return {"CORPUS_DIR": str(corpus), "PORT": port, **extra}


def test_valid_env_gives_settings(tmp_path: Path) -> None:
    settings = Settings.from_env(env(tmp_path, GEMINI_API_KEY="secret-value"))
    assert settings.corpus_dir == tmp_path.resolve()
    assert settings.port == 8080
    assert settings.gemini_api_key == "secret-value"
    assert settings.ask_deadline_s == ASK_DEADLINE_S
    assert "secret-value" not in repr(settings)


@pytest.mark.parametrize("key", [None, ""])
def test_api_key_is_optional(tmp_path: Path, key: str | None) -> None:
    extra = {} if key is None else {"GEMINI_API_KEY": key}
    assert Settings.from_env(env(tmp_path, **extra)).gemini_api_key is None


def test_relative_corpus_dir_is_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "corpus").mkdir()
    monkeypatch.chdir(tmp_path)
    assert Settings.from_env(env("corpus")).corpus_dir == (tmp_path / "corpus").resolve()


def test_missing_corpus_dir_is_a_clear_error() -> None:
    with pytest.raises(ConfigError, match="CORPUS_DIR is not set"):
        Settings.from_env({"PORT": "8080"})


@pytest.mark.parametrize("name", ["", "does-not-exist", "a-file.json"])
def test_corpus_dir_must_be_an_existing_directory(tmp_path: Path, name: str) -> None:
    (tmp_path / "a-file.json").write_text("{}", encoding="utf-8")
    raw = "" if name == "" else str(tmp_path / name)
    with pytest.raises(ConfigError, match="CORPUS_DIR"):
        Settings.from_env(env(raw))


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read anything")
def test_unreadable_corpus_dir_is_a_clear_error(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    locked.mkdir(mode=0o000)
    try:
        with pytest.raises(ConfigError, match="not readable"):
            Settings.from_env(env(locked))
    finally:
        locked.chmod(0o755)


def test_missing_port_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="PORT is not set"):
        Settings.from_env({"CORPUS_DIR": str(tmp_path)})


@pytest.mark.parametrize("port", ["", "http", "80.5", "0", "-1", "65536"])
def test_invalid_port_is_a_clear_error(tmp_path: Path, port: str) -> None:
    with pytest.raises(ConfigError, match="PORT"):
        Settings.from_env(env(tmp_path, port=port))


# ---------------------------------------------------------------- optional Langfuse settings


def test_langfuse_is_off_by_default_like_at_grading(tmp_path: Path) -> None:
    settings = Settings.from_env(env(tmp_path))
    assert settings.langfuse_public_key is None
    assert settings.langfuse_secret_key is None
    assert settings.langfuse_host == LANGFUSE_DEFAULT_HOST


@pytest.mark.parametrize("value", ["", None])
def test_empty_langfuse_keys_count_as_unset(tmp_path: Path, value: str | None) -> None:
    extra = {} if value is None else {"LANGFUSE_PUBLIC_KEY": value, "LANGFUSE_SECRET_KEY": value}
    settings = Settings.from_env(env(tmp_path, **extra))
    assert (settings.langfuse_public_key, settings.langfuse_secret_key) == (None, None)


def test_langfuse_keys_are_read_and_kept_out_of_repr(tmp_path: Path) -> None:
    public, secret = "fake-public-value", "fake-secret-value"
    settings = Settings.from_env(
        env(tmp_path, LANGFUSE_PUBLIC_KEY=public, LANGFUSE_SECRET_KEY=secret)
    )
    assert (settings.langfuse_public_key, settings.langfuse_secret_key) == (public, secret)
    assert public not in repr(settings)
    assert secret not in repr(settings)


@pytest.mark.parametrize(
    ("extra", "host"),
    [
        ({"LANGFUSE_HOST": "https://eu.example.test/"}, "https://eu.example.test"),
        ({"LANGFUSE_BASE_URL": "http://localhost:3000"}, "http://localhost:3000"),
        (
            {"LANGFUSE_HOST": "https://old.example.test", "LANGFUSE_BASE_URL": "https://new.test"},
            "https://new.test",
        ),
        ({"LANGFUSE_HOST": ""}, LANGFUSE_DEFAULT_HOST),
    ],
    ids=["host", "base-url", "base-url-wins", "empty"],
)
def test_langfuse_host_comes_from_base_url_or_host(
    tmp_path: Path, extra: dict[str, str], host: str
) -> None:
    assert Settings.from_env(env(tmp_path, **extra)).langfuse_host == host
