# File role: Load and validate per-stage OpenAI-compatible model routing from TOML.
# Relation: Used by STORMWikiLMConfigs and CollaborativeStormLMConfigs to initialize LMs without hardcoded providers.
import os
from pathlib import Path
from typing import Any, Dict

import toml
from .lm_routing_spec import SECTION_TO_ROLES, SECTION_DEFAULT_MAX_TOKENS

_REQUIRED_FIELDS = ("url", "key", "model")


def _normalize_model_name(model_name: str) -> str:
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Field 'model' must be a non-empty string.")
    model_name = model_name.strip()
    if model_name.startswith("openai/"):
        return model_name
    return f"openai/{model_name}"


def _load_toml(config_path: str) -> dict:
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"LM routing config file not found: {path}")

    try:
        return toml.load(path)
    except toml.TomlDecodeError as exc:
        raise ValueError(f"Invalid TOML in LM routing config {path}: {exc}") from exc


def _resolve_api_key(env_var_name: str, section: str, role: str) -> str:
    if not isinstance(env_var_name, str) or not env_var_name.strip():
        raise ValueError(
            f"[{section}.{role}] field 'key' must be a non-empty environment variable name."
        )

    env_var_name = env_var_name.strip()
    api_key = os.getenv(env_var_name)
    if not api_key:
        raise ValueError(
            f"Environment variable '{env_var_name}' is not set for [{section}.{role}]."
        )
    return api_key


def _require_dict(value: Any, error_message: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(error_message)
    return value


def _to_positive_int(value: Any, field_name: str, section: str, role: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"[{section}.{role}] field '{field_name}' must be a positive integer."
        )
    return value


def _create_litellm_model(**kwargs) -> Any:
    # Import lazily so this module can still be imported in environments
    # where runtime LM dependencies are not installed yet.
    from .lm import LitellmModel

    return LitellmModel(**kwargs)


def _build_lm_for_role(
    role: str,
    role_config: dict,
    section: str,
    defaults: dict,
    default_max_tokens: int,
) -> Any:
    for field in _REQUIRED_FIELDS:
        if field not in role_config:
            raise ValueError(f"[{section}.{role}] missing required field '{field}'.")

    url = role_config["url"]
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"[{section}.{role}] field 'url' must be a non-empty string.")

    model = _normalize_model_name(role_config["model"])
    api_key = _resolve_api_key(role_config["key"], section=section, role=role)

    max_tokens = role_config.get("max_tokens", default_max_tokens)
    max_tokens = _to_positive_int(max_tokens, "max_tokens", section, role)

    temperature = role_config.get("temperature", defaults.get("temperature", 1.0))
    top_p = role_config.get("top_p", defaults.get("top_p", 0.9))

    extra_kwargs = {
        key: value
        for key, value in role_config.items()
        if key
        not in {
            "url",
            "key",
            "model",
            "max_tokens",
            "temperature",
            "top_p",
        }
    }

    return _create_litellm_model(
        model=model,
        max_tokens=max_tokens,
        api_base=url.strip(),
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        **extra_kwargs,
    )


def build_lm_models_from_toml(config_path: str, section: str) -> Dict[str, Any]:
    if section not in SECTION_TO_ROLES:
        supported = ", ".join(sorted(SECTION_TO_ROLES.keys()))
        raise ValueError(f"Unsupported section '{section}'. Supported: {supported}.")

    config_data = _load_toml(config_path)
    defaults = _require_dict(
        config_data.get("defaults", {}),
        "[defaults] must be a TOML table.",
    )

    section_config = _require_dict(
        config_data.get(section),
        f"Missing section '[{section}]' in LM routing config.",
    )

    role_to_model = {}
    for role in SECTION_TO_ROLES[section]:
        role_config = _require_dict(
            section_config.get(role),
            f"Missing section '[{section}.{role}]' in LM routing config.",
        )
        role_to_model[role] = _build_lm_for_role(
            role=role,
            role_config=role_config,
            section=section,
            defaults=defaults,
            default_max_tokens=SECTION_DEFAULT_MAX_TOKENS[section][role],
        )

    return role_to_model


def apply_lm_models_from_toml(config_obj: Any, config_path: str, section: str) -> None:
    role_to_model = build_lm_models_from_toml(config_path=config_path, section=section)
    for role, model in role_to_model.items():
        if not hasattr(config_obj, role):
            raise ValueError(
                f"Object '{type(config_obj).__name__}' does not have expected role '{role}'."
            )
        setattr(config_obj, role, model)
