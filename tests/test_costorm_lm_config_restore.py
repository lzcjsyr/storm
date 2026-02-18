import importlib.util
import sys
import types
import unittest
from pathlib import Path


class FakeLitellmModel:
    def __init__(self, model, model_type="chat", **kwargs):
        self.model = model
        self.model_type = model_type
        self.kwargs = kwargs


def _ensure_module(name: str):
    module = types.ModuleType(name)
    sys.modules[name] = module
    return module


def _load_engine_module():
    repo_root = Path(__file__).resolve().parents[1]
    package_root = repo_root / "knowledge_storm"

    pkg = _ensure_module("knowledge_storm")
    pkg.__path__ = [str(package_root)]

    cs_pkg = _ensure_module("knowledge_storm.collaborative_storm")
    cs_pkg.__path__ = [str(package_root / "collaborative_storm")]

    modules_pkg = _ensure_module("knowledge_storm.collaborative_storm.modules")
    modules_pkg.__path__ = [str(package_root / "collaborative_storm" / "modules")]

    _ensure_module("knowledge_storm.collaborative_storm.modules.collaborative_storm_utils")

    callback_mod = _ensure_module("knowledge_storm.collaborative_storm.modules.callback")
    callback_mod.BaseCallbackHandler = type("BaseCallbackHandler", (), {})

    agents_mod = _ensure_module("knowledge_storm.collaborative_storm.modules.co_storm_agents")
    agents_mod.SimulatedUser = type("SimulatedUser", (), {})
    agents_mod.PureRAGAgent = type("PureRAGAgent", (), {})
    agents_mod.Moderator = type("Moderator", (), {})
    agents_mod.CoStormExpert = type("CoStormExpert", (), {})

    expert_mod = _ensure_module("knowledge_storm.collaborative_storm.modules.expert_generation")
    expert_mod.GenerateExpertModule = type("GenerateExpertModule", (), {})

    warm_mod = _ensure_module("knowledge_storm.collaborative_storm.modules.warmstart_hierarchical_chat")
    warm_mod.WarmStartModule = type("WarmStartModule", (), {})

    dataclass_mod = _ensure_module("knowledge_storm.dataclass")
    dataclass_mod.ConversationTurn = type(
        "ConversationTurn",
        (),
        {"from_dict": staticmethod(lambda data: data)},
    )
    dataclass_mod.KnowledgeBase = type(
        "KnowledgeBase",
        (),
        {"from_dict": staticmethod(lambda **kwargs: kwargs)},
    )

    encoder_mod = _ensure_module("knowledge_storm.encoder")
    encoder_mod.Encoder = type("Encoder", (), {})

    interface_mod = _ensure_module("knowledge_storm.interface")
    interface_mod.LMConfigs = type("LMConfigs", (), {})
    interface_mod.Agent = type("Agent", (), {})

    logging_mod = _ensure_module("knowledge_storm.logging_wrapper")
    logging_mod.LoggingWrapper = type("LoggingWrapper", (), {"__init__": lambda self, *a, **k: None})

    lm_mod = _ensure_module("knowledge_storm.lm")
    lm_mod.LitellmModel = FakeLitellmModel

    routing_mod = _ensure_module("knowledge_storm.lm_routing")
    routing_mod.apply_lm_models_from_toml = lambda **kwargs: None

    routing_spec_mod = _ensure_module("knowledge_storm.lm_routing_spec")
    routing_spec_mod.SECTION_TO_ROLES = {
        "co_storm": [
            "question_answering_lm",
            "discourse_manage_lm",
            "utterance_polishing_lm",
            "warmstart_outline_gen_lm",
            "question_asking_lm",
            "knowledge_base_lm",
        ]
    }

    rm_mod = _ensure_module("knowledge_storm.rm")
    rm_mod.BingSearch = type("BingSearch", (), {})

    spec = importlib.util.spec_from_file_location(
        "knowledge_storm.collaborative_storm.engine",
        package_root / "collaborative_storm" / "engine.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["knowledge_storm.collaborative_storm.engine"] = module
    spec.loader.exec_module(module)
    return module


class TestCollaborativeStormLMConfigRestore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = _load_engine_module()
        cls.config_cls = cls.engine.CollaborativeStormLMConfigs

    def _build_config_dict(self):
        return {
            "question_answering_lm": {
                "model": "openai/mock-qa",
                "model_type": "chat",
                "max_tokens": 1000,
                "temperature": 0.5,
                "top_p": 0.9,
                "api_base": "https://api.example.com/v1",
                "api_key": "test-key",
            },
            "discourse_manage_lm": {
                "model": "openai/mock-dm",
                "model_type": "chat",
                "max_tokens": 500,
                "temperature": 0.5,
                "top_p": 0.9,
                "api_base": "https://api.example.com/v1",
                "api_key": "test-key",
            },
            "utterance_polishing_lm": {
                "model": "openai/mock-up",
                "model_type": "chat",
                "max_tokens": 2000,
                "temperature": 0.5,
                "top_p": 0.9,
                "api_base": "https://api.example.com/v1",
                "api_key": "test-key",
            },
            "warmstart_outline_gen_lm": {
                "model": "openai/mock-warm",
                "model_type": "chat",
                "max_tokens": 500,
                "temperature": 0.5,
                "top_p": 0.9,
                "api_base": "https://api.example.com/v1",
                "api_key": "test-key",
            },
            "question_asking_lm": {
                "model": "openai/mock-qask",
                "model_type": "chat",
                "max_tokens": 300,
                "temperature": 0.5,
                "top_p": 0.9,
                "api_base": "https://api.example.com/v1",
                "api_key": "test-key",
            },
            "knowledge_base_lm": {
                "model": "openai/mock-kb",
                "model_type": "chat",
                "max_tokens": 1000,
                "temperature": 0.5,
                "top_p": 0.9,
                "api_base": "https://api.example.com/v1",
                "api_key": "test-key",
            },
        }

    def test_roundtrip_to_dict_from_dict(self):
        config_data = self._build_config_dict()
        restored = self.config_cls.from_dict(config_data)
        roundtrip = restored.to_dict()
        self.assertEqual(roundtrip["question_answering_lm"]["model"], "openai/mock-qa")
        self.assertEqual(roundtrip["knowledge_base_lm"]["max_tokens"], 1000)

    def test_from_dict_missing_roles_raises(self):
        config_data = self._build_config_dict()
        del config_data["knowledge_base_lm"]
        with self.assertRaises(ValueError) as ctx:
            self.config_cls.from_dict(config_data)
        self.assertIn("Missing role definitions", str(ctx.exception))

    def test_from_dict_missing_model_field_raises(self):
        config_data = self._build_config_dict()
        del config_data["question_asking_lm"]["model"]
        with self.assertRaises(ValueError) as ctx:
            self.config_cls.from_dict(config_data)
        self.assertIn("missing required field 'model'", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
