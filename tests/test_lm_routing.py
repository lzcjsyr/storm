import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import toml


class FakeLitellmModel:
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs


class TestLMRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[1]
        package_root = repo_root / "knowledge_storm"

        pkg = types.ModuleType("knowledge_storm")
        pkg.__path__ = [str(package_root)]
        sys.modules["knowledge_storm"] = pkg

        spec_module_path = package_root / "lm_routing_spec.py"
        spec_spec = importlib.util.spec_from_file_location(
            "knowledge_storm.lm_routing_spec", spec_module_path
        )
        spec_module = importlib.util.module_from_spec(spec_spec)
        sys.modules["knowledge_storm.lm_routing_spec"] = spec_module
        spec_spec.loader.exec_module(spec_module)

        routing_module_path = package_root / "lm_routing.py"
        routing_spec = importlib.util.spec_from_file_location(
            "knowledge_storm.lm_routing", routing_module_path
        )
        cls.lm_routing = importlib.util.module_from_spec(routing_spec)
        sys.modules["knowledge_storm.lm_routing"] = cls.lm_routing
        routing_spec.loader.exec_module(cls.lm_routing)

    def setUp(self):
        self._original_env = {
            "TEST_API_KEY": os.environ.get("TEST_API_KEY"),
            "ALT_TEST_API_KEY": os.environ.get("ALT_TEST_API_KEY"),
            "UNSET_TEST_API_KEY": os.environ.get("UNSET_TEST_API_KEY"),
        }
        os.environ["TEST_API_KEY"] = "test-key"
        os.environ["ALT_TEST_API_KEY"] = "alt-test-key"
        os.environ.pop("UNSET_TEST_API_KEY", None)

    def tearDown(self):
        for key, value in self._original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _write_config(self, config_data):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False)
        try:
            toml.dump(config_data, tmp)
            return tmp.name
        finally:
            tmp.close()

    def _base_config(self):
        return {
            "defaults": {"temperature": 0.5, "top_p": 0.9},
            "storm_wiki": {
                "conv_simulator_lm": {
                    "url": "https://api.example.com/v1",
                    "key": "TEST_API_KEY",
                    "model": "demo-conv",
                    "max_tokens": 500,
                },
                "question_asker_lm": {
                    "url": "https://api.example.com/v1",
                    "key": "TEST_API_KEY",
                    "model": "demo-ask",
                    "max_tokens": 500,
                },
                "outline_gen_lm": {
                    "url": "https://api.example.com/v1",
                    "key": "TEST_API_KEY",
                    "model": "demo-outline",
                    "max_tokens": 400,
                },
                "article_gen_lm": {
                    "url": "https://api.example.com/v1",
                    "key": "TEST_API_KEY",
                    "model": "demo-article",
                    "max_tokens": 700,
                },
                "article_polish_lm": {
                    "url": "https://api.example.com/v1",
                    "key": "TEST_API_KEY",
                    "model": "demo-polish",
                    "max_tokens": 4000,
                },
            },
            "co_storm": {
                "question_answering_lm": {
                    "url": "https://api.alt-example.com/v1",
                    "key": "ALT_TEST_API_KEY",
                    "model": "demo-qa",
                    "max_tokens": 1000,
                },
                "discourse_manage_lm": {
                    "url": "https://api.alt-example.com/v1",
                    "key": "ALT_TEST_API_KEY",
                    "model": "demo-discourse",
                    "max_tokens": 500,
                },
                "utterance_polishing_lm": {
                    "url": "https://api.alt-example.com/v1",
                    "key": "ALT_TEST_API_KEY",
                    "model": "demo-utterance",
                    "max_tokens": 2000,
                },
                "warmstart_outline_gen_lm": {
                    "url": "https://api.alt-example.com/v1",
                    "key": "ALT_TEST_API_KEY",
                    "model": "demo-warmstart",
                    "max_tokens": 500,
                },
                "question_asking_lm": {
                    "url": "https://api.alt-example.com/v1",
                    "key": "ALT_TEST_API_KEY",
                    "model": "demo-question",
                    "max_tokens": 300,
                },
                "knowledge_base_lm": {
                    "url": "https://api.alt-example.com/v1",
                    "key": "ALT_TEST_API_KEY",
                    "model": "demo-kb",
                    "max_tokens": 1000,
                },
            },
        }

    def test_storm_wiki_all_stages_init_success(self):
        config_path = self._write_config(self._base_config())
        cfg = types.SimpleNamespace(
            conv_simulator_lm=None,
            question_asker_lm=None,
            outline_gen_lm=None,
            article_gen_lm=None,
            article_polish_lm=None,
        )
        try:
            with patch.object(
                self.lm_routing,
                "_create_litellm_model",
                side_effect=lambda **kwargs: FakeLitellmModel(**kwargs),
            ):
                self.lm_routing.apply_lm_models_from_toml(
                    config_obj=cfg,
                    config_path=config_path,
                    section="storm_wiki",
                )
            self.assertEqual(cfg.article_gen_lm.model, "openai/demo-article")
        finally:
            os.remove(config_path)

    def test_co_storm_all_stages_init_success(self):
        config_path = self._write_config(self._base_config())
        cfg = types.SimpleNamespace(
            question_answering_lm=None,
            discourse_manage_lm=None,
            utterance_polishing_lm=None,
            warmstart_outline_gen_lm=None,
            question_asking_lm=None,
            knowledge_base_lm=None,
        )
        try:
            with patch.object(
                self.lm_routing,
                "_create_litellm_model",
                side_effect=lambda **kwargs: FakeLitellmModel(**kwargs),
            ):
                self.lm_routing.apply_lm_models_from_toml(
                    config_obj=cfg,
                    config_path=config_path,
                    section="co_storm",
                )
            self.assertEqual(cfg.question_answering_lm.model, "openai/demo-qa")
        finally:
            os.remove(config_path)

    def test_missing_required_field_raises(self):
        config = self._base_config()
        del config["storm_wiki"]["article_gen_lm"]["model"]
        config_path = self._write_config(config)
        try:
            with patch.object(
                self.lm_routing,
                "_create_litellm_model",
                side_effect=lambda **kwargs: FakeLitellmModel(**kwargs),
            ):
                with self.assertRaises(ValueError) as ctx:
                    self.lm_routing.build_lm_models_from_toml(
                        config_path=config_path,
                        section="storm_wiki",
                    )
            self.assertIn("[storm_wiki.article_gen_lm]", str(ctx.exception))
            self.assertIn("missing required field 'model'", str(ctx.exception))
        finally:
            os.remove(config_path)

    def test_missing_env_var_raises(self):
        config = self._base_config()
        config["storm_wiki"]["conv_simulator_lm"]["key"] = "UNSET_TEST_API_KEY"
        config_path = self._write_config(config)
        try:
            with patch.object(
                self.lm_routing,
                "_create_litellm_model",
                side_effect=lambda **kwargs: FakeLitellmModel(**kwargs),
            ):
                with self.assertRaises(ValueError) as ctx:
                    self.lm_routing.build_lm_models_from_toml(
                        config_path=config_path,
                        section="storm_wiki",
                    )
            self.assertIn("UNSET_TEST_API_KEY", str(ctx.exception))
        finally:
            os.remove(config_path)

    def test_default_max_tokens_fallback(self):
        config = self._base_config()
        del config["co_storm"]["knowledge_base_lm"]["max_tokens"]
        config_path = self._write_config(config)
        try:
            with patch.object(
                self.lm_routing,
                "_create_litellm_model",
                side_effect=lambda **kwargs: FakeLitellmModel(**kwargs),
            ):
                role_to_model = self.lm_routing.build_lm_models_from_toml(
                    config_path=config_path,
                    section="co_storm",
                )
            self.assertEqual(
                role_to_model["knowledge_base_lm"].kwargs["max_tokens"],
                self.lm_routing.SECTION_DEFAULT_MAX_TOKENS["co_storm"][
                    "knowledge_base_lm"
                ],
            )
        finally:
            os.remove(config_path)

    def test_model_normalization(self):
        config = self._base_config()
        config["storm_wiki"]["outline_gen_lm"]["model"] = "deepseek-chat"
        config["storm_wiki"]["article_gen_lm"]["model"] = "openai/already-prefixed"
        config_path = self._write_config(config)
        try:
            with patch.object(
                self.lm_routing,
                "_create_litellm_model",
                side_effect=lambda **kwargs: FakeLitellmModel(**kwargs),
            ):
                role_to_model = self.lm_routing.build_lm_models_from_toml(
                    config_path=config_path,
                    section="storm_wiki",
                )
            self.assertEqual(role_to_model["outline_gen_lm"].model, "openai/deepseek-chat")
            self.assertEqual(
                role_to_model["article_gen_lm"].model,
                "openai/already-prefixed",
            )
        finally:
            os.remove(config_path)

    def test_extra_kwargs_are_passed_to_model(self):
        config = self._base_config()
        config["storm_wiki"]["outline_gen_lm"]["response_format"] = {
            "type": "json_schema"
        }
        config_path = self._write_config(config)
        try:
            with patch.object(
                self.lm_routing,
                "_create_litellm_model",
                side_effect=lambda **kwargs: FakeLitellmModel(**kwargs),
            ):
                role_to_model = self.lm_routing.build_lm_models_from_toml(
                    config_path=config_path,
                    section="storm_wiki",
                )
            self.assertEqual(
                role_to_model["outline_gen_lm"].kwargs["response_format"],
                {"type": "json_schema"},
            )
        finally:
            os.remove(config_path)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            self.lm_routing.build_lm_models_from_toml(
                config_path="/tmp/does-not-exist-lm-routing.toml",
                section="storm_wiki",
            )
        self.assertIn("LM routing config file not found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
