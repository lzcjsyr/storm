import ast
import unittest
from pathlib import Path


class TestExampleLMConfigRequired(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[1]
        cls.storm_script = repo_root / "examples" / "storm_examples" / "run_storm_wiki_gpt.py"
        cls.costorm_script = repo_root / "examples" / "costorm_examples" / "run_costorm_gpt.py"

    def _parse(self, path: Path):
        return ast.parse(path.read_text())

    def _has_exists_guard(self, tree: ast.AST):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (
                    isinstance(node.func.value, ast.Attribute)
                    and isinstance(node.func.value.value, ast.Name)
                    and node.func.value.value.id == "os"
                    and node.func.value.attr == "path"
                    and node.func.attr == "exists"
                ):
                    return True
        return False

    def _has_init_from_toml_call(self, tree: ast.AST, target_name: str):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr != "init_from_toml":
                    continue
                if isinstance(node.func.value, ast.Name) and node.func.value.id == target_name:
                    return True
        return False

    def test_storm_script_uses_direct_toml_init_without_exists_guard(self):
        tree = self._parse(self.storm_script)
        self.assertTrue(self._has_init_from_toml_call(tree, "lm_configs"))
        self.assertFalse(self._has_exists_guard(tree))

    def test_costorm_script_uses_direct_toml_init_without_exists_guard(self):
        tree = self._parse(self.costorm_script)
        self.assertTrue(self._has_init_from_toml_call(tree, "lm_config"))
        self.assertFalse(self._has_exists_guard(tree))


if __name__ == "__main__":
    unittest.main()
