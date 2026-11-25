"""Catch broken local imports after moving reusable code into the package."""

import ast
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
OLD_TOP_LEVEL = {
    "cartesian_impedance", "direct_rl_env", "effort_control", "grasp_diagnostics",
    "grasp_gif", "grasp_rewards", "grasp_scene_cfg", "grasp_task", "smoke_direct_rl",
}


class ScriptImportTests(unittest.TestCase):
    def test_local_package_imports_resolve(self):
        for source in SCRIPTS.rglob("*.py"):
            with self.subTest(source=source.relative_to(SCRIPTS)):
                tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
                for node in ast.walk(tree):
                    if not isinstance(node, ast.ImportFrom) or not node.module:
                        continue
                    self.assertNotIn(node.module, OLD_TOP_LEVEL)
                    if not node.module.startswith("franka_impedance"):
                        continue
                    base = SCRIPTS.joinpath(*node.module.split("."))
                    self.assertTrue(
                        base.with_suffix(".py").is_file() or (base / "__init__.py").is_file(),
                        f"{source}: unresolved local import {node.module}",
                    )


if __name__ == "__main__":
    unittest.main()
