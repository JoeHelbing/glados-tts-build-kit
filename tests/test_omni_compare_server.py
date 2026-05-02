from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "12_omni_compare_server.py"


def load_module(checkpoint_root: Path | None = None):
    spec = importlib.util.spec_from_file_location("omni_compare_server", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    env = {}
    if checkpoint_root is not None:
        env["OMNI_COMPARE_CHECKPOINT_ROOT"] = str(checkpoint_root)
    with patch.dict(os.environ, env):
        spec.loader.exec_module(module)
    return module


class OmniCompareServerTest(unittest.TestCase):
    def test_existing_comparison_matrix_has_five_prompts_and_selected_models(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            module = load_module(Path(tmp_dir) / "checkpoints")

        matrix = module.comparison_matrix()

        self.assertEqual(
            [row["id"] for row in matrix],
            [
                "prompt-01",
                "prompt-02",
                "prompt-03",
                "prompt-04",
                "prompt-05",
            ],
        )
        self.assertEqual(
            sorted(matrix[0]["samples"]),
            [
                "checkpoint-1500",
                "checkpoint-2000",
                "checkpoint-2500",
            ],
        )
        self.assertEqual(
            matrix[0]["samples"]["checkpoint-1500"]["url"],
            "/samples/omni_compare/checkpoint-1500_prompt-01.wav",
        )

    def test_selects_best_eval_checkpoint_neighbors_and_pinned_steps(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for step in (500, 1000, 1500, 2000, 2500, 5000):
                (root / f"checkpoint-{step}").mkdir()
            (root / "train.log").write_text(
                "\n".join(
                    [
                        "Running evaluation at step 500...",
                        "Eval Loss: 4.1",
                        "Running evaluation at step 1000...",
                        "Eval Loss: 3.9",
                        "Running evaluation at step 1500...",
                        "Eval Loss: 3.7",
                        "Running evaluation at step 2000...",
                        "Eval Loss: 3.8",
                        "Running evaluation at step 2500...",
                        "Eval Loss: 4.0",
                        "Running evaluation at step 5000...",
                        "Eval Loss: 4.4",
                    ]
                ),
                encoding="utf-8",
            )

            selected = module.select_checkpoints(root)

        self.assertEqual(
            [checkpoint["name"] for checkpoint in selected],
            [
                "checkpoint-1000",
                "checkpoint-1500",
                "checkpoint-2000",
                "checkpoint-5000",
            ],
        )
        self.assertTrue(selected[1]["is_best_eval"])
        self.assertEqual(selected[1]["eval_loss"], 3.7)

    def test_build_infer_command_matches_omni_voice_contract(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            module = load_module(Path(tmp_dir) / "checkpoints")
            out_path = Path(tmp_dir) / "sample.wav"

            command = module.build_infer_command(
                checkpoint="checkpoint-2000",
                text="Testing. Please remain still.",
                output_path=out_path,
                seed=123,
                speed=1.15,
            )

        self.assertEqual(command[:2], [str(Path.home() / "git" / "OmniVoice" / ".venv" / "bin" / "python"), "-c"])
        self.assertIn("fix_random_seed(123)", command[2])
        self.assertIn("omnivoice.cli.infer", command[2])
        self.assertEqual(
            command[command.index("--model") + 1],
            str(module.CHECKPOINT_ROOT / "checkpoint-2000"),
        )
        self.assertEqual(
            command[command.index("--text") + 1],
            "Testing. Please remain still.",
        )
        self.assertEqual(command[command.index("--output") + 1], str(out_path))
        self.assertEqual(
            command[command.index("--ref_audio") + 1],
            str(ROOT / "data" / "pcm" / "glados" / "a2_triple_laser01.wav"),
        )
        self.assertEqual(command[command.index("--num_step") + 1], "32")
        self.assertEqual(command[command.index("--guidance_scale") + 1], "1.5")
        self.assertEqual(command[command.index("--speed") + 1], "1.15")

    def test_generation_payload_parsers_validate_seeds_checkpoints_and_speed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            module = load_module(Path(tmp_dir) / "checkpoints")

        self.assertEqual(module.parse_seed_list("42, 43 44,42"), [42, 43, 44])
        self.assertEqual(
            module.parse_checkpoint_list(["checkpoint-1500", "checkpoint-2500"]),
            ["checkpoint-1500", "checkpoint-2500"],
        )
        self.assertEqual(module.parse_speed("1.2"), 1.2)

        with self.assertRaises(ValueError):
            module.parse_seed_list("-1")
        with self.assertRaises(ValueError):
            module.parse_checkpoint_list(["checkpoint-9999"])
        with self.assertRaises(ValueError):
            module.parse_speed("2.0")

    def test_eval_records_validate_and_summarize_by_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            module = load_module(Path(tmp_dir) / "checkpoints")
            eval_path = Path(tmp_dir) / "eval.jsonl"
            module.write_eval_record(
                eval_path,
                {
                    "sample_id": "prompt-01",
                    "checkpoint": "checkpoint-1500",
                    "glados": 5,
                    "clean": 4,
                    "artifacts": 1,
                    "notes": "best so far",
                },
            )
            module.write_eval_record(
                eval_path,
                {
                    "sample_id": "prompt-02",
                    "checkpoint": "checkpoint-1500",
                    "glados": 3,
                    "clean": 5,
                    "artifacts": 2,
                    "notes": "",
                },
            )
            module.write_eval_record(
                eval_path,
                {
                    "sample_id": "prompt-01",
                    "checkpoint": "checkpoint-2000",
                    "glados": 2,
                    "clean": 2,
                    "artifacts": 4,
                    "notes": "rough",
                },
            )

            records = module.read_eval_records(eval_path)
            latest = module.latest_eval_by_sample(records)
            summary = module.eval_summary(records)

        self.assertEqual(len(records), 3)
        self.assertEqual(
            latest["prompt-01::checkpoint-1500"]["notes"],
            "best so far",
        )
        self.assertEqual(summary["checkpoint-1500"]["count"], 2)
        self.assertEqual(summary["checkpoint-1500"]["glados_avg"], 4.0)
        self.assertEqual(summary["checkpoint-1500"]["clean_avg"], 4.5)
        self.assertEqual(summary["checkpoint-1500"]["artifacts_avg"], 1.5)

    def test_eval_record_rejects_out_of_range_scores(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            module = load_module(Path(tmp_dir) / "checkpoints")
            eval_path = Path(tmp_dir) / "eval.jsonl"
            with self.assertRaises(ValueError):
                module.write_eval_record(
                    eval_path,
                    {
                        "sample_id": "prompt-01",
                        "checkpoint": "checkpoint-1500",
                        "glados": 6,
                        "clean": 4,
                        "artifacts": 1,
                        "notes": "",
                    },
                )

            self.assertFalse(eval_path.exists())


if __name__ == "__main__":
    unittest.main()
