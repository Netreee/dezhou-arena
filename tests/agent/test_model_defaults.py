"""Future experiment defaults must never inherit an expensive global model."""

from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from poker.agent.language import ModelRequest
from poker.agent.naive_language import CodexLanguageModel
from scripts import run_agent_overnight, verify_six_agents
from tests.agent.test_naive_language import CHECK, FakePopen, Scenario


class ModelDefaultsTests(TestCase):
    def test_default_request_ignores_global_model_and_pins_low_reasoning(self) -> None:
        with TemporaryDirectory() as directory:
            Path(directory, "config.toml").write_text('model = "gpt-6-astra"\n', encoding="utf-8")
            provider = FakePopen(Scenario(answer=CHECK))
            with patch.dict(os.environ, {"CODEX_HOME": directory}), patch(
                "poker.agent.naive_language.subprocess.Popen", provider
            ):
                model = CodexLanguageModel(executable=Path("fixture-codex.exe"))
                model.complete(ModelRequest("Choose check.", {"revision": 1}, (), (), 3.0))
            command = provider.processes[0].command
            self.assertEqual(command[command.index("--model") + 1], "gpt-6-luna")
            self.assertIn('model_reasoning_effort="low"', command)
            self.assertIn("--ignore-user-config", command)
            self.assertEqual(len(provider.processes), 1)

    def test_explicit_model_is_preserved_and_empty_model_is_rejected(self) -> None:
        model = CodexLanguageModel(model="explicit-model", executable=Path("fixture-codex.exe"))
        self.assertEqual(model.model, "explicit-model")
        for value in ("", " ", "bad\nmodel"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CodexLanguageModel(model=value, executable=Path("fixture-codex.exe"))

    def test_six_agent_cli_passes_project_default_without_launching_a_provider(self) -> None:
        args = ["verify_six_agents.py", "--output-dir", "unused-test-output", "--language-backend", "codex"]
        with patch("sys.argv", args), patch.object(verify_six_agents, "verify", return_value={"ok": True}) as verify:
            with redirect_stdout(StringIO()), self.assertRaises(SystemExit) as stopped:
                verify_six_agents.main()
        self.assertEqual(stopped.exception.code, 0)
        self.assertEqual(verify.call_args.kwargs["language_model"], "gpt-6-luna")

    def test_overnight_cli_records_and_forwards_the_project_default(self) -> None:
        args = ["run_agent_overnight.py", "--output-dir", "unused-test-output", "--until-utc", "2026-10-09T00:00:00Z"]
        with patch("sys.argv", args), patch.object(
            run_agent_overnight, "run_supervisor", return_value={"overall_status": "completed"}
        ) as supervisor:
            with redirect_stdout(StringIO()), self.assertRaises(SystemExit) as stopped:
                run_agent_overnight.main()
        self.assertEqual(stopped.exception.code, 0)
        config = supervisor.call_args.args[0]
        self.assertEqual(config.language_model, "gpt-6-luna")
        command = run_agent_overnight.driver_command(
            run_agent_overnight.plan_round(0, config), Path("unused-evidence"), 8, 90, config
        )
        self.assertEqual(command[command.index("--language-model") + 1], "gpt-6-luna")
