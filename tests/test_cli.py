import pytest
import os
import shutil
from dfler.dfler import main
from unittest.mock import patch, MagicMock

@pytest.fixture
def evidence_dir(tmp_path):
    d = tmp_path / "evidence"
    d.mkdir()
    (d / "android").mkdir()
    (d / "android" / "log.txt").write_text("dummy log")
    return d

@pytest.fixture
def output_dir(tmp_path):
    d = tmp_path / "output"
    d.mkdir()
    return d

def test_check_command(evidence_dir, output_dir):
    with patch("sys.argv", ["dfler", "check", "--evidence", str(evidence_dir), "--output", str(output_dir)]):
        with patch("dfler.dfler.check_evidence") as mock_check:
            main()
            mock_check.assert_called_once()

def test_timeline_command(output_dir):
    with patch("sys.argv", ["dfler", "timeline", "--output", str(output_dir)]):
        with patch("dfler.dfler.construct_timeline") as mock_timeline:
            main()
            mock_timeline.assert_called_once()

def test_ner_command(output_dir):
    with patch("sys.argv", ["dfler", "ner", "--output", str(output_dir), "--model", "dummy_model"]):
        with patch("dfler.dfler.run_ner") as mock_ner:
            main()
            mock_ner.assert_called_once()

def test_report_command(output_dir):
    with patch("sys.argv", ["dfler", "report", "--output", str(output_dir)]):
        with patch("dfler.dfler.run_report") as mock_report:
            main()
            mock_report.assert_called_once()
