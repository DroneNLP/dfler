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

def test_pipeline(evidence_dir, output_dir):
    with patch("sys.argv", ["dfler", "--evidence", str(evidence_dir), "--output", str(output_dir), "--model", "dummy_model"]):
        with patch("dfler.dfler.check_evidence", return_value=True) as mock_check, \
             patch("dfler.dfler.construct_timeline", return_value=True) as mock_timeline, \
             patch("dfler.dfler.run_ner", return_value=True) as mock_ner, \
             patch("dfler.dfler.run_report", return_value=True) as mock_report:
            main()
            mock_check.assert_called_once()
            mock_timeline.assert_called_once()
            mock_ner.assert_called_once()
            mock_report.assert_called_once()
