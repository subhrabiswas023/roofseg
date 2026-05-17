from subprocess import CalledProcessError
from unittest.mock import patch, Mock

import pytest

import scripts.kaggle.executor as executor

class TestRunCommand:
    def test_stdout(self):
        with patch.object(executor, executor.run.__name__) as mock:
            fake_result = Mock()
            fake_result.stdout = "hello\n"
            fake_result.stderr = ""

            mock.return_value = fake_result

            output = executor.run_command("echo", "hello")

            assert output == "hello"

            mock.assert_called_once_with(
                ("echo", "hello"), check=True, capture_output=True, text=True
            )
    
    def test_stderr(self):
        with patch.object(executor, executor.run.__name__) as mock:
            fake_result = Mock()
            fake_result.stdout = ""
            fake_result.stderr = "stderr message\n"
            
            mock.return_value = fake_result
            
            output = executor.run_command("dummy", "command")
            
            assert output == "stderr message"
            
    def test_runtime_error(self):
        with patch.object(executor, executor.run.__name__) as mock:
            mock.side_effect = CalledProcessError(1, ("dummy", "command"), stderr="fatal error")
            
            with pytest.raises(RuntimeError, match="fatal error"):
                executor.run_command("dummy", "command")
                
    def test_runtime_error_from_unknown_error(self):
        with patch.object(executor, executor.run.__name__) as mock:
            mock.side_effect = CalledProcessError(1, ("dummy", "command"))
            
            with pytest.raises(RuntimeError):
                executor.run_command("dummy", "command")  
            
