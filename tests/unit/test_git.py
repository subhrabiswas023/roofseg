from unittest.mock import patch

import pytest
import scripts.kaggle.git as git

class TestAssertCleanRepo:
    def test_when_clean(self):
        with patch.object(git, git.get_status.__name__) as mock:
            mock.return_value = ""
            
            git.assert_clean_repo()
            
    def test_when_dirty(self):
        with patch.object(git, git.get_status.__name__) as mock:
            mock.return_value = "M file.py"
            
            with pytest.raises(RuntimeError):
                git.assert_clean_repo()
                
def test_git_stamp():
    with patch.object(git, git.get_branch.__name__) as mock_brancn, patch.object(git, git.get_commit.__name__) as mock_commit, patch.object(git, git.get_description.__name__) as mock_description:
        mock_brancn.return_value = "branch"
        mock_commit.return_value = "commit"
        mock_description.return_value = "description"
        
        stamp = git.GitStamp.capture()
        assert stamp.branch == "branch"
        assert stamp.commit == "commit"
        assert stamp.description == "description"
            