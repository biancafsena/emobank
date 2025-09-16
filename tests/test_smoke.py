import os
def test_repo_layout():
    assert os.path.exists('src/emobank/api.py')
    assert os.path.exists('scripts/make_data.py')
