def test_nothing_in_particular():
    from pathlib import Path
    current_dir = Path.cwd()
    print('hello world!')

    assert 2 + 2 != 5

def test_that_broken_means_broken():
    assert 4 == 2