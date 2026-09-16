from copy import deepcopy

import pytest

from scripts.reproduce_published_results import (
    EXTERNAL, FACTORIAL, ROOT, read_rows, validate_external, validate_factorial,
)
from scripts.verify_publication import public_violations


def test_published_factorial_arithmetic_and_external_counts():
    validate_factorial(read_rows(ROOT / FACTORIAL))
    validate_external(read_rows(ROOT / EXTERNAL))


def test_external_validation_rejects_omitted_arm_and_wrong_false_positive_cost():
    rows = read_rows(ROOT / EXTERNAL)
    with pytest.raises(ValueError, match="all six"):
        validate_external(rows[:-1])
    changed = deepcopy(rows)
    changed[0]["target_false_positives_per_image"] = "0"
    with pytest.raises(ValueError, match="FP/image"):
        validate_external(changed)


def test_factorial_validation_rejects_wrong_interaction():
    rows = read_rows(ROOT / FACTORIAL)
    rows[-1]["difference_in_differences"] = "0.5"
    with pytest.raises(ValueError, match="difference_in_differences"):
        validate_factorial(rows)


def test_publication_guard_rejects_local_artifacts_and_personal_paths():
    assert public_violations("runs/model.pt", b"weights")
    private = "C:" + "/Users/" + "example/private.json"
    assert public_violations("results/example.json", private.encode())
    assert not public_violations("results/example.json", b'{"path": "artifacts/example.json"}')
    assert public_violations("docs/sample.md", b"Official share token: `abcdefghijklmnop`")
