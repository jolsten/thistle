import pytest
from thistle.alpha5 import from_alpha5, to_alpha5


@pytest.mark.parametrize(
    "objnum, alpha5",
    [
        (100000, "A0000"),
        (148493, "E8493"),
        (182931, "J2931"),
        (234018, "P4018"),
        (301928, "W1928"),
        (339999, "Z9999"),
    ],
)
class TestAlpha5:
    def test_to_alpha5(self, objnum: int, alpha5: str) -> None:
        assert alpha5 == to_alpha5(objnum)

    def test_from_alpha5(self, objnum: int, alpha5: str) -> None:
        assert from_alpha5(alpha5) == objnum


@pytest.mark.parametrize("satnum", [-1, 999_999])
def test_invalid_satnum(satnum: int):
    with pytest.raises(ValueError):
        to_alpha5(satnum)
