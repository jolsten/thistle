from typing import Union

Satnum = Union[str, int]

ALPHA_TO_INT = {
    "A": 10, "J": 18, "S": 26,
    "B": 11, "K": 19, "T": 27,
    "C": 12, "L": 20, "U": 28,
    "D": 13, "M": 21, "V": 29,
    "E": 14, "N": 22, "W": 30,
    "F": 15, "P": 23, "X": 31,
    "G": 16, "Q": 24, "Y": 32,
    "H": 17, "R": 25, "Z": 33,
}
INT_TO_ALPHA = {val: key for key, val in ALPHA_TO_INT.items()}

def to_alpha5(satnum: int) -> str:
    """Encode an integer to an Alpha-5 string."""
    if satnum < 100_000:
        return f'{satnum:05}'
    
    if satnum > 339999:
        msg = "satnum exceeds maximum value for Alpha-5 encoding 339999 (encoded as Z9999)"
        raise ValueError(msg)
    
    a, b = divmod(satnum, 10_000)
    return f"{INT_TO_ALPHA[a]}{b:04}"

def from_alpha5(satnum: str) -> int:
    """Decode an Alpha-5 string to an integer."""
    satnum = str(satnum)
    if satnum[0].isnumeric():
        return int(satnum)
    return ALPHA_TO_INT[satnum[0]] * 10_000 + int(satnum[1:])
