import numpy as np
import datetime


TIME_SCALE = "us"
EPOCH_DTYPE = np.dtype(f"datetime64[{TIME_SCALE}]")


def dt_to_dt64(dt: datetime.datetime) -> np.datetime64:
    dt = dt.replace(tzinfo=None)
    return np.datetime64(
        dt.isoformat(sep="T", timespec="microseconds"), TIME_SCALE
    )

def dt64_to_datetime(dt: np.datetime64) -> datetime.datetime:
    datetime.datetime.fromisoformat(str(dt))
