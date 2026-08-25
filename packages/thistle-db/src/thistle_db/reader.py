import concurrent.futures
import csv
import datetime
import json
import os
import pathlib
from typing import Callable, Iterable, Iterator, TypeVar, Union

from sgp4 import omm as sgp4_omm

PathLike = Union[str, os.PathLike, pathlib.Path]
TLETuple = tuple[str, str]

T = TypeVar("T")

GroupByKey = TypeVar("GroupByKey")


def group_by(
    tles: Iterable[TLETuple], key: Callable[[TLETuple], GroupByKey]
) -> dict[GroupByKey, list[TLETuple]]:
    """Groups input TLEs by values from a callable key."""
    results: dict[GroupByKey, list[TLETuple]] = {}
    for tle in tles:
        group = key(tle)
        if group not in results:
            results[group] = []
        results[group].append(tle)
    return results


def unique(tles: Iterable[T]) -> list[T]:
    """Returns input as a list list ensuring unique entries."""
    return list(dict.fromkeys(tles).keys())


def tle_epoch(tle: TLETuple) -> float:
    """Get the epoch (float) from a TLE, adjusted for Y2K."""
    epoch = float(tle[0][18:32].replace(" ", "0"))
    epoch += 1900_000 if epoch // 1000 >= 57 else 2000_000
    return epoch


def tle_date(tle: TLETuple) -> str:
    """Get the date (as str) from a TLE."""
    epoch = tle_epoch(tle)
    year, doy = divmod(epoch, 1000)
    doy = doy // 1
    dt = datetime.datetime(int(year), 1, 1) + datetime.timedelta(days=int(doy - 1))
    return dt.strftime("%Y%m%d")


def tle_satnum(tle: TLETuple) -> str:
    """Extract the (Alpha-5) Satnum from a TLE."""
    return tle[0][2:7].replace(" ", "0")


def read_tle(
    file: PathLike,
) -> Iterator[TLETuple]:
    """Read a single TLE file, yielding (line1, line2) pairs lazily.

    A generator so that arbitrarily large files (full-catalog restores) can
    be ingested without materializing every record in memory.
    """
    with open(file, "r") as f:
        line1: str | None = None
        for raw in f:
            line = raw.rstrip()
            if not line:
                continue
            # "1 "/"2 " prefixes (line number + mandatory blank), matching
            # the generator's tail guard — a bare "1" would also accept
            # 3LE name lines of satellites whose names start with a digit.
            if line.startswith("1 "):
                line1 = line
            elif line.startswith("2 ") and line1 is not None:
                yield (line1, line)
                line1 = None


def _read_tle_list(file: PathLike) -> list[TLETuple]:
    return list(read_tle(file))


def read_tles(files: Iterable[PathLike]) -> list[TLETuple]:
    """Read multiple TLE files."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(_read_tle_list, file) for file in files]
        tles = []
        for future in futures:
            results = future.result()
            tles.extend(results)
    return tles


def render_tle(tles: Iterable[TLETuple]) -> str:
    """Render element sets as two-line text, ready for a single write.

    One join and one write instead of two `print` calls per record: writing
    a full catalog was 800k `print` calls, all of them buffered churn.
    """
    return "".join(f"{line1}\n{line2}\n" for line1, line2 in tles)


def write_tle(
    file_path: PathLike,
    tles: Iterable[TLETuple],
    *,
    sort: bool = False,
    deduplicate: bool = False,
) -> None:
    if deduplicate:
        tles = unique(list(tles))

    if sort:
        tles = sorted(tles, key=tle_epoch)
        tles = sorted(tles, key=tle_satnum)

    with open(file_path, "w") as f:
        f.write(render_tle(tles))


def write_tles(
    files: dict[pathlib.Path, Iterable[TLETuple]],
    *,
    deduplicate: bool = True,
    sort: bool = False,
) -> None:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures: dict[concurrent.futures.Future, pathlib.Path] = {
            executor.submit(
                write_tle, file, tles, deduplicate=deduplicate, sort=sort
            ): file
            for file, tles in files.items()
        }

        for future in futures:
            future.result()


# --- OMM Format Support ---


def read_omm_json(file: PathLike) -> list[dict]:
    """Read Space-Track JSON format OMM data.

    Handles both single-object (bare dict) and array forms.
    """
    with open(file, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    return data


def read_omm_csv(file: PathLike) -> list[dict]:
    """Read OMM CSV format using sgp4's parser."""
    with open(file, "r") as f:
        return list(sgp4_omm.parse_csv(f))


def read_omm_xml(file: PathLike) -> list[dict]:
    """Read OMM XML format using sgp4's parser."""
    with open(file, "rb") as f:
        return list(sgp4_omm.parse_xml(f))


# Standard OMM CSV field order (matches Space-Track output)
OMM_CSV_FIELDS = [
    "OBJECT_NAME",
    "OBJECT_ID",
    "EPOCH",
    "MEAN_MOTION",
    "ECCENTRICITY",
    "INCLINATION",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    "MEAN_ANOMALY",
    "EPHEMERIS_TYPE",
    "CLASSIFICATION_TYPE",
    "NORAD_CAT_ID",
    "ELEMENT_SET_NO",
    "REV_AT_EPOCH",
    "BSTAR",
    "MEAN_MOTION_DOT",
    "MEAN_MOTION_DDOT",
]


def write_omm_csv(
    file_path: PathLike,
    records: list[dict],
) -> None:
    """Write OMM records as CSV."""
    if not records:
        return
    fieldnames = OMM_CSV_FIELDS
    with open(file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def detect_format(file: PathLike) -> str:
    """Detect file format by extension.

    Returns one of: "tle", "omm_json", "omm_csv", "omm_xml"
    """
    ext = pathlib.Path(file).suffix.lower()
    if ext in (".tle", ".txt", ".3le"):
        return "tle"
    elif ext == ".json":
        return "omm_json"
    elif ext == ".csv":
        return "omm_csv"
    elif ext == ".xml":
        return "omm_xml"
    else:
        return "tle"  # default to TLE format
