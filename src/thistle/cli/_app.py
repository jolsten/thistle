"""Typer application wiring all thistle subcommands."""

from __future__ import annotations

import json
import math
import pathlib
import sys
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import typer
from sgp4.api import Satrec
from typing_extensions import Annotated

from thistle.cli._helpers import (
    ALL_COLUMNS,
    ALL_GROUPS,
    DEFAULT_COLUMNS,
    MU,
    RE,
    AlignedWriter,
    epoch_to_datetime,
    parse_site,
    parse_tle_epochs,
    read_and_emit_tles,
    warn_if_tty,
)

app = typer.Typer(
    name="thistle",
    help="Satellite orbit propagation and TLE analysis tools.",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


class SwitchStrategy(str, Enum):
    epoch = "epoch"
    midpoint = "midpoint"
    tca = "tca"


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


@app.command()
def inspect(
    file: Annotated[
        Optional[pathlib.Path],
        typer.Argument(help="TLE file path (default: stdin)"),
    ] = None,
    delimiter: Annotated[
        Optional[str],
        typer.Option("-d", "--delimiter", help="Output field delimiter (default: whitespace)"),
    ] = None,
    all_fields: Annotated[
        bool,
        typer.Option("--all", help="Include all fields (classification, ma, bstar)"),
    ] = False,
    header: Annotated[
        bool,
        typer.Option("--header", help="Print column names as first row"),
    ] = False,
    columns: Annotated[
        bool,
        typer.Option("--columns", help="Print column reference and exit"),
    ] = False,
) -> None:
    """Parse TLE files and display orbital parameters as a table."""
    if columns:
        hdr = f"{'#':>3}  {'Name':<8}  Description"
        print(hdr)
        print("-" * len(hdr))
        for num, name, desc in ALL_COLUMNS:
            print(f"{num:>3}  {name:<8}  {desc}")
        raise typer.Exit()

    out_delim = delimiter if delimiter is not None else " "
    col_defs = DEFAULT_COLUMNS
    numeric = [n for _, _, _, n in col_defs]
    writer = AlignedWriter(out_delim, numeric=numeric)
    writer._widths = [max(w, len(name)) for name, w, _, _ in col_defs]

    if header:
        writer.add_row([name for name, _, _, _ in col_defs])

    if file is not None:
        try:
            with open(file) as f:
                read_and_emit_tles(f, writer, include_all=all_fields)
        except FileNotFoundError:
            print(f"Error: file not found: {file}", file=sys.stderr)
            raise typer.Exit(code=2)
    else:
        warn_if_tty(sys.stdin, "inspect")
        read_and_emit_tles(sys.stdin, writer, include_all=all_fields)

    writer.flush()


# ---------------------------------------------------------------------------
# find-tle
# ---------------------------------------------------------------------------


@app.command("find-tle")
def find_tle(
    file: Annotated[pathlib.Path, typer.Argument(help="TLE file path")],
    switch: Annotated[
        SwitchStrategy,
        typer.Option("--switch", help="TLE switching strategy"),
    ] = SwitchStrategy.midpoint,
    unique: Annotated[
        bool,
        typer.Option("--unique", help="Only output each unique TLE once"),
    ] = False,
) -> None:
    """Find the correct TLE for given timestamps read from stdin."""
    from thistle import Propagator, read_tle as read_tle_file

    try:
        tles = read_tle_file(file)
    except FileNotFoundError:
        print(f"Error: TLE file not found: {file}", file=sys.stderr)
        raise typer.Exit(code=2)

    if not tles:
        print(f"Error: no TLEs found in {file}", file=sys.stderr)
        raise typer.Exit(code=2)

    propagator = Propagator(tles, method=switch.value)

    warn_if_tty(sys.stdin, "find-tle")

    seen: Optional[set[tuple[str, str]]] = set() if unique else None

    for raw_line in sys.stdin:
        time_str = raw_line.strip()
        if not time_str:
            continue

        try:
            dt = datetime.fromisoformat(time_str)
        except ValueError:
            print(
                f"Warning: skipping unparseable time: {time_str}",
                file=sys.stderr,
            )
            continue

        line1, line2 = propagator.find_tle(dt)

        if seen is not None:
            key = (line1, line2)
            if key in seen:
                continue
            seen.add(key)

        sys.stdout.write(line1 + "\n" + line2 + "\n")


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


@app.command()
def summary(
    file: Annotated[pathlib.Path, typer.Argument(help="TLE file path")],
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
    gap: Annotated[
        float,
        typer.Option("--gap", help="Gap threshold in days"),
    ] = 2.0,
) -> None:
    """Summarize TLE data quality for a single object."""
    try:
        f = open(file)
    except FileNotFoundError:
        print(f"Error: file not found: {file}", file=sys.stderr)
        raise typer.Exit(code=2)

    with f:
        parsed = parse_tle_epochs(f)

    if not parsed:
        print(f"Error: no TLEs found in {file}", file=sys.stderr)
        raise typer.Exit(code=2)

    epochs = sorted(epoch_to_datetime(s.epochyr, s.epochdays) for _, _, s in parsed)
    first_sat = parsed[0][2]
    satnum = str(first_sat.satnum)
    intl = first_sat.intldesg.strip()

    count = len(epochs)
    first = epochs[0]
    last = epochs[-1]
    span_days = (last - first).total_seconds() / 86400.0

    result: dict = {
        "satnum": satnum,
        "intl": intl,
        "count": count,
        "first_epoch": first.strftime("%Y-%m-%dT%H:%M:%S"),
        "last_epoch": last.strftime("%Y-%m-%dT%H:%M:%S"),
        "span_days": round(span_days, 2),
    }

    if count >= 2:
        intervals = [
            (epochs[i + 1] - epochs[i]).total_seconds() / 86400.0
            for i in range(count - 1)
        ]
        avg_per_day = count / span_days if span_days > 0 else float("inf")
        arr = np.array(intervals)

        gaps = []
        for i, iv in enumerate(intervals):
            if iv > gap:
                gaps.append({
                    "from": epochs[i].strftime("%Y-%m-%dT%H:%M:%S"),
                    "to": epochs[i + 1].strftime("%Y-%m-%dT%H:%M:%S"),
                    "days": round(iv, 2),
                })
        gaps.sort(key=lambda g: g["from"], reverse=True)

        result["avg_per_day"] = round(avg_per_day, 2)
        result["interval"] = {
            "min": round(float(arr.min()), 2),
            "max": round(float(arr.max()), 2),
            "mean": round(float(arr.mean()), 2),
            "median": round(float(np.median(arr)), 2),
        }
        result["gap_threshold_days"] = gap
        result["gaps"] = gaps

    if json_out:
        print(json.dumps(result, indent=2))
    else:
        print(f"Object:     {satnum} ({intl})")
        print(f"TLE count:  {count}")
        print(f"First TLE:  {result['first_epoch']}")
        print(f"Last TLE:   {result['last_epoch']}")
        print(f"Span:       {result['span_days']} days")
        if count >= 2:
            print(f"Avg/day:    {result['avg_per_day']}")
            iv = result["interval"]
            print()
            print("Epoch interval (days):")
            print(f"  min:    {iv['min']}")
            print(f"  max:    {iv['max']}")
            print(f"  mean:   {iv['mean']}")
            print(f"  median: {iv['median']}")
            if result["gaps"]:
                print()
                print(f"Gaps (>{result['gap_threshold_days']} days): {len(result['gaps'])}")
                for g in result["gaps"]:
                    print(f"  {g['from'][:10]} to {g['to'][:10]}  ({g['days']} days)")
            else:
                print()
                print("No gaps detected.")


# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------


@app.command()
def catalog(
    path: Annotated[pathlib.Path, typer.Argument(help="Directory path")],
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
    pattern: Annotated[
        str,
        typer.Option("--pattern", help="Glob patterns for TLE files, comma-separated"),
    ] = "*.tle,*.txt,*.tce",
    gap: Annotated[
        float,
        typer.Option("--gap", help="Gap threshold in days"),
    ] = 2.0,
) -> None:
    """Summarize TLE data across a directory of files."""
    if not path.is_dir():
        print(f"Error: not a directory: {path}", file=sys.stderr)
        raise typer.Exit(code=2)

    from thistle import read_tle as read_tle_file

    patterns = [p.strip() for p in pattern.split(",")]
    files: list[pathlib.Path] = []
    for pat in patterns:
        files.extend(path.rglob(pat))
    files = sorted(set(files))

    if not files:
        print(f"Error: no files matching {pattern} in {path}", file=sys.stderr)
        raise typer.Exit(code=2)

    all_satrecs: list[Satrec] = []
    for fp in files:
        tles = read_tle_file(fp)
        for line1, line2 in tles:
            try:
                all_satrecs.append(Satrec.twoline2rv(line1, line2))
            except Exception:
                pass

    if not all_satrecs:
        print(f"Error: no valid TLEs found in {path}", file=sys.stderr)
        raise typer.Exit(code=2)

    epochs = sorted(epoch_to_datetime(s.epochyr, s.epochdays) for s in all_satrecs)
    satnums = {str(s.satnum) for s in all_satrecs}
    first = epochs[0]
    last = epochs[-1]
    span_days = round((last - first).total_seconds() / 86400.0, 2)
    count = len(epochs)

    result: dict = {
        "directory": str(path),
        "files": len(files),
        "objects": len(satnums),
        "tle_count": count,
        "first_epoch": first.strftime("%Y-%m-%dT%H:%M:%S"),
        "last_epoch": last.strftime("%Y-%m-%dT%H:%M:%S"),
        "span_days": span_days,
    }

    if count >= 2:
        intervals = [
            (epochs[i + 1] - epochs[i]).total_seconds() / 86400.0
            for i in range(count - 1)
        ]
        arr = np.array(intervals)

        gaps = []
        for i, iv in enumerate(intervals):
            if iv > gap:
                gaps.append({
                    "from": epochs[i].strftime("%Y-%m-%dT%H:%M:%S"),
                    "to": epochs[i + 1].strftime("%Y-%m-%dT%H:%M:%S"),
                    "days": round(iv, 2),
                })
        gaps.sort(key=lambda g: g["from"], reverse=True)

        result["interval"] = {
            "min": round(float(arr.min()), 2),
            "max": round(float(arr.max()), 2),
            "mean": round(float(arr.mean()), 2),
            "median": round(float(np.median(arr)), 2),
        }
        result["gap_threshold_days"] = gap
        result["gaps"] = gaps

    if json_out:
        print(json.dumps(result, indent=2))
    else:
        print(f"Directory:  {path}")
        print(f"Files:      {len(files)}")
        print(f"Objects:    {len(satnums)}")
        print(f"TLE count:  {count}")
        print(f"First TLE:  {result['first_epoch']}")
        print(f"Last TLE:   {result['last_epoch']}")
        print(f"Span:       {span_days} days")
        if count >= 2:
            iv = result["interval"]
            print()
            print("Epoch interval (days):")
            print(f"  min:    {iv['min']}")
            print(f"  max:    {iv['max']}")
            print(f"  mean:   {iv['mean']}")
            print(f"  median: {iv['median']}")
            if result["gaps"]:
                print()
                print(f"Gaps (>{result['gap_threshold_days']} days): {len(result['gaps'])}")
                for g in result["gaps"]:
                    print(f"  {g['from'][:10]} to {g['to'][:10]}  ({g['days']} days)")
            else:
                print()
                print("No gaps detected.")


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------


@app.command("filter")
def filter_(
    file: Annotated[
        Optional[pathlib.Path],
        typer.Argument(help="TLE file path (default: stdin)"),
    ] = None,
    after: Annotated[
        Optional[str],
        typer.Option("--after", help="Epoch at or after (ISO 8601)"),
    ] = None,
    before: Annotated[
        Optional[str],
        typer.Option("--before", help="Epoch at or before (ISO 8601)"),
    ] = None,
    satnum: Annotated[
        Optional[str],
        typer.Option("--satnum", help="NORAD catalog numbers, comma-separated"),
    ] = None,
    min_inc: Annotated[Optional[float], typer.Option("--min-inc", help="Min inclination (deg)")] = None,
    max_inc: Annotated[Optional[float], typer.Option("--max-inc", help="Max inclination (deg)")] = None,
    min_ecc: Annotated[Optional[float], typer.Option("--min-ecc", help="Min eccentricity")] = None,
    max_ecc: Annotated[Optional[float], typer.Option("--max-ecc", help="Max eccentricity")] = None,
    min_period: Annotated[Optional[float], typer.Option("--min-period", help="Min period (minutes)")] = None,
    max_period: Annotated[Optional[float], typer.Option("--max-period", help="Max period (minutes)")] = None,
    min_sma: Annotated[Optional[float], typer.Option("--min-sma", help="Min semi-major axis (km)")] = None,
    max_sma: Annotated[Optional[float], typer.Option("--max-sma", help="Max semi-major axis (km)")] = None,
    min_perigee: Annotated[Optional[float], typer.Option("--min-perigee", help="Min perigee altitude (km)")] = None,
    max_perigee: Annotated[Optional[float], typer.Option("--max-perigee", help="Max perigee altitude (km)")] = None,
    min_apogee: Annotated[Optional[float], typer.Option("--min-apogee", help="Min apogee altitude (km)")] = None,
    max_apogee: Annotated[Optional[float], typer.Option("--max-apogee", help="Max apogee altitude (km)")] = None,
    min_revnum: Annotated[Optional[int], typer.Option("--min-revnum", help="Min revolution number")] = None,
    max_revnum: Annotated[Optional[int], typer.Option("--max-revnum", help="Max revolution number")] = None,
) -> None:
    """Filter TLEs by time, satnum, or orbital elements."""
    satnums: Optional[set[int]] = None
    if satnum:
        try:
            satnums = {int(s.strip()) for s in satnum.split(",")}
        except ValueError:
            print(f"Error: invalid --satnum value: {satnum}", file=sys.stderr)
            raise typer.Exit(code=2)

    after_dt: Optional[datetime] = None
    before_dt: Optional[datetime] = None
    if after:
        try:
            after_dt = datetime.fromisoformat(after)
        except ValueError:
            print(f"Error: invalid --after value: {after}", file=sys.stderr)
            raise typer.Exit(code=2)
    if before:
        try:
            before_dt = datetime.fromisoformat(before)
        except ValueError:
            print(f"Error: invalid --before value: {before}", file=sys.stderr)
            raise typer.Exit(code=2)

    def _passes(sat: Satrec) -> bool:
        if satnums is not None and sat.satnum not in satnums:
            return False

        epoch_dt = epoch_to_datetime(sat.epochyr, sat.epochdays)
        if after_dt is not None and epoch_dt < after_dt:
            return False
        if before_dt is not None and epoch_dt > before_dt:
            return False

        inc_deg = math.degrees(sat.inclo)
        if min_inc is not None and inc_deg < min_inc:
            return False
        if max_inc is not None and inc_deg > max_inc:
            return False

        if min_ecc is not None and sat.ecco < min_ecc:
            return False
        if max_ecc is not None and sat.ecco > max_ecc:
            return False

        mean_motion_revday = sat.no_kozai * 1440.0 / (2.0 * math.pi)
        period = 1440.0 / mean_motion_revday if mean_motion_revday > 0 else float("inf")
        if min_period is not None and period < min_period:
            return False
        if max_period is not None and period > max_period:
            return False

        mm_rad_per_sec = sat.no_kozai / 60.0
        sma_km = (MU / (mm_rad_per_sec**2)) ** (1.0 / 3.0) if mm_rad_per_sec > 0 else 0.0
        if min_sma is not None and sma_km < min_sma:
            return False
        if max_sma is not None and sma_km > max_sma:
            return False

        perigee_alt = sma_km * (1 - sat.ecco) - RE
        if min_perigee is not None and perigee_alt < min_perigee:
            return False
        if max_perigee is not None and perigee_alt > max_perigee:
            return False

        apogee_alt = sma_km * (1 + sat.ecco) - RE
        if min_apogee is not None and apogee_alt < min_apogee:
            return False
        if max_apogee is not None and apogee_alt > max_apogee:
            return False

        if min_revnum is not None and sat.revnum < min_revnum:
            return False
        if max_revnum is not None and sat.revnum > max_revnum:
            return False

        return True

    if file is not None:
        try:
            source = open(file)
        except FileNotFoundError:
            print(f"Error: file not found: {file}", file=sys.stderr)
            raise typer.Exit(code=2)
    else:
        warn_if_tty(sys.stdin, "filter")
        source = sys.stdin

    try:
        line1: Optional[str] = None
        for raw_line in source:
            line = raw_line.rstrip()
            if not line:
                continue
            if line[0] == "1":
                line1 = line
            elif line[0] == "2" and line1 is not None:
                try:
                    sat = Satrec.twoline2rv(line1, line)
                    if _passes(sat):
                        sys.stdout.write(line1 + "\n" + line + "\n")
                except Exception:
                    pass
                line1 = None
    finally:
        if source is not sys.stdin:
            source.close()


# ---------------------------------------------------------------------------
# propagate
# ---------------------------------------------------------------------------


@app.command()
def propagate(
    file: Annotated[pathlib.Path, typer.Argument(help="TLE file path")],
    delimiter: Annotated[
        Optional[str],
        typer.Option("-d", "--delimiter", help="Output field delimiter (default: whitespace)"),
    ] = None,
    skip_header: Annotated[
        Optional[int],
        typer.Option("-H", "--skip-header", help="Skip N header lines from stdin"),
    ] = None,
    print_header: Annotated[
        bool,
        typer.Option("--print-header", help="Print column names as first row"),
    ] = False,
    switch: Annotated[
        SwitchStrategy,
        typer.Option("--switch", help="TLE switching strategy"),
    ] = SwitchStrategy.midpoint,
    chunk: Annotated[
        int,
        typer.Option("--chunk", help="Timestamps per batch"),
    ] = 10000,
    align: Annotated[
        bool,
        typer.Option("--align", help="Adaptively align output columns"),
    ] = False,
    eci: Annotated[bool, typer.Option("--eci", help="ECI position/velocity")] = False,
    ecef: Annotated[bool, typer.Option("--ecef", help="ECEF position/velocity")] = False,
    lla: Annotated[bool, typer.Option("--lla", help="Latitude/longitude/altitude")] = False,
    keplerian: Annotated[bool, typer.Option("--keplerian", help="Keplerian elements")] = False,
    equinoctial: Annotated[bool, typer.Option("--equinoctial", help="Equinoctial elements")] = False,
    sunlight: Annotated[bool, typer.Option("--sunlight", help="Sunlight status")] = False,
    beta: Annotated[bool, typer.Option("--beta", help="Beta angle")] = False,
    lst: Annotated[bool, typer.Option("--lst", help="Local sidereal time")] = False,
    mag_enu: Annotated[bool, typer.Option("--mag-enu", help="Magnetic field (ENU)")] = False,
    mag_total: Annotated[bool, typer.Option("--mag-total", help="Magnetic field (total)")] = False,
    mag_ecef: Annotated[bool, typer.Option("--mag-ecef", help="Magnetic field (ECEF)")] = False,
    site: Annotated[
        Optional[list[str]],
        typer.Option("--site", help="Ground site: NAME:LAT:LON[:ALT] (repeatable)"),
    ] = None,
) -> None:
    """Propagate TLEs and generate orbital data from stdin timestamps."""
    flag_map = {
        "eci": eci, "ecef": ecef, "lla": lla, "keplerian": keplerian,
        "equinoctial": equinoctial, "sunlight": sunlight, "beta": beta,
        "lst": lst, "mag_enu": mag_enu, "mag_total": mag_total, "mag_ecef": mag_ecef,
    }
    groups = [g for g in ALL_GROUPS if flag_map[g]]

    if not groups:
        print(
            "Error: no data groups requested (use --eci, --lla, etc.)",
            file=sys.stderr,
        )
        raise typer.Exit(code=2)

    sites_dict: Optional[dict[str, tuple[float, float] | tuple[float, float, float]]] = None
    if site:
        sites_dict = {}
        for s in site:
            try:
                name, coords = parse_site(s)
                sites_dict[name] = coords
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                raise typer.Exit(code=2)

    from thistle import Propagator, generate, read_tle as read_tle_file

    try:
        tles = read_tle_file(file)
    except FileNotFoundError:
        print(f"Error: TLE file not found: {file}", file=sys.stderr)
        raise typer.Exit(code=2)

    if not tles:
        print(f"Error: no TLEs found in {file}", file=sys.stderr)
        raise typer.Exit(code=2)

    propagator = Propagator(tles, method=switch.value)

    out_delim = delimiter if delimiter is not None else " "
    header_emitted = False

    warn_if_tty(sys.stdin, "propagate")

    if skip_header is not None:
        for _ in range(skip_header):
            line = sys.stdin.readline()
            if not line:
                break

    def _emit_header(data_keys: list[str]) -> None:
        nonlocal header_emitted
        if header_emitted or not print_header:
            return
        fields = ["time"] + [k for k in data_keys if k != "times"]
        sys.stdout.write(out_delim.join(fields) + "\n")
        header_emitted = True

    def _emit_rows(data: dict[str, np.ndarray], times_iso: list[str]) -> None:
        keys = [k for k in data.keys() if k != "times"]
        _emit_header(keys)

        n = len(times_iso)
        rows: list[list[str]] = []
        for i in range(n):
            fields = [times_iso[i]]
            for k in keys:
                val = data[k][i]
                if isinstance(val, (np.floating, float)):
                    fields.append(f"{val:g}")
                elif isinstance(val, (np.integer, int)):
                    fields.append(str(int(val)))
                else:
                    fields.append(str(val))
            rows.append(fields)

        if align:
            writer = AlignedWriter(out_delim)
            for row in rows:
                writer.add_row(row)
            writer.flush()
        else:
            for row in rows:
                sys.stdout.write(out_delim.join(row) + "\n")

    while True:
        chunk_times: list[np.datetime64] = []
        chunk_iso: list[str] = []

        for line in sys.stdin:
            stripped = line.rstrip("\r\n")
            if not stripped:
                continue
            time_str = stripped.split()[0] if delimiter is None else stripped.split(delimiter)[0]

            try:
                dt = datetime.fromisoformat(time_str)
                chunk_times.append(np.datetime64(dt.isoformat(), "ns"))
                chunk_iso.append(time_str)
            except ValueError:
                print(
                    f"Warning: skipping unparseable time: {time_str}",
                    file=sys.stderr,
                )
                continue

            if len(chunk_times) >= chunk:
                break

        if not chunk_times:
            break

        times_arr = np.array(chunk_times, dtype="datetime64[ns]")
        data = generate(times_arr, propagator, groups, sites=sites_dict)
        _emit_rows(data, chunk_iso)
