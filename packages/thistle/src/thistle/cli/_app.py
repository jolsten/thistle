"""Typer application wiring all thistle subcommands."""

from __future__ import annotations

import json
import math
import pathlib
import sys
from datetime import datetime, timedelta
from enum import Enum
from typing import NoReturn, Optional

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
    maneuver_epochs,
    nodal_rate_rev_per_day,
    parse_site,
    parse_tle_epochs,
    read_and_emit_tles,
    rev_bounds,
    revnum_at,
    warn_if_tty,
)

from thistle.cli import _db
from thistle.cli._config import (
    TLE_DIR_ENV,
    TLE_EXT_ENV,
    find_candidate_file,
    is_catalog_id,
    load_config,
)

app = typer.Typer(
    name="thistle",
    help="Satellite orbit propagation and TLE analysis tools.\n\n"
    f"Environment: set {TLE_DIR_ENV} to a directory of per-object TLE files "
    "to pass NORAD catalog IDs (e.g. 25544 or A0001) instead of file paths; "
    f"{TLE_EXT_ENV} sets their extension (default .tle). With 'thistle[db]' "
    "installed and THISTLE_DB_* configured, IDs not found there fall back to "
    "the thistle-db database. See 'thistle config'.",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _resolve_tle_path(file: pathlib.Path) -> pathlib.Path:
    """Resolve a FILE argument, treating catalog IDs via THISTLE_TLE_DIR.

    A path that exists is always used verbatim. A non-existent all-digit or
    Alpha-5 argument is looked up as {tle_dir}/{id}{ext}, then in the
    thistle-db database when that fallback is active ('thistle[db]' installed
    and THISTLE_DB_* configured); a miss is a hard error naming everything
    tried. Anything else passes through for the caller's normal
    file-not-found handling.
    """
    if file.exists():
        return file
    cfg = load_config()
    name = str(file)
    if not is_catalog_id(name):
        return file
    found, tried = find_candidate_file(name, cfg)
    if found is not None:
        return found
    tles = _db.lookup_tles(name)
    if tles:
        return _db.tles_to_tempfile(tles, cfg.tle_ext)
    if tles is not None:
        tried.append("the thistle-db database")
    if not tried:
        return file
    print(
        f"Error: TLE file not found: {file} (also tried {', '.join(tried)})",
        file=sys.stderr,
    )
    raise typer.Exit(code=2)


class SwitchStrategy(str, Enum):
    epoch = "epoch"
    midpoint = "midpoint"
    tca = "tca"


class PlotPreset(str, Enum):
    leo = "leo"
    geo = "geo"
    sunsync = "sunsync"
    heo = "heo"


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
        file = _resolve_tle_path(file)
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

    file = _resolve_tle_path(file)
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
# revnum
# ---------------------------------------------------------------------------


def _parse_rev_or_time(value: str) -> int | datetime:
    """Classify a CLI value as a revnum (int) or an epoch (datetime)."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(
            f"expected a revolution number or ISO 8601 timestamp, got: {value}"
        )


def _ascending_crossings(
    line1: str, line2: str, start: datetime, stop: datetime
) -> list[datetime]:
    """SGP4-exact ascending-node crossing times within [start, stop]."""
    from skyfield.api import EarthSatellite

    from thistle.events import find_node_crossings

    satellite = EarthSatellite(line1, line2)
    crossings = find_node_crossings(
        np.datetime64(start.isoformat(), "us"),
        np.datetime64(stop.isoformat(), "us"),
        satellite,
    )
    return [
        c["start"].astype("datetime64[us]").item()
        for c in crossings
        if c["ascending"]
    ]


@app.command("revnum")
def revnum_cmd(
    file: Annotated[pathlib.Path, typer.Argument(help="TLE file path")],
    value: Annotated[
        str,
        typer.Argument(
            help="Revolution number (integer) or epoch (ISO 8601). "
            "Integers always parse as revnums, so 20240315 is a revnum, not a date."
        ),
    ],
    value2: Annotated[
        Optional[str],
        typer.Argument(
            help="Optional second value: give one revnum and one epoch "
            "(either order) to find which object(s) match"
        ),
    ] = None,
    refine: Annotated[
        bool,
        typer.Option(
            "--refine",
            help="Snap to SGP4-exact ascending-node crossings. Trusts the "
            "estimate to be within half a rev for the integer count, so "
            "accuracy degrades far from the TLE epoch. Convert modes only.",
        ),
    ] = False,
    tol: Annotated[
        float,
        typer.Option("--tol", help="Match-mode tolerance in revolutions"),
    ] = 1.0,
) -> None:
    """Convert between revolution number and epoch, or match objects to a rev.

    Revolutions are counted at ascending-node crossings. With a revnum,
    prints the estimated start and stop of that revolution. With an epoch,
    prints the estimated fractional revnum at that time. With both, scans a
    multi-object TLE file for objects on that revnum (within --tol) at that
    time. Note: the TLE revnum field is 5 digits and rolls over at 100000;
    rollover is not handled.
    """
    try:
        parsed_values = [_parse_rev_or_time(value)]
        if value2 is not None:
            parsed_values.append(_parse_rev_or_time(value2))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise typer.Exit(code=2)

    file = _resolve_tle_path(file)
    try:
        f = open(file)
    except FileNotFoundError:
        print(f"Error: TLE file not found: {file}", file=sys.stderr)
        raise typer.Exit(code=2)

    with f:
        tles = parse_tle_epochs(f)

    if not tles:
        print(f"Error: no TLEs found in {file}", file=sys.stderr)
        raise typer.Exit(code=2)

    revs = [v for v in parsed_values if isinstance(v, int)]
    times = [v for v in parsed_values if isinstance(v, datetime)]

    if len(parsed_values) == 2:
        if len(revs) != 1 or len(times) != 1:
            print(
                "Error: match mode requires one revnum and one timestamp",
                file=sys.stderr,
            )
            raise typer.Exit(code=2)
        _revnum_match(tles, revs[0], times[0], tol, refine)
    elif revs:
        _rev_to_epoch(tles, revs[0], refine)
    else:
        _epoch_to_rev(tles, times[0], refine)


def _rev_to_epoch(
    tles: list[tuple[str, str, Satrec]], target: int, refine: bool
) -> None:
    line1, line2, sat = min(tles, key=lambda t: abs(t[2].revnum - target))
    start, stop = rev_bounds(sat, target)

    if refine:
        period = timedelta(days=1.0 / nodal_rate_rev_per_day(sat))
        crossings = _ascending_crossings(
            line1, line2, start - 1.5 * period, stop + 1.5 * period
        )
        if crossings:
            start = min(crossings, key=lambda c: abs(c - start))
            stop = min(crossings, key=lambda c: abs(c - stop))
        else:
            print(
                "Warning: no ascending-node crossings found; "
                "falling back to estimate",
                file=sys.stderr,
            )

    print(f"{start.isoformat()} {stop.isoformat()}")


def _epoch_to_rev(
    tles: list[tuple[str, str, Satrec]], t: datetime, refine: bool
) -> None:
    def epoch_dist(item: tuple[str, str, Satrec]) -> float:
        sat = item[2]
        return abs((epoch_to_datetime(sat.epochyr, sat.epochdays) - t).total_seconds())

    line1, line2, sat = min(tles, key=epoch_dist)
    result = revnum_at(sat, t)

    if refine:
        period = timedelta(days=1.0 / nodal_rate_rev_per_day(sat))
        crossings = _ascending_crossings(
            line1, line2, t - 1.5 * period, t + 0.5 * period
        )
        prev = [c for c in crossings if c <= t]
        nxt = [c for c in crossings if c > t]
        if prev:
            t_prev = max(prev)
            t_next = min(nxt) if nxt else t_prev + period
            whole = round(revnum_at(sat, t_prev))
            frac = (t - t_prev).total_seconds() / (t_next - t_prev).total_seconds()
            result = whole + frac
        else:
            print(
                "Warning: no ascending-node crossings found; "
                "falling back to estimate",
                file=sys.stderr,
            )

    print(f"{result:.3f}")


def _revnum_match(
    tles: list[tuple[str, str, Satrec]],
    target: int,
    t: datetime,
    tol: float,
    refine: bool,
) -> None:
    if refine:
        print(
            "Note: --refine is ignored in match mode (estimates only)",
            file=sys.stderr,
        )

    best_by_satnum: dict[int, Satrec] = {}
    for _, _, sat in tles:
        age = abs((epoch_to_datetime(sat.epochyr, sat.epochdays) - t).total_seconds())
        current = best_by_satnum.get(sat.satnum)
        if current is None or age < abs(
            (epoch_to_datetime(current.epochyr, current.epochdays) - t).total_seconds()
        ):
            best_by_satnum[sat.satnum] = sat

    matches = []
    for sat in best_by_satnum.values():
        est = revnum_at(sat, t)
        delta = est - target
        if abs(math.floor(est) - target) <= tol:
            age_days = (
                t - epoch_to_datetime(sat.epochyr, sat.epochdays)
            ).total_seconds() / 86400.0
            matches.append((sat, est, delta, age_days))

    matches.sort(key=lambda m: abs(m[2]))

    writer = AlignedWriter(" ", numeric=[True, False, True, True, True])
    for sat, est, delta, age_days in matches:
        writer.add_row([
            str(sat.satnum),
            sat.intldesg.strip(),
            f"{est:.3f}",
            f"{delta:+.3f}",
            f"{age_days:.2f}",
        ])
    writer.flush()


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


@app.command("plot")
def plot_cmd(
    file: Annotated[
        Optional[pathlib.Path],
        typer.Argument(help="TLE file path, single object (default: stdin)"),
    ] = None,
    output: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "-o",
            "--output",
            help="Save figure to file (format from extension: png/pdf/svg). "
            "Without -o, opens an interactive window.",
        ),
    ] = None,
    fields: Annotated[
        Optional[str],
        typer.Option(
            "--fields",
            help="Comma-separated panels: sma, mm, ecc, inc, raan, aop, "
            "bstar, peri, apo, revnum, lon, ltan",
        ),
    ] = None,
    preset: Annotated[
        Optional[PlotPreset],
        typer.Option(
            "--preset",
            help="Panel preset for an orbit regime: leo (decay), geo "
            "(stationkeeping/longitude), sunsync (LTAN drift), heo "
            "(frozen perigee). Mutually exclusive with --fields.",
        ),
    ] = None,
    maneuvers: Annotated[
        bool,
        typer.Option(
            "--maneuvers",
            help="Mark discrete jumps in mean motion (likely maneuvers) "
            "with vertical lines",
        ),
    ] = False,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            help="Maneuver detection sensitivity in robust sigmas of the "
            "TLE-to-TLE mean motion deltas (lower = more sensitive)",
        ),
    ] = 10.0,
    title: Annotated[
        Optional[str],
        typer.Option("--title", help="Figure title (default: satnum and intl)"),
    ] = None,
) -> None:
    """Plot TLE-derived orbital elements over time as stacked subplots.

    Requires the plot extra: pip install 'thistle[plot]'. For files with
    multiple objects, select one first: thistle filter --satnum NNN | thistle plot
    """
    try:
        from thistle.cli._plot import DEFAULT_FIELDS, FIELDS, PRESETS
    except ImportError:
        _plot_import_error()

    if preset is not None and fields is not None:
        print(
            "Error: --preset and --fields are mutually exclusive",
            file=sys.stderr,
        )
        raise typer.Exit(code=2)

    if preset is not None:
        field_list = PRESETS[preset.value]
    elif fields is not None:
        field_list = [f.strip() for f in fields.split(",") if f.strip()]
        unknown = [f for f in field_list if f not in FIELDS]
        if unknown or not field_list:
            print(
                f"Error: unknown field(s): {', '.join(unknown) or '(none given)'} "
                f"(valid: {', '.join(FIELDS)})",
                file=sys.stderr,
            )
            raise typer.Exit(code=2)
    else:
        field_list = DEFAULT_FIELDS

    sats = _load_single_object(file, "plot")

    try:
        if output is not None:
            import matplotlib

            matplotlib.use("Agg")

        from thistle.cli._plot import make_figure

        events: list[datetime] = []
        if maneuvers:
            events = maneuver_epochs(sats, threshold)
            print(
                f"Detected {len(events)} maneuver event(s)",
                file=sys.stderr,
            )
            if len(events) > 100:
                print(
                    "Hint: markers may be dense; raise --threshold or narrow "
                    "the time range (thistle filter --after/--before | thistle plot)",
                    file=sys.stderr,
                )

        fig = make_figure(sats, field_list, events, title)

        if output is not None:
            fig.savefig(output)
        else:
            import matplotlib.pyplot as plt

            plt.show()
    except ImportError:
        _plot_import_error()


def _plot_import_error() -> NoReturn:
    print(
        "thistle plot requires additional packages.\n"
        "Install with: pip install 'thistle[plot]'",
        file=sys.stderr,
    )
    raise typer.Exit(code=1)


def _load_single_object(
    file: Optional[pathlib.Path], name: str
) -> list[Satrec]:
    """Load an epoch-sorted single-object TLE list from a file or stdin.

    Exits with code 2 on missing file, no TLEs, or multiple objects.
    """
    if file is not None:
        file = _resolve_tle_path(file)
        try:
            source = open(file)
        except FileNotFoundError:
            print(f"Error: TLE file not found: {file}", file=sys.stderr)
            raise typer.Exit(code=2)
        with source:
            parsed = parse_tle_epochs(source)
    else:
        warn_if_tty(sys.stdin, name)
        parsed = parse_tle_epochs(sys.stdin)

    if not parsed:
        print("Error: no TLEs found", file=sys.stderr)
        raise typer.Exit(code=2)

    satnums = {sat.satnum for _, _, sat in parsed}
    if len(satnums) > 1:
        print(
            f"Error: file contains {len(satnums)} objects; "
            "use 'thistle filter --satnum NNN' to select one object",
            file=sys.stderr,
        )
        raise typer.Exit(code=2)

    return sorted(
        (sat for _, _, sat in parsed),
        key=lambda s: epoch_to_datetime(s.epochyr, s.epochdays),
    )


# ---------------------------------------------------------------------------
# groundtrack
# ---------------------------------------------------------------------------


@app.command("groundtrack")
def groundtrack_cmd(
    tle: Annotated[
        Optional[pathlib.Path],
        typer.Argument(
            help="Default TLE file or catalog ID (spec lines may override "
            "with sat=...)"
        ),
    ] = None,
    spec: Annotated[
        Optional[pathlib.Path],
        typer.Option("--spec", help="Trace spec file (default: stdin)"),
    ] = None,
    output: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            "-o",
            "--output",
            help="Save figure to file (format from extension). "
            "Without -o, opens an interactive window.",
        ),
    ] = None,
    step: Annotated[
        float,
        typer.Option("--step", help="Trace sampling step in seconds"),
    ] = 60.0,
    site: Annotated[
        Optional[list[str]],
        typer.Option("--site", help="Ground site: NAME:LAT:LON[:ALT] (repeatable)"),
    ] = None,
    min_el: Annotated[
        float,
        typer.Option("--min-el", help="Visibility ring elevation angle (deg)"),
    ] = 0.0,
    sat_alt: Annotated[
        Optional[float],
        typer.Option(
            "--sat-alt",
            help="Visibility ring satellite altitude in meters "
            "(default: mean altitude of plotted traces)",
        ),
    ] = None,
    title: Annotated[
        Optional[str],
        typer.Option("--title", help="Figure title"),
    ] = None,
) -> None:
    """Plot ground traces on a world map from spec lines.

    Each spec line is 'START STOP [key=val ...]' (ISO 8601 times). Options:
    color, style, width, alpha, marker, label, and sat=<id-or-path> to plot
    a different object on that line. '#' comments and blank lines are
    skipped. Ground sites get a marker and, when a satellite altitude is
    known, a dashed visibility-horizon ring. Requires the plot extra.
    """
    from thistle.cli._map import parse_spec_line

    sites_dict: dict[str, tuple] = {}
    if site:
        for s in site:
            try:
                name, coords = parse_site(s)
                sites_dict[name] = coords
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                raise typer.Exit(code=2)

    if spec is not None:
        try:
            f = open(spec)
        except FileNotFoundError:
            print(f"Error: spec file not found: {spec}", file=sys.stderr)
            raise typer.Exit(code=2)
        with f:
            spec_lines = f.readlines()
    else:
        warn_if_tty(sys.stdin, "groundtrack")
        spec_lines = sys.stdin.readlines()

    specs = [s for s in map(parse_spec_line, spec_lines) if s is not None]

    from thistle import Propagator, read_tle as read_tle_file

    propagators: dict[pathlib.Path, "Propagator"] = {}

    def get_propagator(source: pathlib.Path) -> Optional["Propagator"]:
        try:
            resolved = _resolve_tle_path(source)
        except typer.Exit:
            # resolver already printed the paths it tried; skip this line
            return None
        if resolved not in propagators:
            try:
                tles = read_tle_file(resolved)
            except FileNotFoundError:
                print(f"Warning: TLE file not found: {resolved}", file=sys.stderr)
                return None
            if not tles:
                print(f"Warning: no TLEs found in {resolved}", file=sys.stderr)
                return None
            propagators[resolved] = Propagator(tles, method="midpoint")
        return propagators[resolved]

    from thistle.cli._map import trace_lla

    traces: list[tuple[np.ndarray, np.ndarray, dict]] = []
    all_alts: list[np.ndarray] = []
    for sp in specs:
        if sp.sat is not None:
            source = pathlib.Path(sp.sat)
        elif tle is not None:
            source = tle
        else:
            print(
                f"Warning: skipping spec line (no sat= and no TLE argument): "
                f"{sp.start.isoformat()}",
                file=sys.stderr,
            )
            continue
        propagator = get_propagator(source)
        if propagator is None:
            continue
        lons, lats, alts = trace_lla(propagator, sp.start, sp.stop, step)
        traces.append((lons, lats, sp.plot_kwargs))
        all_alts.append(alts)

    if not traces:
        print("Error: no valid traces to plot", file=sys.stderr)
        raise typer.Exit(code=2)

    ring_alt = sat_alt
    if ring_alt is None and all_alts:
        ring_alt = float(np.mean(np.concatenate(all_alts)))

    try:
        if output is not None:
            import matplotlib

            matplotlib.use("Agg")

        from thistle.cli._map import make_map

        fig = make_map(traces, sites_dict, ring_alt, min_el, title)

        if output is not None:
            fig.savefig(output)
        else:
            import matplotlib.pyplot as plt

            plt.show()
    except ImportError:
        _plot_import_error()


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@app.command("config")
def config_cmd(
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
) -> None:
    """Show the effective CLI configuration and where each setting came from."""
    import os

    cfg = load_config()
    dir_source = TLE_DIR_ENV if os.environ.get(TLE_DIR_ENV) else None
    ext_source = TLE_EXT_ENV if os.environ.get(TLE_EXT_ENV) else None
    db = _db.db_status()

    if json_out:
        print(
            json.dumps(
                {
                    "tle_dir": str(cfg.tle_dir) if cfg.tle_dir else None,
                    "tle_dir_source": dir_source,
                    "tle_ext": cfg.tle_ext,
                    "tle_ext_source": ext_source or "default",
                    **db,
                },
                indent=2,
            )
        )
    else:
        if cfg.tle_dir is not None:
            print(f"tle_dir: {cfg.tle_dir}  (from {dir_source})")
        else:
            print(f"tle_dir: (not set)  (set {TLE_DIR_ENV} to enable ID lookup)")
        print(f"tle_ext: {cfg.tle_ext}  (from {ext_source or 'default'})")
        if not db["db_installed"]:
            print("db_fallback: thistle-db not installed  (pip install 'thistle[db]')")
        elif not db["db_configured"]:
            print(
                "db_fallback: installed but not configured  "
                f"(set THISTLE_DB_DATABASE__* or {_db.DB_CONFIG_ENV})"
            )
        else:
            print(
                f"db_fallback: active  (thistle-db {db['db_version']}, {db['db_url']})"
            )

    if cfg.tle_dir is not None and not cfg.tle_dir.is_dir():
        print(
            f"Warning: {TLE_DIR_ENV} points to a non-existent directory: "
            f"{cfg.tle_dir}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# maneuvers
# ---------------------------------------------------------------------------


@app.command("maneuvers")
def maneuvers_cmd(
    file: Annotated[
        Optional[pathlib.Path],
        typer.Argument(help="TLE file path, single object (default: stdin)"),
    ] = None,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            help="Detection sensitivity in robust sigmas of the TLE-to-TLE "
            "mean motion deltas (lower = more sensitive)",
        ),
    ] = 10.0,
) -> None:
    """Print detected maneuver epochs, one ISO 8601 timestamp per line.

    Detects impulsive jumps in mean motion between consecutive TLEs (same
    detector as 'thistle plot --maneuvers'). Continuous low-thrust (e.g.
    electric orbit raising) appears as a smooth trend and is largely
    invisible to it. No output means no maneuvers detected.
    """
    sats = _load_single_object(file, "maneuvers")
    for event in maneuver_epochs(sats, threshold):
        print(event.isoformat())


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
    file = _resolve_tle_path(file)
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
        file = _resolve_tle_path(file)
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

    file = _resolve_tle_path(file)
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
