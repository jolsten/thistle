#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["rich>=13"]
# ///
"""Check each workspace package for changes since its last release tag.

For each per-package tag pattern declared below, finds the most recent
matching tag and reports:

- The tag itself, or that no matching tag exists yet.
- Commits since the tag whose changes touch the package path.
- Files changed in the package path since the tag.
- Whether the worktree has uncommitted changes inside the package path.
- A "releasable" verdict (yes if any commits or uncommitted changes in
  the package path; no otherwise).

The default output is a summary table. `rich` is declared as an inline
script dependency, so running through uv renders a styled table; run via
bare `python` (without rich installed) it falls back to a plain-text
table. Useful when deciding which package(s) need a release tag and what
the release notes should cover.

Usage:
    uv run scripts/check-release-status.py            # summary table
    uv run scripts/check-release-status.py --verbose  # also list commits
    uv run scripts/check-release-status.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

# Per-package tag patterns, mirroring the release workflow
# (.github/workflows/release.yml). Add new workspace packages here.
# The [0-9] in thistle's pattern keeps it from matching thistle-db tags.
PACKAGES: list[tuple[str, str]] = [
    ("packages/thistle", "thistle-v[0-9]*"),
    ("packages/thistle-db", "thistle-db-v[0-9]*"),
]


def _git(*args: str) -> tuple[int, str]:
    """Run `git <args>`; return (exit_code, stdout_stripped)."""
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout.strip()


def _git_lines(*args: str) -> list[str]:
    """Run a git command and return non-empty stdout lines."""
    _, out = _git(*args)
    return [line for line in out.splitlines() if line]


def _last_tag(pattern: str) -> str | None:
    """Return the most recent reachable tag matching `pattern`, or None."""
    code, out = _git("describe", "--tags", "--match", pattern, "--abbrev=0")
    return out if code == 0 else None


def check_package(path: str, pattern: str) -> dict:
    """Gather release-status info for one package."""
    tag = _last_tag(pattern)
    # When no tag exists yet, treat the whole history of the path as
    # "since last tag" — every commit is unreleased work.
    range_spec = f"{tag}..HEAD" if tag else "HEAD"

    commits = _git_lines("log", "--oneline", range_spec, "--", path)
    files = _git_lines("diff", "--name-only", range_spec, "--", path)
    worktree = _git_lines("status", "--porcelain", "--", path)

    return {
        "path": path,
        "tag_pattern": pattern,
        "last_tag": tag,
        "commits_since_tag": commits,
        "files_changed": files,
        "uncommitted_changes": worktree,
        "releasable": bool(commits) or bool(worktree),
    }


# --- summary columns (shared by the rich and plain-text renderers) ---------
#
# Each entry: (header, justify, accessor) where accessor maps a report dict to
# the cell's (plain_text, rich_markup) pair. Keeping one source of truth means
# the two renderers can never drift in which columns they show.

_HEADERS = ["Package", "Last tag", "Commits", "Files", "Worktree", "Releasable"]
_JUSTIFY = ["left", "left", "right", "right", "left", "center"]


def _cells(pkg: dict) -> list[tuple[str, str]]:
    """Return per-column (plain, rich-markup) cell text for one package."""
    n_worktree = len(pkg["uncommitted_changes"])
    if n_worktree:
        worktree = (f"DIRTY ({n_worktree})", f"[yellow]DIRTY ({n_worktree})[/]")
    else:
        worktree = ("clean", "[green]clean[/]")
    if pkg["releasable"]:
        releasable = ("yes", "[bold green]yes[/]")
    else:
        releasable = ("no", "[dim]no[/]")
    tag = pkg["last_tag"] or "(none yet)"
    tag_markup = pkg["last_tag"] or "[dim](none yet)[/]"
    n_commits = str(len(pkg["commits_since_tag"]))
    n_files = str(len(pkg["files_changed"]))
    return [
        (pkg["path"], pkg["path"]),
        (tag, tag_markup),
        (n_commits, n_commits),
        (n_files, n_files),
        worktree,
        releasable,
    ]


def _verbose_detail_lines(pkg: dict) -> list[str]:
    """Per-package commit / uncommitted-file detail (plain text)."""
    lines: list[str] = []
    if pkg["commits_since_tag"]:
        lines.append(f"{pkg['path']}: commits since {pkg['last_tag'] or 'repo start'}:")
        lines += [f"    {line}" for line in pkg["commits_since_tag"]]
    if pkg["uncommitted_changes"]:
        lines.append(f"{pkg['path']}: uncommitted in worktree:")
        lines += [f"    {line}" for line in pkg["uncommitted_changes"]]
    return lines


def _plain_rows(reports: list[dict]) -> list[list[str]]:
    return [[plain for plain, _ in _cells(pkg)] for pkg in reports]


def _column_widths(rows: list[list[str]]) -> list[int]:
    return [max(len(_HEADERS[i]), *(len(row[i]) for row in rows)) for i in range(len(_HEADERS))]


def render_rich(reports: list[dict], *, verbose: bool) -> None:
    """Render the summary as a styled rich table; plain fallback if no rich."""
    try:
        from rich import box  # pyright: ignore[reportMissingImports]
        from rich.console import Console  # pyright: ignore[reportMissingImports]
        from rich.table import Table  # pyright: ignore[reportMissingImports]
    except ImportError:
        render_plain(reports, verbose=verbose)
        return

    # Size the console to the content. A narrow (e.g. 80-col legacy Windows)
    # terminal would otherwise make rich truncate cells with an ellipsis that
    # is itself non-cp1252 and renders as a replacement glyph.
    widths = _column_widths(_plain_rows(reports))
    table_width = sum(widths) + 4 * len(_HEADERS) + 1  # padding + borders headroom
    console = Console(width=max(table_width, 80))
    table = Table(title="Release status (changes since last tag)", box=box.SIMPLE_HEAVY)
    for header, justify in zip(_HEADERS, _JUSTIFY):
        table.add_column(header, justify=justify, no_wrap=True)
    for pkg in reports:
        table.add_row(*(markup for _, markup in _cells(pkg)))
    console.print(table)

    if verbose:
        detail = [line for pkg in reports for line in _verbose_detail_lines(pkg)]
        if detail:
            console.rule("[bold]detail")
            for line in detail:
                console.print(line, highlight=False)


def render_plain(reports: list[dict], *, verbose: bool) -> None:
    """Render the summary as a dependency-free aligned text table."""
    rows = _plain_rows(reports)
    widths = _column_widths(rows)

    def fmt(cells: list[str]) -> str:
        out = []
        for cell, width, justify in zip(cells, widths, _JUSTIFY):
            out.append(cell.rjust(width) if justify == "right" else cell.ljust(width))
        return "  ".join(out).rstrip()

    print(fmt(_HEADERS))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))

    if verbose:
        detail = [line for pkg in reports for line in _verbose_detail_lines(pkg)]
        if detail:
            print()
            for line in detail:
                print(line)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="list every commit and uncommitted-file path per package",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of the summary table",
    )
    args = parser.parse_args()

    code, root = _git("rev-parse", "--show-toplevel")
    if code != 0 or not root:
        print("not inside a git repository", file=sys.stderr)
        return 2
    # Run all subsequent git commands from the repo root so pathspecs
    # resolve consistently regardless of where the script was invoked.
    os.chdir(root)

    reports = [check_package(path, pattern) for path, pattern in PACKAGES]

    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        render_rich(reports, verbose=args.verbose)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
