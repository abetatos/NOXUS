"""NOXUS command-line interface.

A thin dispatcher over the pipeline stages. The stages are scaffolds for now; the CLI exists so the
intended end-to-end shape (fetch → attribute → index → validate) is visible from the start.
"""

from __future__ import annotations

import argparse

from noxus import __version__
from noxus.config.region import TANGSHAN


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="noxus", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"noxus {__version__}")

    sub = parser.add_subparsers(dest="command")
    sub.add_parser("fetch", help="ingest TROPOMI NO2 for the study region")
    sub.add_parser("attribute", help="attribute the NO2 column to the cluster")
    sub.add_parser("index", help="build the monthly activity index")
    sub.add_parser("validate", help="lead/lag test against the official benchmark")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    print(f"[noxus] '{args.command}' over region '{TANGSHAN.name}' is not yet implemented.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
