"""Unit tests for shared typed CLI argument parser and Pydantic helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from backend.cli_common import (
    CommonConfigArgs,
    add_common_config_args,
    build_common_config_parser,
    parse_typed_args,
)


class SampleFlatArgs(CommonConfigArgs):
    """Sample flat CLI arguments model."""

    name: str = "default_name"
    count: int = 1
    flag: bool = False


class SampleSubCmdA(CommonConfigArgs):
    """Sample subcommand A."""

    command: Literal["cmd_a"]
    option_a: str = "val_a"


class SampleSubCmdB(CommonConfigArgs):
    """Sample subcommand B."""

    command: Literal["cmd_b"]
    option_b: int = 42


SampleUnionArgs = Annotated[
    SampleSubCmdA | SampleSubCmdB,
    Field(discriminator="command"),
]
SAMPLE_UNION_ADAPTER: TypeAdapter[SampleUnionArgs] = TypeAdapter(SampleUnionArgs)


def test_parse_typed_args_flat() -> None:
    # Given: An ArgumentParser with common and custom flags
    parser = argparse.ArgumentParser()
    add_common_config_args(parser)
    _ = parser.add_argument("--name", type=str, default="default_name")
    _ = parser.add_argument("--count", type=int, default=1)
    _ = parser.add_argument("--flag", action="store_true")

    # When: Parsing argument list
    args = parse_typed_args(
        parser,
        SampleFlatArgs,
        ["--name", "custom", "--count", "5", "--flag", "--data-dir", "/custom/data"],
    )

    # Then: Arguments are strongly typed
    assert isinstance(args, SampleFlatArgs)
    assert args.name == "custom"
    assert args.count == 5
    assert args.flag is True
    assert args.data_dir == "/custom/data"

    # And: Resolves AppConfig properly
    app_cfg = args.resolve_app_config()
    assert app_cfg.data_dir == Path("/custom/data").resolve()


def test_parse_typed_args_subcommands() -> None:
    # Given: A parser with subparsers using build_common_config_parser()
    parent = build_common_config_parser()
    parser = argparse.ArgumentParser(parents=[parent])
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_a = subparsers.add_parser("cmd_a", parents=[parent])
    _ = p_a.add_argument("--option-a", dest="option_a", type=str, default="val_a")

    p_b = subparsers.add_parser("cmd_b", parents=[parent])
    _ = p_b.add_argument("--option-b", dest="option_b", type=int, default=42)

    # When: Parsing subcommand A
    args_a = parse_typed_args(parser, SAMPLE_UNION_ADAPTER, ["cmd_a", "--option-a", "hello"])

    # Then: Subcommand A model is returned with correct types
    assert isinstance(args_a, SampleSubCmdA)
    assert args_a.command == "cmd_a"
    assert args_a.option_a == "hello"

    # When: Parsing subcommand B with common path options
    args_b = parse_typed_args(
        parser,
        SAMPLE_UNION_ADAPTER,
        ["cmd_b", "--option-b", "99", "--db", "/path/to/test.db"],
    )

    # Then: Subcommand B model is returned
    assert isinstance(args_b, SampleSubCmdB)
    assert args_b.command == "cmd_b"
    assert args_b.option_b == 99
    assert args_b.db == "/path/to/test.db"
    assert args_b.resolve_app_config().db_path == Path("/path/to/test.db").resolve()
