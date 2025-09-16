# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from argparse import ArgumentParser

from phyagi.cli.convert import ConvertCommand
from phyagi.cli.evaluate import EvaluateCommand
from phyagi.cli.speed_benchmark import SpeedBenchmarkCommand
from phyagi.cli.start_ray_cluster import StartRayClusterCommand
from phyagi.utils.config import load_config


def main() -> None:
    parser = ArgumentParser("phyagi-cli", usage="phyagi-cli <command> [<args>]")
    commands_parser = parser.add_subparsers(help="phyagi-cli command helpers")

    ConvertCommand.register(commands_parser)
    EvaluateCommand.register(commands_parser)
    SpeedBenchmarkCommand.register(commands_parser)
    StartRayClusterCommand.register(commands_parser)

    args, extra_args = parser.parse_known_args()
    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    service = args.func(args, extra_args=load_config(extra_args))
    service.run()


if __name__ == "__main__":
    main()
