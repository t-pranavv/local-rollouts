# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess
from typing import Dict

__version__ = "3.2.2.dev"


def get_package_information() -> Dict[str, str]:
    """Get package information from ``git``.

    This function requires ``git`` to be available as a command line tool and is useful
    for debugging purposes.

    Returns:
        Package information, such a current version, branch, and commit hash.

    """

    def _get_branch() -> str:
        try:
            return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf-8").strip()
        except subprocess.CalledProcessError:
            return ""

    def _get_commit_hash() -> str:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        except subprocess.CalledProcessError:
            return ""

    return {"version": __version__, "branch": _get_branch(), "commit": _get_commit_hash()}
