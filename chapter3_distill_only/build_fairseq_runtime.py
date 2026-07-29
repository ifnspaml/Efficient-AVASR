#!/usr/bin/env python3
"""Build only Fairseq's required Cython batching extensions out of tree."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy
from Cython.Build import cythonize
from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext


MODULES = (
    ("fairseq.data.data_utils_fast", "fairseq/data/data_utils_fast.pyx"),
    (
        "fairseq.data.token_block_utils_fast",
        "fairseq/data/token_block_utils_fast.pyx",
    ),
)


def build(source_root: Path, output_root: Path) -> None:
    source_root = source_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    extensions = [
        Extension(
            module,
            sources=[str(source_root / relative)],
            include_dirs=[numpy.get_include()],
            language="c++",
            extra_compile_args=["-std=c++11", "-O3"],
        )
        for module, relative in MODULES
    ]
    generated = cythonize(
        extensions,
        build_dir=str(output_root / "cython"),
        compiler_directives={"language_level": 3},
    )
    distribution = Distribution({"ext_modules": generated})
    command = build_ext(distribution)
    command.initialize_options()
    command.build_lib = str(output_root)
    command.build_temp = str(output_root / "build")
    command.inplace = False
    command.force = True
    command.finalize_options()
    command.run()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    build(args.source_root, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
