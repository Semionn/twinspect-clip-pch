# -*- coding: utf-8 -*-
from loguru import logger as log
import mmap
import os
import shutil
import sys


import blake3
from rich.progress import track

import twinspect as ts
import importlib
import pathlib
import yaml

__all__ = [
    "load_function",
    "install_dataset",
    "clusterize",
    "count_files",
    "hash_file",
    "hash_folder",
    "check_folder",
    "iter_original_files",
]

ROOT = pathlib.Path(__file__).parent.parent.absolute()


def load_function(path: str):
    module_path, function_name = path.split(':')
    module = importlib.import_module(module_path)
    function = getattr(module, function_name)
    return function


def install_dataset(dataset: ts.Dataset) -> pathlib.Path:
    install = load_function(dataset.installer)
    return install(dataset)


def clusterize(src: pathlib.Path, dst: pathlib.Path, clusters: int):
    """Copy files from source to destination into a cluster folder structure."""
    clustered = 0
    for path in src.rglob("*"):
        if path.is_file():
            if clustered < clusters:
                cluster_folder_name = f"{clustered:07d}"
                target = dst / cluster_folder_name
                target.mkdir()
                shutil.copy(path, target)
                clustered += 1
            else:
                shutil.copy(path, dst)


def count_files(path: pathlib.Path) -> int:
    """Count number of files in path recursively"""
    file_count = 0
    for root, dirs, files in os.walk(path):
        file_count += len(files)
    return file_count


def iter_files(path: pathlib.Path):
    """Iterate all files in path recurively with deterministic ordering"""
    for root, dirs, files in os.walk(path):
        dirs.sort()
        files.sort()
        for filename in files:
            yield pathlib.Path(os.path.join(root, filename))


def iter_original_files(data_folder: pathlib.Path):
    """Iterate over clusters in a data_folder and yield original files"""
    for subdir in data_folder.iterdir():
        if subdir.is_dir():
            files = list(subdir.glob('*'))
            if files:
                sorted_files = sorted(files)
                yield sorted_files[0]


def hash_file(file_path: pathlib.Path) -> bytes:
    """Create hash for file at file_path"""
    with file_path.open("r+b") as infile:
        mm = mmap.mmap(infile.fileno(), 0, access=mmap.ACCESS_READ)
        return blake3.blake3(mm, max_threads=blake3.blake3.AUTO).digest(8)


def hash_folder(path: pathlib.Path) -> str:
    """Create checksum of folder for reproducibility"""
    total = count_files(path)
    hasher = blake3.blake3()
    hashes = set()
    log.debug(f"Hashing {total} files in {path}")
    for file in track(iter_files(path), description="Hashing...", total=total, console=ts.console):
        file_hash = hash_file(file)
        hasher.update(file_hash)
        if file_hash in hashes:
            log.warning(f"Warning - Duplicate File - {file}")
        hashes.add(file_hash)
    return hasher.digest(8).hex()


def check_folder(path: pathlib.Path, checksum: str):
    """Check folder against checksum"""
    log.debug(f"Checking integrity for {path}")
    log.debug(f"Expected checksum: {checksum}")
    folder_hash = hash_folder(path)
    log.debug(f"Actual checksum:   {folder_hash}")
    if checksum == folder_hash:
        log.debug(f"Integrity verified")
    else:
        log.debug(f"Integrity error for {path}!")
        log.debug("Remove folder or update hash in configuration")
        sys.exit(1)


def format_yml():
    """Format all .yml files"""
    for f in ROOT.glob("**/*.yml"):
        with open(f, "rt", encoding="utf-8") as infile:
            data = yaml.safe_load(infile)
        with open(f, "wt", encoding="utf-8", newline="\n") as outf:
            yaml.safe_dump(
                data,
                outf,
                indent=2,
                width=100,
                encoding="utf-8",
                sort_keys=False,
                default_flow_style=False,
                default_style=None,
                allow_unicode=True,
            )


def fix_line_endings():
    """Normalize all line endings to unix LF"""
    WINDOWS_LINE_ENDING = b"\r\n"
    UNIX_LINE_ENDING = b"\n"
    extensions = {".py", ".toml", ".lock", ".md", ".yml", ".yaml"}
    for fp in ROOT.glob("**/*"):
        if fp.suffix in extensions:
            with open(fp, "rb") as infile:
                content = infile.read()
            new_content = content.replace(WINDOWS_LINE_ENDING, UNIX_LINE_ENDING)
            if new_content != content:
                with open(fp, "wb") as outfile:
                    outfile.write(new_content)
                print(f"       fixed line endings for {fp.name}")
