import tempfile
import shutil
import os
from itertools import count
from datetime import datetime

import sys
from functools import wraps
import contextlib


class FileCount:

    _max = 2

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.counter = count(start=1, step=1)

    def __call__(self):
        i = next(self.counter)
        return f"{str(i).zfill(self._max)}"

    def add_prefix(self, file_name: os.PathLike) -> os.PathLike:
        return os.path.normpath(f"{self.prefix}_{self()}_{os.path.basename(file_name)}")

    def get_path(self, dir_path, file_name) -> os.PathLike:
        fname = self.add_prefix(file_name=file_name)
        return os.path.join(dir_path, fname)

    def reset(self):
        self.counter = count(start=1, step=1)


def cleanup_temp_folder(directory: os.PathLike):
    particles = ["_temp.", ".txt"]

    for file in os.listdir(directory):
        if any(p in file for p in particles):
            _path = os.path.join(directory, file)
            os.remove(_path)


class SimEnv:

    def __init__(
        self,
        source_file: os.PathLike,
        delete_after: bool = False,
        stdout_redirect: bool = False,
        cleanup: bool = True,
    ):
        self.source_file = os.path.normpath(source_file)
        self.delete_after = delete_after
        self.redirect = stdout_redirect
        self._tempdir = None
        self.cleanup = cleanup

    def __enter__(self) -> os.PathLike:
        self._tempdir = tempfile.mkdtemp(
            dir=os.path.dirname(self.source_file),
            prefix=f"{os.path.splitext(os.path.basename(self.source_file))[0]}_{datetime.now().strftime("%d-%m-%Y")}_",
        )

        self._stdout = sys.stdout
        if self.redirect:
            sys.stdout = open(os.path.join(self._tempdir, "events.LOG"), "w")

        return self._tempdir

    def __exit__(self, exc_type, exc_value, traceback):

        if self.redirect:
            sys.stdout.close()
            sys.stdout = self._stdout

        if self.delete_after and self._tempdir:
            shutil.rmtree(self._tempdir, ignore_errors=True)
            return

        if self.cleanup:
            cleanup_temp_folder(self._tempdir)


def silent(func):
    @contextlib.contextmanager
    def block_stdout():
        original_stdout = sys.stdout
        try:
            sys.stdout = open("nul", "w")
            yield
        finally:
            sys.stdout = original_stdout  # restore stdout

    def wrapper(*args, **kwargs):
        with block_stdout():
            return func(*args, **kwargs)

    return wrapper
