# The Python package only ships the headers: they are copied from cusvna/ to cusvna/include/cusvna/ in the
# wheel, cusvna.get_include() returns the directory to add to the include paths. All metadata lives in
# pyproject.toml.
import glob
import os

from setuptools import setup
from setuptools.command.build_py import build_py

root = os.path.dirname(os.path.abspath(__file__))


class build_py_with_headers(build_py):
    def run(self):
        super().run()
        headers = sorted(glob.glob(os.path.join(root, "cusvna", "*.cuh")))
        if not headers:
            raise RuntimeError("cusvna headers not found")
        target = os.path.join(self.build_lib, "cusvna", "include", "cusvna")
        self.mkpath(target)
        for header in headers:
            self.copy_file(header, os.path.join(target, os.path.basename(header)))


setup(cmdclass={"build_py": build_py_with_headers})
