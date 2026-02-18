import os
import re
import subprocess
import sys
import shutil
from multiprocessing import cpu_count
from pathlib import Path

from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection & inclusion of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.3
        
        # Get PyTorch cmake path
        try:
            import torch
            torch_cmake_path = torch.utils.cmake_prefix_path
        except ImportError:
            torch_cmake_path = None
        
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]
        if not cmake_generator and self.compiler.compiler_type != "msvc" and shutil.which("ninja"):
            # Ninja is the most common fast default for local CMake builds.
            cmake_args += ["-G", "Ninja"]
        
        # Add Torch cmake path if available
        if torch_cmake_path:
            cmake_args.append(f"-DCMAKE_PREFIX_PATH={torch_cmake_path}")
        build_args = []

        # Pass the root directory of the project to CMake (one level up from python/)
        # formatting for different platforms
        if self.compiler.compiler_type == "msvc":
            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by CMakeExtension
            # by default, so we do it manually.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]
            else:
                # Default to all available cores for editable installs.
                os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = str(cpu_count())

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # The CMakeLists.txt is in the root directory, not in python/
        root_dir = Path(__file__).parent.parent.absolute()
        
        subprocess.check_call(
            ["cmake", str(root_dir)] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )

setup(
    name="ddlp",
    version="0.1.0",
    author="Samuel Nordmann",
    author_email="snordmann@nvidia.com",
    description="Distributed Deep Learning Primitives",
    long_description="",
    ext_modules=[CMakeExtension("ddlp._C", sourcedir="..")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    extras_require={"test": ["pytest>=6.0"]},
    install_requires=[
        "torch",
        "numpy",
    ],
    python_requires=">=3.8",
    packages=find_packages(),
)

