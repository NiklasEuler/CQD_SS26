import os
import sys
import subprocess
import tempfile

class Test_exercise_sheets:

    def test_exercise_sheet_1(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "script",
                    "--output",
                    "exercise1",
                    "--output-dir",
                    temp_dir,
                    "Notebooks/exercise1.ipynb",
                ],
                check=True,
            )
            subprocess.run(
                [sys.executable, "-m", "IPython", f"{temp_dir}/exercise1.py"],
                check=True,
                env={**os.environ, "MPLBACKEND": "Agg"},
            )  # Check that exercise1.ipynb runs without errors.

    def test_exercise_solution_1(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "script",
                    "--output",
                    "exercise1_sol",
                    "--output-dir",
                    temp_dir,
                    "Solutions/exercise1_sol.ipynb",
                ],
                check=True,
            )
            subprocess.run(
                [sys.executable, "-m", "IPython", f"{temp_dir}/exercise1_sol.py"],
                check=True,
                env={**os.environ, "MPLBACKEND": "Agg"},
            )  # Check that exercise1_sol.ipynb runs without errors.


    def test_exercise_solution_2(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "script",
                    "--output",
                    "exercise2_sol",
                    "--output-dir",
                    temp_dir,
                    "Solutions/exercise2_sol.ipynb",
                ],
                check=True,
            )
            subprocess.run(
                [sys.executable, "-m", "IPython", f"{temp_dir}/exercise2_sol.py"],
                check=True,
                env={**os.environ, "MPLBACKEND": "Agg"},
            )  # Check that exercise2_sol.ipynb runs without errors.


    def test_exercise_solution_3(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "script",
                    "--output",
                    "exercise3_sol",
                    "--output-dir",
                    temp_dir,
                    "Solutions/exercise3_sol.ipynb",
                ],
                check=True,
            )
            subprocess.run(
                [sys.executable, "-m", "IPython", f"{temp_dir}/exercise3_sol.py"],
                check=True,
                env={**os.environ, "MPLBACKEND": "Agg"},
            )  # Check that exercise3_sol.ipynb runs without errors.

    def test_exercise_solution_4(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "script",
                    "--output",
                    "exercise4_sol",
                    "--output-dir",
                    temp_dir,
                    "Solutions/exercise4_sol.ipynb",
                ],
                check=True,
            )
            subprocess.run(
                [sys.executable, "-m", "IPython", f"{temp_dir}/exercise4_sol.py"],
                check=True,
                env={**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"},
            )  # Check that exercise4_sol.ipynb runs without errors.
