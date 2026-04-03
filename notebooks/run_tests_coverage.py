# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-trend: expanded test coverage run
# MAGIC
# MAGIC Runs the full test suite including the three new coverage files.

# COMMAND ----------

# MAGIC %pip install -e /Workspace/insurance-trend[dev] --quiet

# COMMAND ----------

import subprocess, sys

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-trend/tests/",
     "-x", "-q",
     "--tb=short",
     "--no-header",
     "-p", "no:warnings",
    ],
    capture_output=True,
    text=True,
    cwd="/Workspace/insurance-trend",
)
print(result.stdout[-8000:] if len(result.stdout) > 8000 else result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-2000:])
print("Return code:", result.returncode)

# COMMAND ----------

# Run only the new coverage files to get a focused view

result2 = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-trend/tests/test_coverage_utils_breaks.py",
     "/Workspace/insurance-trend/tests/test_coverage_fitters.py",
     "/Workspace/insurance-trend/tests/test_coverage_inflation_index.py",
     "-v", "--tb=short", "--no-header",
    ],
    capture_output=True,
    text=True,
    cwd="/Workspace/insurance-trend",
)
print(result2.stdout[-10000:] if len(result2.stdout) > 10000 else result2.stdout)
if result2.stderr:
    print("STDERR:", result2.stderr[-2000:])
print("Return code:", result2.returncode)
