# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-trend: Test Runner
# MAGIC
# MAGIC Runs the full pytest suite for `insurance-trend` on Databricks serverless compute.
# MAGIC Upload this notebook and the project to the workspace before running.

# COMMAND ----------

# MAGIC %pip install pandas numpy statsmodels scipy ruptures matplotlib requests polars pytest

# COMMAND ----------

import subprocess
import sys
import os

# COMMAND ----------

# Clone or upload the project. If running via CI, files will already be at /tmp/insurance-trend.
# If running manually, upload via: databricks workspace import-dir ./src /Workspace/...
# and adjust the path below.

project_path = "/tmp/insurance-trend"

# COMMAND ----------

# Install the package in editable mode
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", project_path],
    capture_output=True, text=True
)
print(result.stdout)
print(result.stderr)

# COMMAND ----------

# Run pytest
result = subprocess.run(
    [sys.executable, "-m", "pytest", f"{project_path}/tests", "-v", "--tb=short"],
    capture_output=True, text=True, cwd=project_path
)
print(result.stdout[-10000:])  # Last 10k chars to avoid truncation
print(result.stderr[-3000:])

if result.returncode != 0:
    raise Exception(f"Tests failed with return code {result.returncode}")
else:
    print("\n=== ALL TESTS PASSED ===")
