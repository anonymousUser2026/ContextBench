#!/usr/bin/env bash

source ~/.bashrc
SWEUTIL_DIR=/swe_util

# FIXME: Cannot read SWE_INSTANCE_ID from the environment variable
# SWE_INSTANCE_ID=django__django-11099
if [ -z "$SWE_INSTANCE_ID" ]; then
    echo "Error: SWE_INSTANCE_ID is not set." >&2
    exit 1
fi

# Read the swe-bench-test-lite.json file and extract the required item based on instance_id
item=$(jq --arg INSTANCE_ID "$SWE_INSTANCE_ID" '.[] | select(.instance_id == $INSTANCE_ID)' $SWEUTIL_DIR/eval_data/instances/swe-bench-instance.json)

if [[ -z "$item" ]]; then
  echo "No item found for the provided instance ID."
  exit 1
fi


# Check if version exists and is not null
version=$(echo "$item" | jq -r '.version // "null"')
if [[ "$version" == "null" || -z "$version" ]]; then
    # For PolyBench and SWE-bench Pro, use instance_id as workspace name
    WORKSPACE_NAME=$(echo "$item" | jq -r '.instance_id | tostring')
else
    # For standard SWE-bench, construct from repo and version
    WORKSPACE_NAME=$(echo "$item" | jq -r '(.repo | tostring) + "__" + (.version | tostring) | gsub("/"; "__")')
fi

echo "WORKSPACE_NAME: $WORKSPACE_NAME"

# Clear the workspace
if [ -d /workspace ]; then
    rm -rf /workspace/*
else
    mkdir /workspace
fi
# Copy repo to workspace
if [ -d /workspace/$WORKSPACE_NAME ]; then
    rm -rf /workspace/$WORKSPACE_NAME
fi
mkdir -p /workspace
# Determine source directory: SWE-bench Pro uses /app, others use /testbed
if [ -d /app ]; then
    # SWE-bench Pro: codebase is in /app
    cp -r /app /workspace/$WORKSPACE_NAME
elif [ -d /testbed ]; then
    # Standard SWE-bench: codebase is in /testbed
    cp -r /testbed /workspace/$WORKSPACE_NAME
else
    echo "Error: Neither /app nor /testbed directory found." >&2
    exit 1
fi

# Activate instance-specific environment
if [ -d /opt/miniconda3 ]; then
    . /opt/miniconda3/etc/profile.d/conda.sh
    conda activate testbed
fi
