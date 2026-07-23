#!/usr/bin/env bash

set -u

required_commands=(git curl ps top vmstat iostat ss lsof gcc g++ make uv docker)
optional_commands=(jq stress-ng nvidia-smi cmake)
missing_required=0

printf 'AI Infra lab preflight\n'
printf 'timestamp=%s\n' "$(date --iso-8601=seconds 2>/dev/null || date)"
printf 'kernel=%s\n' "$(uname -srmo)"
printf 'workspace=%s\n' "$(pwd)"

if [[ "$(pwd)" == /mnt/* ]]; then
  printf 'FAIL workspace is under /mnt; use the WSL Linux filesystem\n'
  missing_required=1
else
  printf 'PASS workspace is on the Linux filesystem\n'
fi

for command_name in "${required_commands[@]}"; do
  if command -v "$command_name" >/dev/null 2>&1; then
    printf 'PASS required command: %s\n' "$command_name"
  else
    printf 'FAIL required command: %s\n' "$command_name"
    missing_required=1
  fi
done

for command_name in "${optional_commands[@]}"; do
  if command -v "$command_name" >/dev/null 2>&1; then
    printf 'PASS optional command: %s\n' "$command_name"
  else
    printf 'INFO optional command unavailable: %s\n' "$command_name"
  fi
done

if [[ -x .venv/bin/python ]]; then
  python_version="$(.venv/bin/python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
  if [[ "$python_version" == "3.12" ]]; then
    printf 'PASS project Python: %s\n' "$python_version"
  else
    printf 'FAIL project Python is %s; expected 3.12\n' "$python_version"
    missing_required=1
  fi
else
  printf 'FAIL .venv/bin/python not found\n'
  missing_required=1
fi

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  printf 'PASS Docker daemon is reachable\n'
else
  printf 'FAIL Docker daemon is unavailable\n'
  missing_required=1
fi

printf 'memory:\n'
free -h 2>/dev/null || true
printf 'disk:\n'
df -h . 2>/dev/null || true
printf 'git_status:\n'
git status --short 2>/dev/null || true

if (( missing_required != 0 )); then
  printf 'RESULT=FAIL\n'
  exit 1
fi

printf 'RESULT=PASS\n'
