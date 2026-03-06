#!/usr/bin/env bash
# deploy-galaxy.sh — Backward-compatible wrapper for deploy.sh
#
# Delegates to deploy.sh with --device galaxy. All arguments are passed through.
#
# Usage (unchanged):
#   ./deploy-galaxy.sh                           # run mode (starts vLLM)
#   ./deploy-galaxy.sh --mode diagnostics        # diagnostics only (no vLLM)
#   ./deploy-galaxy.sh --mode shell              # interactive shell with tt-tools
exec "$(dirname "$0")/deploy.sh" --device galaxy "$@"
