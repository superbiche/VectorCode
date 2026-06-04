# `mcp__vectorcode__query` returns empty results

**Date:** 2026-02-24
**Source:** backlog (migrated from `~/.claude/backlog.md`)
**Affects:** VectorCode (superbiche fork) — MCP server

## Observed
An MCP parsing bug causes `mcp__vectorcode__query` to return empty results. The USearch adapter is ready in the `superbiche/VectorCode` fork; upstream PR #282 is still open.

## Why it matters
The MCP query path is currently unusable; CLI-based routing is the stopgap workaround.

## Suggested action
Investigate and fix the MCP parsing bug — or evaluate whether CLI-based routing is good enough to not bother fixing it. Track upstream PR #282.
