# robust-mvo

A small mean-variance optimizer for one rebalance of a long-short book, built around numerical stability and explicit failure handling. Work in progress; see `SPEC.md` for scope.

## Development

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pytest
```
