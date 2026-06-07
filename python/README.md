# simsam-rs (Python)

Python bindings for the [simsam](../README.md) Rust crate, built with PyO3 and maturin.

Install from PyPI (import name is still `simsam`):

```bash
pip install simsam-rs
```

```python
import simsam

dist = simsam.from_pdf(lambda x: 2.0 * x, simsam.Interval(0.0, 1.0))
print(dist.mean())
print(dist.sample_n(10_000))
```

## Development

```bash
cd python
uv sync
uv run maturin develop --release
uv run pytest
```

## Publishing to PyPI (maintainers)

PyPI package name: **`simsam-rs`**. User code: **`import simsam`**.

Default wheels do **not** include the optional `symbolic` feature.

### One-off local release

```bash
cd python
uv run maturin build --release
uv run maturin publish --username __token__ --password "$PYPI_API_TOKEN"
```

Or upload artifacts from `dist/` with [twine](https://pypi.org/project/twine/).

### CI release (recommended)

1. On [PyPI](https://pypi.org/), create project **simsam-rs** and configure a [trusted publisher](https://docs.pypi.org/trusted-publishers/) for this GitHub repo (workflow: `publish-pypi.yml`, environment: `pypi`).
2. Push a tag: `git tag py-v0.1.0 && git push origin py-v0.1.0`
3. GitHub Actions builds macOS / Linux / Windows wheels plus an sdist, then uploads to PyPI.

Bump `version` in `python/pyproject.toml` and `python/Cargo.toml` (keep in sync with the root Rust crate when releasing together).

### Optional symbolic builds (not published to PyPI by default)

```bash
uv run maturin develop --release --features symbolic,pyo3/extension-module
```
