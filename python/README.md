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

### Dry-run CI locally

From the **repository root** (same layout as GitHub Actions):

```bash
# macOS wheel (needs python.org CPython framework, e.g. 3.9 from python.org)
uv tool run cibuildwheel --platform macos --output-dir wheelhouse python

# Linux wheel — closest match to CI; requires Docker
uv tool run cibuildwheel --platform linux --output-dir wheelhouse python

# Windows wheel — only on Windows
uv tool run cibuildwheel --platform windows --output-dir wheelhouse python
```

Note: the GitHub Action uses `package-dir: python`; the CLI equivalent is the positional `python` argument at the end (not `--package-dir`).

On macOS outside CI, cibuildwheel expects official [python.org](https://www.python.org/downloads/macos/) framework installs. If you only have Homebrew/uv Python, either install the python.org pkg or use the simpler dev loop below.

**Quick local check without cibuildwheel** (builds a wheel for your current machine only):

```bash
cd python
uv sync
uv run maturin build --release
uv run pytest -k "not symbolic"
```

**Smoke-test a built wheel:**

```bash
uv venv /tmp/simsam-wheel-test
source /tmp/simsam-wheel-test/bin/activate
pip install wheelhouse/simsam_rs-*.whl   # or python/target/wheels/*.whl
pytest python/tests -k "not symbolic"
```

Skip cibuildwheel tests while debugging build-only issues:

```bash
CIBW_TEST_SKIP='*' uv tool run cibuildwheel --platform macos --output-dir wheelhouse python
```

### Optional symbolic builds (not published to PyPI by default)

```bash
uv run maturin develop --release --features symbolic,pyo3/extension-module
```
