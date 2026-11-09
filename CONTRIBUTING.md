# Contributing

Keep Barebones small enough to inspect. Prefer changes that clarify numerical behavior or
add a focused invariant over framework-like abstractions.

## Verification

Run from the repository root:

```sh
python -m unittest discover -s . -p 'test_*.py' -v
python main.py
```

Add deterministic tests for behavior changes and finite-difference checks for gradient
changes. Keep generated output, caches, and local environments out of commits.

Use [SECURITY.md](SECURITY.md) for security-sensitive reports.
