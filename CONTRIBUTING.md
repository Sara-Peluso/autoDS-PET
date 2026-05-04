# Contributing
Thank you for your interest in autoDS-PET!
Guidelines to set up a development environment, run tests, and submit contributions are provided in this document.

## Setup

```bash
pip install -e ".[dev]"
pre-commit install
```

## Testing

```bash
pytest tests/ -v
```

## Pull Requests

1. Fork and clone the repo
2. Create a branch (`git checkout -b feature/your-feature`)
3. Make changes and ensure tests pass
4. Push and open a PR