# Contributing to Inkforge

Thank you for your interest in contributing to Inkforge! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Bugs

1. Check existing [issues](https://github.com/SarmaHighOnCode/INKFORGE/issues) to avoid duplicates
2. Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md)
3. Include reproduction steps, expected vs actual behavior, and environment details

### Suggesting Features

1. Open a [feature request](.github/ISSUE_TEMPLATE/feature_request.md)
2. Describe the use case, not just the solution
3. Include mockups or examples where possible

### Pull Requests

1. Fork the repo and create a branch from `main`
2. Follow the naming convention: `feat/`, `fix/`, `docs/`, `refactor/`
3. Write clear, descriptive commit messages
4. Add tests for new functionality
5. Ensure all tests pass before submitting
6. Fill out the [PR template](.github/PULL_REQUEST_TEMPLATE.md)

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/INKFORGE.git
cd INKFORGE

# Backend
cd backend
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Frontend
cd ../frontend
npm install
```

## Code Style

### Python (Backend + ML)
- Follow **PEP 8**
- Use **type hints** for all function signatures
- Write **docstrings** (Google style) for public functions
- Format with `ruff format`, lint with `ruff check`
- Maximum line length: 100 characters

### JavaScript/JSX (Frontend)
- Follow **ESLint** configuration in the project
- Format with **Prettier**
- Use functional components with hooks
- Use descriptive variable and function names

### Commits
- Use [Conventional Commits](https://www.conventionalcommits.org/):
  - `feat:` new feature
  - `fix:` bug fix
  - `docs:` documentation
  - `test:` adding tests
  - `refactor:` code refactoring
  - `chore:` maintenance tasks

## Testing

```bash
# Backend tests
cd backend
pytest tests/ -v

# Frontend tests (when configured)
cd frontend
npm test
```

## Questions?

Open a [discussion](https://github.com/SarmaHighOnCode/INKFORGE/discussions) or reach out via issues.

---

Thank you for helping make Inkforge better! 🖊
