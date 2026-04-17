# Contributing to Email Spam Detector

Thank you for your interest in contributing! This document provides guidelines and information for contributors.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How to Contribute

### Reporting Bugs

- Check existing issues to see if the bug has already been reported
- Create a new issue with:
  - Clear title and description
  - Steps to reproduce
  - Expected vs actual behavior
  - Environment details (OS, Python version, etc.)
  - Logs or error messages

### Suggesting Features

- Open an issue to discuss proposed features
- Provide clear use cases and examples
- Consider backwards compatibility

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Follow the coding standards (see below)
4. Add or update tests for changes
5. Ensure all tests pass (`make test`)
6. Commit with clear, descriptive messages
7. Push to your fork (`git push origin feature/AmazingFeature`)
8. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/email-spam-detector.git
cd email-spam-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install-dev

# Install pre-commit hooks (optional but recommended)
make pre-commit-install

# Run tests to verify setup
make test
```

## Code Standards

### Python Style

- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://github.com/psf/black) for formatting (line length: 88)
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use [flake8](https://flake8.pycqa.org/) for linting
- Run `make format` before committing

### Type Hints

- All functions and methods should have type hints
- Use mypy for type checking (`make lint`)

### Documentation

- All public functions, classes, and modules should have docstrings
- Use Google-style or NumPy-style docstrings
- Update README.md for user-facing changes

### Tests

- Write tests for new features and bug fixes
- Aim for high test coverage
- Place tests in `tests/` with naming `test_*.py`
- Use pytest fixtures for common setup

## Project Structure

```
email-spam-detector/
├── src/spam_detector/    # Main package source code
│   ├── api/              # FastAPI endpoints
│   ├── core/             # Configuration, logging, exceptions
│   ├── db/               # Database/mongo integration
│   ├── ml/               # ModelService and ML utilities
│   ├── nlp/              # Text preprocessing
│   ├── features/         # Feature extraction
│   ├── training/         # Model training pipeline
│   ├── learning/         # Self-learning and feedback
│   ├── advanced/         # LIME, phishing detection, multilingual
│   └── gmail/            # Gmail API integration
├── scripts/              # Command-line entry points
├── tests/                # Test suite
├── models/               # Trained model artifacts
├── data/                 # Training data and feedback
├── frontend/             # Frontend web interface
├── .github/workflows/    # CI/CD pipelines
└── docs/                 # Documentation (future)
```

## Commit Messages

Follow [conventional commits](https://www.conventionalcommits.org/):

```
type(scope): description

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation changes
- style: Code style changes (formatting, etc.)
- refactor: Code refactoring
- test: Adding or updating tests
- chore: Maintenance tasks
- perf: Performance improvements
```

Examples:
```
feat(api): add /dashboard/feedback-analytics endpoint
fix(ml): correct probability threshold calculation
docs(readme): add installation troubleshooting section
test(api): add tests for predict endpoint
```

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create release branch
4. Run full test suite and linting
5. Merge to main and tag release
6. Create GitHub release with notes

## Questions?

Contact: prasannapbhurke@github.com or open an issue.

---

Happy contributing! 🚀
