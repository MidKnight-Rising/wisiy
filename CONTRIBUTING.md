# Contributing to Smoke Runtime

Thank you for your interest in contributing to Smoke Runtime! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions. We're building this together.

## Getting Started

### Development Setup

1. **Fork and clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/wisiy.git
cd wisiy
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode**
```bash
pip install -e ".[dev]"
```

4. **Run tests to verify setup**
```bash
pytest
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions/fixes

### 2. Make Changes

- Write clean, readable code
- Follow existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=smoke_runtime

# Run specific test file
pytest tests/test_cache.py -v

# Run tests matching pattern
pytest -k "test_cache"
```

### 4. Format and Lint

```bash
# Format code with black
black smoke_runtime/ tests/ examples/

# Lint with flake8
flake8 smoke_runtime/ tests/ examples/

# Type check with mypy (optional)
mypy smoke_runtime/
```

### 5. Commit Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add feature: description of what you did"
```

Good commit message format:
```
Short summary (50 chars or less)

More detailed explanation if needed. Wrap at 72 characters.
Explain what and why, not how.

- Bullet points are okay
- Use present tense: "Add feature" not "Added feature"
```

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title describing the change
- Description of what was changed and why
- Reference to any related issues
- Screenshots for UI changes

## Contribution Guidelines

### Code Style

- Follow PEP 8 style guide
- Use Black for formatting (line length: 88)
- Use type hints where appropriate
- Write docstrings for public functions/classes

Example:
```python
def function_name(param1: str, param2: int) -> bool:
    """Brief description of function.
    
    More detailed description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something is wrong
    """
    pass
```

### Testing

- Write tests for all new features
- Maintain or improve code coverage
- Use pytest for testing
- Test both success and failure cases

Test structure:
```python
class TestComponentName:
    """Test ComponentName class."""
    
    def test_basic_functionality(self):
        """Test basic use case."""
        # Arrange
        component = Component()
        
        # Act
        result = component.method()
        
        # Assert
        assert result == expected
    
    def test_edge_case(self):
        """Test edge case handling."""
        pass
```

### Documentation

- Update README.md for user-facing changes
- Add docstrings to all public APIs
- Update examples/ if adding new features
- Update docs/ for architectural changes

### Performance

- Profile performance-critical code
- Avoid unnecessary allocations
- Use appropriate data structures
- Consider memory implications

## Types of Contributions

### Bug Fixes

1. Check if issue exists, if not create one
2. Reference issue in PR
3. Add test that reproduces bug
4. Fix the bug
5. Verify test passes

### New Features

1. Discuss in an issue first
2. Follow roadmap priorities
3. Include tests and documentation
4. Update examples if applicable

### Documentation

- Fix typos and clarify unclear sections
- Add examples and use cases
- Improve API documentation
- Create tutorials

### Tests

- Increase test coverage
- Add missing test cases
- Improve test quality

## Review Process

1. **Automated checks**: Tests, linting, type checking
2. **Code review**: Maintainers review code
3. **Feedback**: Address review comments
4. **Approval**: Once approved, PR is merged

### Review Criteria

- Code quality and style
- Test coverage
- Documentation completeness
- Performance considerations
- Backward compatibility

## Areas Needing Help

See [ROADMAP.md](docs/ROADMAP.md) for planned work. Priority areas:

1. **Model Integration**: HuggingFace Transformers support
2. **Testing**: More comprehensive tests
3. **Documentation**: Examples and tutorials
4. **Performance**: Profiling and optimization
5. **Features**: Quantization, multi-GPU support

## Questions?

- **General questions**: Open a GitHub Discussion
- **Bug reports**: Open an Issue
- **Feature requests**: Open an Issue with [Feature Request] tag
- **Security issues**: Email maintainers directly

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in documentation

Thank you for contributing to Smoke Runtime! 🚀
