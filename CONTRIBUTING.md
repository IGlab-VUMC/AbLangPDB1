# Contributing to AbLangPDB1

Thank you for your interest in contributing to AbLangPDB1! We welcome contributions from the community to help improve this antibody embedding model and its applications.

## 🚀 Ways to Contribute

### 1. 🐛 Bug Reports
- Report bugs through GitHub Issues
- Include detailed reproduction steps
- Provide system information (OS, Python version, GPU details)
- Include error messages and stack traces

### 2. ✨ Feature Requests
- Suggest new features or improvements
- Explain the use case and expected benefits
- Discuss implementation approaches if possible

### 3. 🔧 Code Contributions
- Bug fixes
- Performance improvements
- New benchmarking models
- Documentation improvements
- New example notebooks

### 4. 📊 Benchmarking Contributions
- Add new antibody embedding models for comparison
- Contribute new evaluation datasets
- Improve evaluation metrics
- Add visualization tools

## 🛠️ Development Setup

### Prerequisites
```bash
# Clone the repository
git clone https://github.com/your-username/AbLangPDB1.git
cd AbLangPDB1

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install development dependencies
pip install -e ".[dev,benchmarking]"
```

### Development Dependencies
```bash
pip install pytest black flake8 isort jupyter
```

## 📝 Code Style Guidelines

### Python Code Style
- Follow PEP 8 conventions
- Use Black for code formatting: `black .`
- Use isort for import sorting: `isort .`
- Run flake8 for linting: `flake8 .`
- Maximum line length: 88 characters (Black default)

### Documentation Style
- Use clear, concise docstrings for all functions and classes
- Follow Google docstring format
- Include type hints for function parameters and returns
- Update README.md for significant changes

### Example Code Style

```python
from typing import List, Optional, Tuple
import torch
import pandas as pd

def embed_antibodies(
    heavy_chains: List[str], 
    light_chains: List[str],
    model: torch.nn.Module,
    batch_size: int = 256
) -> torch.Tensor:
    """Generate embeddings for antibody sequences.
    
    Args:
        heavy_chains: List of heavy chain amino acid sequences
        light_chains: List of light chain amino acid sequences  
        model: Trained AbLangPDB1 model
        batch_size: Number of sequences to process at once
        
    Returns:
        Tensor of shape (N, 1536) containing embeddings
        
    Raises:
        ValueError: If heavy and light chain lists have different lengths
    """
    if len(heavy_chains) != len(light_chains):
        raise ValueError("Heavy and light chain lists must have same length")
    
    # Implementation here...
    return embeddings
```

## 🧪 Testing Guidelines

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ablangpdb1

# Run specific test file
pytest tests/test_model.py
```

### Writing Tests
- Write tests for all new functions and classes
- Use descriptive test names: `test_embed_antibodies_with_invalid_sequences`
- Include edge cases and error conditions
- Mock external dependencies (HuggingFace downloads, GPU operations)

### Test Structure
```python
import pytest
import torch
from ablangpaired_model import AbLangPaired, AbLangPairedConfig

class TestAbLangPaired:
    def test_model_initialization(self):
        """Test that model initializes correctly."""
        config = AbLangPairedConfig()
        model = AbLangPaired(config, torch.device("cpu"))
        assert model is not None
        
    def test_embedding_generation(self):
        """Test that embeddings are generated with correct shape."""
        # Test implementation
        pass
```

## 🔀 Pull Request Process

### Before Submitting
1. **Fork the repository** and create a feature branch
2. **Run all tests** and ensure they pass
3. **Run linting tools** (black, flake8, isort)
4. **Update documentation** if adding new features
5. **Add tests** for new functionality

### Pull Request Guidelines
1. **Clear title and description**
   - Summarize what the PR does
   - Reference related issues with `Fixes #123`
   
2. **Small, focused changes**
   - One feature or fix per PR
   - Break large changes into multiple PRs
   
3. **Test coverage**
   - Maintain or improve test coverage
   - Include both positive and negative test cases
   
4. **Documentation updates**
   - Update README.md for user-facing changes
   - Update docstrings for API changes
   - Add examples for new features

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated documentation

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Tests added/updated
- [ ] Documentation updated
```

## 🏗️ Adding New Models to Benchmarking

### Model Integration Process
1. **Implement embedding function** in `benchmarking/models.py`
2. **Generate embeddings** for SAbDab and DMS datasets
3. **Save as parquet files** following naming convention
4. **Test with existing benchmarking scripts**
5. **Update documentation** with model details

### Example Model Addition
```python
# In benchmarking/models.py
class YourModelEmbedder:
    def __init__(self, model_path: str, batch_size: int = 256):
        self.model = load_your_model(model_path)
        self.batch_size = batch_size
    
    def embed(self, df: pd.DataFrame) -> List[np.ndarray]:
        """Generate embeddings for antibody sequences."""
        embeddings = []
        for batch in self._create_batches(df):
            batch_embeddings = self.model.encode(batch)
            embeddings.extend(batch_embeddings)
        return embeddings
```

## 📊 Dataset Contributions

### Adding New Evaluation Datasets
1. **Prepare dataset** in consistent format
2. **Generate reference embeddings** for all models
3. **Create evaluation script** following existing patterns
4. **Document dataset characteristics** and evaluation metrics
5. **Submit PR** with dataset and evaluation code

### Dataset Format Requirements
- **Antibody sequences**: `HC_AA` and `LC_AA` columns
- **Labels**: Clear positive/negative labels for epitope overlap
- **Metadata**: Include relevant antibody and antigen information
- **Documentation**: Describe dataset source, size, and characteristics

## 🌟 Recognition

Contributors will be acknowledged in:
- Repository contributors list
- Release notes for significant contributions  
- Future publications (for substantial contributions)

## 📞 Getting Help

- **GitHub Discussions**: For questions about contributing
- **Issues**: For bug reports and feature requests
- **Email**: clinton.m.holt@vanderbilt.edu for complex questions

## 📄 Code of Conduct

### Our Standards
- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain professional communication

### Unacceptable Behavior
- Harassment or discrimination
- Trolling or inflammatory comments
- Publishing private information
- Other unprofessional conduct

## 📚 Resources

- [GitHub Flow Guide](https://guides.github.com/introduction/flow/)
- [Python Code Style Guide](https://pep8.org/)
- [Writing Good Commit Messages](https://chris.beams.io/posts/git-commit/)
- [AbLangPDB1 Paper](https://doi.org/10.1101/2025.02.25.640114)

Thank you for contributing to AbLangPDB1! 🧬✨