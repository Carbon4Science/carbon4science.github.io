# Contributing to Carbon4Science

Thank you for your interest in contributing to The Carbon Cost of Generative AI for Science!

## How to Contribute

### Reporting Issues

- Use the [GitHub Issues](https://github.com/shuan4638/Carbon4Science/issues) to report bugs or request features
- Include your hardware configuration, Python version, and steps to reproduce

### Adding a New Model

1. **Create a subdirectory** under the appropriate task folder:
   ```
   Retrosynthesis/YourModel/
   MolGen/YourModel/
   MatGen/YourModel/
   ```

2. **Include required files**:
   - `README.md` - Model description, paper reference, setup instructions
   - `CLAUDE.md` - AI assistant guidance for the module
   - `Inference.py` - Standardized inference interface with `run()` function
   - `requirements.txt` or `environment.yml` - Dependencies

3. **Implement the standard interface**:
   ```python
   # Inference.py
   def run(smiles, **kwargs):
       """
       Run inference on input SMILES.

       Args:
           smiles: Single SMILES string or list of SMILES
           **kwargs: Model-specific parameters

       Returns:
           List of predictions (format depends on task)
       """
       pass
   ```

4. **Add carbon tracking** to your training script:
   ```python
   from benchmarks.carbon_tracker import CarbonTracker

   tracker = CarbonTracker(
       project_name="yourmodel_training",
       model_name="YourModel",
       task="training"
   )

   with tracker:
       train_model()

   tracker.add_accuracy(top1=accuracy)
   ```

5. **Run benchmarks** and submit results:
   ```bash
   python benchmarks/evaluate_retrosynthesis.py --model YourModel --runs 3
   ```

6. **Submit a pull request** with:
   - Model implementation
   - Benchmark results in `benchmarks/results/`
   - Updated documentation

### Contributing Benchmark Results

If you have hardware different from the original benchmarks:

1. Copy `benchmarks/configs/hardware_template.yaml` to `benchmarks/configs/hardware_yourname.yaml`
2. Fill in your hardware specifications
3. Run the standard benchmark protocol
4. Submit results via pull request

### Code Style

- Python: Follow PEP 8
- Use type hints where practical
- Document public functions with docstrings

### Task Assignments

| Task | Lead | Status |
|------|------|--------|
| Retrosynthesis | @shuan4638 | Active |
| Molecule Generation | TBD | Planned |
| Material Generation | TBD | Planned |

## Questions?

Open an issue or contact the maintainers.
