# Visualizer Module - Quick Reference

## Overview
The Visualizer class generates professional, publication-ready visualizations for the sentiment analysis project.

## Quick Start

```python
from visualizer import Visualizer

# Initialize with professional theme
viz = Visualizer()

# Generate all plots
viz.plot_metrics_comparison(results, metrics=['accuracy', 'f1_macro'])
viz.plot_boxplots(simulations_df, metric='f1_macro')
viz.plot_line_evolution(simulations_df, metric='f1_macro')
viz.plot_pvalue_matrix(pvalue_matrix, model_names)
viz.plot_confidence_intervals(ci_results, metric='f1_macro')
viz.plot_confusion_matrix(cm, 'svm_bow')

# Save all at 300 DPI
viz.save_all_figures('results/plots', dpi=300)
viz.close_all()
```

## Available Methods

| Method | Purpose | Requirements |
|--------|---------|--------------|
| `plot_metrics_comparison()` | Grouped bar chart | Req 9.2 |
| `plot_confusion_matrix()` | Heatmap confusion matrix | Req 9.3 |
| `plot_boxplots()` | Distribution across simulations | Req 9.4 |
| `plot_line_evolution()` | Metric evolution over time | Req 9.5 |
| `plot_pvalue_matrix()` | Statistical significance matrix | Req 9.6 |
| `plot_confidence_intervals()` | Bars with 95% CI | Req 9.7 |
| `save_all_figures()` | Export PNG at 300 DPI | Req 9.11 |

## Input Data Formats

### For metrics comparison:
```python
results = {
    'svm_bow': {'accuracy': 0.796, 'f1_macro': 0.795},
    'bert': {'accuracy': 0.845, 'f1_macro': 0.844}
}
```

### For simulations (boxplots, line evolution):
```python
simulations_df = pd.DataFrame({
    'simulation_id': [0, 1, 2, ...],
    'model_name': ['svm_bow', 'svm_bow', ...],
    'accuracy': [0.795, 0.797, ...],
    'f1_macro': [0.794, 0.796, ...]
})
```

### For p-value matrix:
```python
pvalue_matrix = np.array([
    [1.0, 0.02, 0.001],
    [0.02, 1.0, 0.03],
    [0.001, 0.03, 1.0]
])
model_names = ['svm_bow', 'svm_embeddings', 'bert']
```

### For confidence intervals:
```python
ci_results = {
    'svm_bow': {'mean': 0.795, 'std': 0.01},
    'bert': {'mean': 0.844, 'std': 0.008}
}
```

### For confusion matrix:
```python
cm = np.array([[450, 50], [30, 470]])  # 2x2 matrix
```

## Color Coding

### P-value Matrix:
- 🟢 **Green** (p < 0.05): Significant difference between models
- 🔴 **Red** (p ≥ 0.05): No significant difference

### General:
- Uses seaborn 'Set2' color palette
- Consistent colors across all plots
- Professional whitegrid theme

## Output

All figures are saved as:
- **Format:** PNG
- **Resolution:** 300 DPI (high quality for presentations)
- **Location:** `results/plots/` (configurable)
- **Naming:** Descriptive names (e.g., `metrics_comparison.png`, `boxplot_f1_macro.png`)

## Testing

Run tests with:
```bash
python -m pytest tests/unit/test_visualizer.py -v
```

All 13 tests pass ✅

## Integration

Works seamlessly with:
- **SimulationRunner**: Accepts simulation results DataFrames
- **StatisticalValidator**: Uses p-value matrices
- **Evaluator**: Uses metrics dictionaries and confusion matrices

## Example Output Files

1. `metrics_comparison.png` - Compare all models side-by-side
2. `confusion_matrix_svm_bow.png` - Heatmap for SVM+BoW
3. `confusion_matrix_bert.png` - Heatmap for BERT
4. `boxplot_f1_macro.png` - F1 distribution across simulations
5. `line_evolution_f1_macro.png` - F1 evolution over simulations
6. `pvalue_matrix_f1_macro.png` - Statistical significance
7. `confidence_intervals_f1_macro.png` - F1 with 95% CI

## Tips

1. **Generate all plots before saving**: Call all `plot_*()` methods first, then `save_all_figures()` once
2. **Clean up memory**: Always call `close_all()` after saving
3. **Custom styling**: Pass `style` and `figsize` to constructor
4. **Multiple metrics**: Generate separate plots for each metric (accuracy, f1_macro, etc.)
5. **High DPI**: Use `dpi=300` for presentations, `dpi=150` for quick previews

## Troubleshooting

**Issue:** Figures not saving  
**Solution:** Ensure you've called `plot_*()` methods before `save_all_figures()`

**Issue:** Memory warnings  
**Solution:** Call `close_all()` after saving figures

**Issue:** Style not applied  
**Solution:** Check matplotlib/seaborn versions, fallback to default theme

**Issue:** Empty plots  
**Solution:** Verify input data format matches expected structure
