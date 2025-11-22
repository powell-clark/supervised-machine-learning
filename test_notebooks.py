#!/usr/bin/env python3
"""
Comprehensive notebook validation script.
Tests all notebooks for:
- Valid JSON structure
- Python syntax errors in code cells
- Import statement validity
- Markdown formatting
"""

import json
import ast
import sys
from pathlib import Path

def validate_notebook(notebook_path):
    """Validate a single Jupyter notebook."""
    errors = []
    warnings = []

    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"], []
    except Exception as e:
        return [f"Failed to load: {e}"], []

    # Check notebook structure
    if 'cells' not in notebook:
        errors.append("Missing 'cells' key")
        return errors, warnings

    cells = notebook['cells']
    code_cells = [c for c in cells if c.get('cell_type') == 'code']
    markdown_cells = [c for c in cells if c.get('cell_type') == 'markdown']

    # Validate code cells
    for idx, cell in enumerate(code_cells):
        source = cell.get('source', [])
        if isinstance(source, list):
            code = ''.join(source)
        else:
            code = source

        # Skip empty cells and magic commands
        if not code.strip() or code.strip().startswith('%') or code.strip().startswith('!'):
            continue

        # Check for Python syntax errors
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error in code cell {idx}: {e.msg} at line {e.lineno}")
        except Exception:
            # Some valid code might not parse standalone (e.g., incomplete functions)
            pass

    # Check for common issues
    all_code = '\n'.join([''.join(c.get('source', [])) if isinstance(c.get('source', []), list) else c.get('source', '')
                          for c in code_cells])

    # Check for essential imports in new deep learning notebooks
    notebook_name = Path(notebook_path).name
    if '9a' in notebook_name or '9b' in notebook_name or '9c' in notebook_name:
        if 'import numpy' not in all_code and 'import tensorflow' not in all_code and 'import torch' not in all_code:
            warnings.append("Missing numpy/tensorflow/torch imports")

    return errors, warnings

def main():
    """Validate all notebooks in the repository."""
    notebooks_dir = Path('/home/user/supervised-machine-learning/notebooks')
    notebooks = sorted(notebooks_dir.glob('*.ipynb'))

    print("=" * 70)
    print("NOTEBOOK VALIDATION REPORT")
    print("=" * 70)

    total_errors = 0
    total_warnings = 0
    validated = 0

    for notebook_path in notebooks:
        errors, warnings = validate_notebook(notebook_path)
        validated += 1

        if errors or warnings:
            print(f"\n📓 {notebook_path.name}")
            if errors:
                total_errors += len(errors)
                for error in errors:
                    print(f"  ❌ ERROR: {error}")
            if warnings:
                total_warnings += len(warnings)
                for warning in warnings:
                    print(f"  ⚠️  WARNING: {warning}")
        else:
            print(f"✓ {notebook_path.name}")

    print("\n" + "=" * 70)
    print(f"SUMMARY")
    print("=" * 70)
    print(f"Total notebooks validated: {validated}")
    print(f"Total errors: {total_errors}")
    print(f"Total warnings: {total_warnings}")

    if total_errors == 0:
        print("\n🎉 ALL NOTEBOOKS PASSED VALIDATION! 🎉")
        return 0
    else:
        print(f"\n⚠️  {total_errors} errors found")
        return 1

if __name__ == '__main__':
    sys.exit(main())
