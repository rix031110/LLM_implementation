#!/usr/bin/env python3
"""
Fix Jupyter notebook widget metadata issues.
Adds missing 'state' key to metadata.widgets entries.
"""
import json
import sys


def fix_notebook(notebook_path):
    """Fix notebook widget metadata."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error reading notebook: {e}")
        return False
    
    # Check if notebook has metadata.widgets
    if 'metadata' not in notebook:
        notebook['metadata'] = {}
    
    if 'widgets' in notebook['metadata']:
        widgets = notebook['metadata']['widgets']
        
        # Ensure each widget has a 'state' key
        if isinstance(widgets, dict):
            for widget_id, widget_data in widgets.items():
                if isinstance(widget_data, dict) and 'state' not in widget_data:
                    widget_data['state'] = {}
                    print(f"Added 'state' to widget: {widget_id}")
    
    # Write fixed notebook back
    try:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print(f"✓ Successfully fixed notebook: {notebook_path}")
        return True
    except Exception as e:
        print(f"Error writing notebook: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_notebook.py <notebook_path>")
        sys.exit(1)
    
    notebook_path = sys.argv[1]
    success = fix_notebook(notebook_path)
    sys.exit(0 if success else 1)
