#!/usr/bin/env python3
"""
Filter instances based on CSV selection.
Reads a CSV with 'original_inst_id' column and checks if an instance ID is selected.
"""
import csv
import sys

def load_selected_ids(csv_path):
    """Load selected original_inst_ids from CSV."""
    selected = set()
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            original_id = row.get('original_inst_id', '').strip()
            if original_id:
                selected.add(original_id)
    return selected

def main():
    if len(sys.argv) != 3:
        print("Usage: filter_instances.py <csv_file> <instance_id>", file=sys.stderr)
        sys.exit(1)
    
    csv_file = sys.argv[1]
    instance_id = sys.argv[2]
    
    selected_ids = load_selected_ids(csv_file)
    
    # Check if instance is selected
    if instance_id in selected_ids:
        sys.exit(0)  # Selected
    else:
        sys.exit(1)  # Not selected

if __name__ == '__main__':
    main()
