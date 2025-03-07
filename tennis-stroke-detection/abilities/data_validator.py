#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
data_validator.py

Provides functionality to validate that dataset files meet certain
format and content requirements. For instance, verifying CSV headers,
checking row counts, or ensuring numeric columns are within valid ranges.
"""

import os
import csv

def validate_csv_structure(csv_path, required_columns):
    """
    Checks if the CSV at csv_path contains at least the columns
    in required_columns.
    
    :param csv_path: Path to the CSV file.
    :param required_columns: A list of column names expected in the CSV header.
    :raises FileNotFoundError: If the CSV file doesn't exist.
    :raises ValueError: If the CSV lacks any of the required columns.
    :returns: True if validation succeeds.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"CSV contains no header: {csv_path}")

        missing = [col for col in required_columns if col not in header]
        if missing:
            raise ValueError(f"CSV {csv_path} missing required columns: {missing}")

    return True

def validate_row_count(csv_path, min_rows=1):
    """
    Ensures the CSV has at least 'min_rows' of data (excluding the header).
    
    :param csv_path: Path to the CSV file.
    :param min_rows: Minimum acceptable row count, not counting the header row.
    :raises ValueError: If the row count is less than min_rows.
    :returns: The actual row count if successful.
    """
    row_count = 0
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for _ in reader:
            row_count += 1
    if row_count < min_rows:
        raise ValueError(f"Not enough rows in {csv_path}: found {row_count}, expected >= {min_rows}")
    return row_count