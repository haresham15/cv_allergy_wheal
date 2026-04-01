"""Allergen-grid ↔ wheal spatial mapping.

The doctor performs a skin prick test in a grid layout on the patient's
back.  The user supplies a mapping like:

    {"A1": "Peanut", "A2": "Dust Mite", "B1": "Cat Dander", ...}

This module spatially assigns each detected wheal to the nearest grid
position so the final report can say:

    "A1 — Peanut — 6.2 mm (mild)"
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def assign_allergens(
    wheals: list,
    grid_labels: Dict[str, str],
    grid_rows: int,
    grid_cols: int,
    image_width: int,
    image_height: int,
    margin_frac: float = 0.1,
) -> list:
    """Assign allergen names to each wheal based on spatial grid position.

    Parameters
    ----------
    wheals : list of WhealResult
        Sorted top-to-bottom, left-to-right (from segmentation).
    grid_labels : dict
        e.g. {"A1": "Peanut", "A2": "Dust Mite", "B1": "Cat Dander"}
    grid_rows : int
        Number of rows in the test grid.
    grid_cols : int
        Number of columns.
    image_width, image_height : int
        Dimensions of the processed image.
    margin_frac : float
        Fraction of image to ignore as margin on each side (0.1 = 10%).

    Returns
    -------
    The same wheals list, with `allergen` attribute filled in.
    """

    if not grid_labels or not wheals:
        return wheals

    # Build expected grid centres in image coordinates
    # We assume the test grid occupies the central region of the image
    x_start = image_width * margin_frac
    x_end = image_width * (1 - margin_frac)
    y_start = image_height * margin_frac
    y_end = image_height * (1 - margin_frac)

    grid_centres: List[Tuple[str, float, float]] = []
    row_labels = [chr(ord("A") + r) for r in range(grid_rows)]

    for r_idx, r_label in enumerate(row_labels):
        cy = y_start + (y_end - y_start) * (r_idx + 0.5) / grid_rows
        for c_idx in range(grid_cols):
            cx = x_start + (x_end - x_start) * (c_idx + 0.5) / grid_cols
            key = f"{r_label}{c_idx + 1}"
            grid_centres.append((key, cx, cy))

    # For each wheal, find the nearest grid cell
    used_keys = set()
    for w in wheals:
        wx, wy = w.center
        best_key = None
        best_dist = float("inf")

        for key, gcx, gcy in grid_centres:
            if key in used_keys:
                continue
            dist = np.hypot(wx - gcx, wy - gcy)
            if dist < best_dist:
                best_dist = dist
                best_key = key

        if best_key is not None:
            used_keys.add(best_key)
            allergen_name = grid_labels.get(best_key, f"Unknown ({best_key})")
            w.allergen = allergen_name
            w.grid_position = best_key  # also store the grid key

    return wheals


def parse_grid_input(grid_data: Optional[Dict[str, str]]) -> Tuple[Dict[str, str], int, int]:
    """Parse and validate the grid spec from the API request.

    Expected format:
        {"A1": "Peanut", "A2": "Dust Mite", "B1": "Cat Dander", ...}

    Returns (labels_dict, num_rows, num_cols).
    """

    if not grid_data:
        return {}, 0, 0

    rows_seen = set()
    cols_seen = set()

    for key in grid_data.keys():
        key = key.strip().upper()
        if len(key) < 2:
            continue
        row_letter = key[0]
        col_num = key[1:]
        if row_letter.isalpha() and col_num.isdigit():
            rows_seen.add(row_letter)
            cols_seen.add(int(col_num))

    num_rows = len(rows_seen) if rows_seen else 0
    num_cols = max(cols_seen) if cols_seen else 0

    # Normalise keys to uppercase
    normalised = {k.strip().upper(): v for k, v in grid_data.items()}

    return normalised, num_rows, num_cols
