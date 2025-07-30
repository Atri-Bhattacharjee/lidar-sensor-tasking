"""
Optimal Subpattern Assignment (OSPA) Distance Calculator

This module provides a function to calculate the OSPA distance between two sets of objects,
which is used to evaluate the performance of multi-object tracking algorithms.

The OSPA distance combines localization error and cardinality error into a single metric.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple


def calculate_ospa(ground_truth_set: List[Dict], estimated_set: List[Dict], 
                   cutoff_c: float, order_p: float = 1.0) -> float:
    """
    Calculate the Optimal Subpattern Assignment (OSPA) distance between two sets of objects.
    
    Args:
        ground_truth_set: List of dictionaries containing ground truth objects.
                         Each dict should have 'mean' key with 6D state vector [Px, Py, Pz, Vx, Vy, Vz]
        estimated_set: List of dictionaries containing estimated objects.
                      Each dict should have 'mean' key with 6D state vector [Px, Py, Pz, Vx, Vy, Vz]
        cutoff_c: Cutoff distance parameter (in meters)
        order_p: Order parameter for the OSPA calculation (default: 1.0)
    
    Returns:
        float: The OSPA distance between the two sets
    """
    
    # Handle empty sets
    if len(ground_truth_set) == 0 and len(estimated_set) == 0:
        return 0.0
    
    if len(ground_truth_set) == 0:
        # Only estimated objects exist (false alarms)
        return cutoff_c
    
    if len(estimated_set) == 0:
        # Only ground truth objects exist (missed detections)
        return cutoff_c
    
    # Extract position vectors (first 3 components) from state vectors
    # Handle both ground truth (with 'position') and estimated (with 'mean') objects
    gt_positions = np.array([obj.get('position', obj.get('mean', [0, 0, 0]))[:3] for obj in ground_truth_set])
    est_positions = np.array([obj.get('mean', obj.get('position', [0, 0, 0]))[:3] for obj in estimated_set])
    
    # Calculate distance matrix between all pairs
    # Use broadcasting to compute pairwise Euclidean distances
    diff = gt_positions[:, np.newaxis, :] - est_positions[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    
    # Apply cutoff
    distances = np.minimum(distances, cutoff_c)
    
    # Use Hungarian algorithm to find optimal assignment
    row_indices, col_indices = linear_sum_assignment(distances)
    
    # Calculate localization error for matched pairs
    matched_distances = distances[row_indices, col_indices]
    localization_error = np.sum(matched_distances**order_p)
    
    # Calculate cardinality error
    num_unmatched = abs(len(ground_truth_set) - len(estimated_set))
    cardinality_error = num_unmatched * (cutoff_c**order_p)
    
    # Combine errors
    total_error = localization_error + cardinality_error
    
    # Normalize by the maximum cardinality
    max_cardinality = max(len(ground_truth_set), len(estimated_set))
    
    if max_cardinality == 0:
        return 0.0
    
    ospa_distance = (total_error / max_cardinality)**(1.0 / order_p)
    
    return ospa_distance


def calculate_ospa_components(ground_truth_set: List[Dict], estimated_set: List[Dict], 
                             cutoff_c: float, order_p: float = 1.0) -> Tuple[float, float, float]:
    """
    Calculate the OSPA distance and its components (localization and cardinality errors).
    
    Args:
        ground_truth_set: List of dictionaries containing ground truth objects
        estimated_set: List of dictionaries containing estimated objects
        cutoff_c: Cutoff distance parameter (in meters)
        order_p: Order parameter for the OSPA calculation (default: 1.0)
    
    Returns:
        Tuple[float, float, float]: (OSPA distance, localization error, cardinality error)
    """
    
    # Handle empty sets
    if len(ground_truth_set) == 0 and len(estimated_set) == 0:
        return 0.0, 0.0, 0.0
    
    if len(ground_truth_set) == 0:
        return cutoff_c, 0.0, cutoff_c
    
    if len(estimated_set) == 0:
        return cutoff_c, 0.0, cutoff_c
    
    # Extract position vectors
    # Handle both ground truth (with 'position') and estimated (with 'mean') objects
    gt_positions = np.array([obj.get('position', obj.get('mean', [0, 0, 0]))[:3] for obj in ground_truth_set])
    est_positions = np.array([obj.get('mean', obj.get('position', [0, 0, 0]))[:3] for obj in estimated_set])
    
    # Calculate distance matrix
    diff = gt_positions[:, np.newaxis, :] - est_positions[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    
    # Apply cutoff
    distances = np.minimum(distances, cutoff_c)
    
    # Find optimal assignment
    row_indices, col_indices = linear_sum_assignment(distances)
    
    # Calculate components
    matched_distances = distances[row_indices, col_indices]
    localization_error = np.sum(matched_distances**order_p)
    
    num_unmatched = abs(len(ground_truth_set) - len(estimated_set))
    cardinality_error = num_unmatched * (cutoff_c**order_p)
    
    total_error = localization_error + cardinality_error
    max_cardinality = max(len(ground_truth_set), len(estimated_set))
    
    ospa_distance = (total_error / max_cardinality)**(1.0 / order_p)
    
    # Normalize components for return
    normalized_localization = (localization_error / max_cardinality)**(1.0 / order_p)
    normalized_cardinality = (cardinality_error / max_cardinality)**(1.0 / order_p)
    
    return ospa_distance, normalized_localization, normalized_cardinality 