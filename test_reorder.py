#!/usr/bin/env python3
"""
Test script for the optimized reorder_vertices function.
Tests that vertices are correctly reordered to counter-clockwise in top view.
"""

import numpy as np
import sys
sys.path.append('/home/ubuntu/CUGER_building')

from cuger.__transform.convexify import Geometry_Option

def test_reorder_vertices():
    """Test the reorder_vertices function with various test cases."""
    
    print("=" * 60)
    print("Testing reorder_vertices function")
    print("=" * 60)
    
    # Test Case 1: Square with counter-clockwise order (should remain unchanged)
    print("\n[Test 1] Counter-clockwise square (should remain unchanged)")
    ccw_square = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    result1 = Geometry_Option.reorder_vertices(ccw_square, is_upward=True)
    print(f"Input:\n{ccw_square}")
    print(f"Output:\n{result1}")
    
    # Calculate signed area for verification
    signed_area1 = calculate_signed_area(result1[:, :2])
    print(f"Signed area: {signed_area1:.4f} (positive = CCW)")
    assert signed_area1 > 0, "Result should be counter-clockwise"
    print("✓ Test 1 passed")
    
    # Test Case 2: Square with clockwise order (should be reversed)
    print("\n[Test 2] Clockwise square (should be reversed to CCW)")
    cw_square = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0]
    ])
    result2 = Geometry_Option.reorder_vertices(cw_square, is_upward=True)
    print(f"Input:\n{cw_square}")
    print(f"Output:\n{result2}")
    
    signed_area2 = calculate_signed_area(result2[:, :2])
    print(f"Signed area: {signed_area2:.4f} (positive = CCW)")
    assert signed_area2 > 0, "Result should be counter-clockwise"
    print("✓ Test 2 passed")
    
    # Test Case 3: Triangle with different Z coordinates
    print("\n[Test 3] Triangle with varying Z (3D geometry)")
    triangle = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 1.0],
        [1.0, 2.0, 0.5]
    ])
    result3 = Geometry_Option.reorder_vertices(triangle, is_upward=False)
    print(f"Input:\n{triangle}")
    print(f"Output:\n{result3}")
    
    signed_area3 = calculate_signed_area(result3[:, :2])
    print(f"Signed area: {signed_area3:.4f} (positive = CCW)")
    assert signed_area3 > 0, "Result should be counter-clockwise"
    print("✓ Test 3 passed")
    
    # Test Case 4: Pentagon clockwise
    print("\n[Test 4] Clockwise pentagon")
    pentagon = np.array([
        [1.0, 0.0, 0.0],
        [0.31, 0.95, 0.0],
        [-0.81, 0.59, 0.0],
        [-0.81, -0.59, 0.0],
        [0.31, -0.95, 0.0]
    ])[::-1]  # Reverse to make it clockwise
    result4 = Geometry_Option.reorder_vertices(pentagon, is_upward=True)
    print(f"Input (first 3 vertices):\n{pentagon[:3]}")
    print(f"Output (first 3 vertices):\n{result4[:3]}")
    
    signed_area4 = calculate_signed_area(result4[:, :2])
    print(f"Signed area: {signed_area4:.4f} (positive = CCW)")
    assert signed_area4 > 0, "Result should be counter-clockwise"
    print("✓ Test 4 passed")
    
    # Test Case 5: Verify is_upward parameter no longer affects result
    print("\n[Test 5] Verify is_upward parameter has no effect")
    test_face = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0]
    ])
    result_upward = Geometry_Option.reorder_vertices(test_face, is_upward=True)
    result_downward = Geometry_Option.reorder_vertices(test_face, is_upward=False)
    
    print(f"Result with is_upward=True:\n{result_upward}")
    print(f"Result with is_upward=False:\n{result_downward}")
    
    assert np.allclose(result_upward, result_downward), "Results should be identical regardless of is_upward"
    print("✓ Test 5 passed - is_upward parameter no longer affects ordering")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

def calculate_signed_area(vertices_2d):
    """Calculate signed area of a 2D polygon using shoelace formula."""
    n = len(vertices_2d)
    signed_area = 0.0
    for i in range(n):
        x1, y1 = vertices_2d[i]
        x2, y2 = vertices_2d[(i + 1) % n]
        signed_area += (x1 * y2 - x2 * y1)
    return signed_area / 2.0

if __name__ == "__main__":
    test_reorder_vertices()
