#!/usr/bin/env python3
"""
Test script for process.py convex_process function.
This tests the integration without requiring moosas dependency.
"""

import sys
import os

sys.path.append('/home/ubuntu/CUGER_building/cuger')

from __transform import process as ps

def test_convex_process():
    """Test convex_process function with real data."""
    print("=" * 60)
    print("Testing convex_process Function")
    print("=" * 60)
    
    input_geo = "/home/ubuntu/CUGER_building/tests/examples/example0.geo"
    output_geo = "/tmp/test_output.geo"
    figure_path = "/tmp/test_convex.png"
    
    if not os.path.exists(input_geo):
        print(f"⚠ Input file not found: {input_geo}")
        return False
    
    try:
        print(f"\nProcessing: {input_geo}")
        ps.convex_process(input_geo, output_geo, figure_path)
        
        print(f"✓ Output saved to: {output_geo}")
        
        if os.path.exists(output_geo):
            # Check output file size
            size = os.path.getsize(output_geo)
            print(f"✓ Output file size: {size} bytes")
            
            if size > 0:
                print("✓ convex_process test PASSED!")
                return True
            else:
                print("❌ Output file is empty")
                return False
        else:
            print("❌ Output file not created")
            return False
            
    except Exception as e:
        print(f"❌ Error during convex_process: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PROCESS MODULE TEST")
    print("=" * 60 + "\n")
    
    success = test_convex_process()
    
    if success:
        print("\n" + "=" * 60)
        print("TEST PASSED! ✓")
        print("=" * 60 + "\n")
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("TEST FAILED! ❌")
        print("=" * 60 + "\n")
        sys.exit(1)
