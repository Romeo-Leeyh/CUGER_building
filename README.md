# BuildingConvex

This is a Python package processing the complex building shape/plan.  The method is based on Arkin, Ronald C.'s report (1987). "Path planning for a vision-based autonomous robot"

## convexify.py

Processing ANY planar polygon with multiple holes inside.

- **BasicOptions** Class of basic geometry methods for 3D points and lines.
  
  - **left_on** decide the third points on the left or right side of the line of the first and the second points, which is a methods for calculate the convave or convex vertex
  - 