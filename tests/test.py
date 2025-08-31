import os
import sys
import __analyse.geo_anal as ga

input = "E:/DATA/Daylighting_test/model/evomass/geo"

for dirpath, dirnames, filenames in os.walk(input):
    for filename in filenames:
        if filename.endswith('.geo'):
            
            input_geo_path = os.path.join(dirpath, filename).replace('\\', '/')
            relative_path = os.path.relpath(input_geo_path, input)
            basename = os.path.splitext(relative_path)[0].replace('\\', '_')
            process_file(input_geo_path, basename)