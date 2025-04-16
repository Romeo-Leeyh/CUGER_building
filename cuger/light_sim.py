





import pygeos, os
import numpy as np
import sys, re, time, json
from datetime import datetime

main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if main_dir not in sys.path:
    sys.path.append(main_dir)

from moosas.python.Lib.MoosasPy import transform
from moosas.python.Lib.MoosasPy.IO import modelFromFile
from moosas.python.Lib.MoosasPy.daylighting import simModel
user_profile = os.environ['USERPROFILE']
input = "E:/DATA/Moosasbuildingdatasets_02/_cleaned"
output = "E:/DATA/Moosasbuildingdatasets_02/simulation"
# input = rf"{user_profile}/AppData/Roaming/SketchUp/SketchUp 2022/SketchUp/Plugins/pkpm_moosas/data/geometry/"
if not os.path.exists(output):
    os.makedirs(output)

for dirpath, dirnames, filenames in os.walk(input):
    for filename in filenames:
        if filename.endswith('.geo'):
            
            input_geo_path = os.path.join(dirpath, filename).replace('\\', '/')
            relative_path = os.path.relpath(input_geo_path, input)
            basename = os.path.splitext(relative_path)[0].replace('\\', '_')
            output_json_path = os.path.join(output, f"{basename}.json").replace('\\', '/')

            try:
                model = transform(input_geo_path, solve_contains=False, divided_zones=False, break_wall_horizontal=True, solve_redundant=True,
                                  attach_shading=False, standardize=True)

                t = time.time()
                floorDict = simModel(model, datetime(2022, 1, 1, 15), skyType="-c")

                # 提取需要的字段
                filtered_data = [
                    {"Uid": item["Uid"], "df": item["df"], "satisfied": item["satisfied"]}
                    for item in floorDict
                ]

                # 保存结果为 JSON 文件
                with open(output_json_path, 'w', encoding='utf-8') as json_file:
                    json.dump(filtered_data, json_file, ensure_ascii=False, indent=4)

                print(f"Processed {filename}, results saved to {output_json_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

            break

print("Time:", time.time() - t)