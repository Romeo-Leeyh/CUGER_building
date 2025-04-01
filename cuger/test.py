import pygeos, os
import numpy as np
import sys,re,time
from datetime import datetime

main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if main_dir not in sys.path:
    sys.path.append(main_dir)

from moosas.python.Lib.MoosasPy import transform
from moosas.python.Lib.MoosasPy.IO import modelFromFile
from moosas.python.Lib.MoosasPy.daylighting import simModel



f = r"C:/Users/LI YIHUI/AppData/Roaming/SketchUp/SketchUp 2022/SketchUp/Plugins/pkpm_moosas/data/geometry/selection0.geo"
#f = rf"BuildingConvex\data\geo\selection0.geo"

#f = rf"moosas\test\test2.geo"
#f = rf"test\test3_geomove.geo"

model_file = rf"BuildingConvex\data\new_xml\selection0.xml"

model=transform(f, model_file, solve_contains=False, divided_zones=False, break_wall_horizontal=True, solve_redundant=True,
          attach_shading=False,standardize=True)

model=modelFromFile(model_file, inputType="xml")

#model=modelFromFile(model_file)

t=time.time()
floorDict = simModel(model,datetime(2022,1,1,15),skyType="-c")
for i in floorDict:
    print("id:",i["Uid"], "df:", i["df"], "sda:", i["satisfied"])
print("Time:",time.time()-t)
