import os, sys, time
import multiprocessing as mp
from __transform.convexify import MoosasConvexify
from __transform.graph import MoosasGraph
import __transform.process as ps

main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if main_dir not in sys.path:
    sys.path.append(main_dir)

import moosas.MoosasPy as Moosas


#main
user_profile = os.environ['USERPROFILE']


_fig_show = True

input = r"//166.111.40.8/protect/moosasTestModelDataset/_cleaned_SRT"
output = r"E:/DATA/SRT_results_0"

def run_transform(paths):
    import moosas.MoosasPy as Moosas
    Moosas.transform(
        paths["output_geo_path"],
        paths["new_idf_path"],
        paths["new_geo_path"],
        solve_contains=False,
        divided_zones=False,
        break_wall_horizontal=True,
        solve_redundant=True,
        attach_shading=False,
        standardize=True
    )

def process_file(input_geo_path, modelname):
    paths = ps.get_output_paths(modelname, output)
    if os.path.exists(paths["new_idf_path"]):
        print(f"--Skip-- | {modelname}")
        return
    print(f"Processing file: {input_geo_path}, basename: {modelname}")

    try:
        p = mp.Process(target=run_transform, args=(paths,))
        p.start()
        p.join(timeout=300)  # 最长 300 秒
        if p.is_alive():
            print(f"超时跳过: {modelname}")
            p.terminate()
            p.join()
    except Exception as e:
        print(f"Unexpected error: {e} - Modelname: {modelname}")


    
    """
    ps.convex_process(input_geo_path, paths["output_geo_path"], paths["figure_convex_path"])
    Moosas.transform(paths["output_geo_path"],
                    solve_contains=False, 
                    divided_zones=True, 
                    break_wall_horizontal=True, 
                    solve_redundant=True,
                    attach_shading=False,
                    standardize=True)

    ps.graph_process(paths["new_geo_path"], paths["new_xml_path"], paths["output_json_path"], paths["figure_graph_path"])
    """
  

for dirpath, dirnames, filenames in os.walk(input):
    for filename in filenames:
        if filename.endswith('.geo'):
            
            input_geo_path = os.path.join(dirpath, filename).replace('\\', '/')
            relative_path = os.path.relpath(input_geo_path, input)
            basename = os.path.splitext(relative_path)[0].replace('\\', '_')
            process_file(input_geo_path, basename)

