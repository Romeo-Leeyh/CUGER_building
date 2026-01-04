import os, sys, time
import __transform.process as ps

main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if main_dir not in sys.path:
    sys.path.append(main_dir)

import moosas.MoosasPy as Moosas

#main
user_profile = os.environ['USERPROFILE']

_fig_show = False
input = r"E:/DATA/CUGER_energy_data/geo"
output = r"E:/DATA/CUGER_energy_data/"

def process_file(input_geo_path, modelname):
    paths = ps.get_output_paths(modelname, output)
    if os.path.exists(paths["new_idf_path"]):
        print(f"--Skip-- | {modelname}")
        #return
    print(f"Processing file: {input_geo_path}, basename: {modelname}")
    
    try:
        #ps.convex_process(input_geo_path, paths["convex_geo_path"], paths["figure_convex_path"])
        #model = Moosas.transform(paths["convex_geo_path"], 
        #               solve_overlap=True, 
        #               divided_zones=False, 
        #               break_wall_horizontal=True, 
        #               solve_redundant=True,
        #               attach_shading=False,
        #               standardize=True)

        #Moosas.saveModel(model, paths["new_geo_path"], save_type="geo")
        #Moosas.saveModel(model, paths["new_xml_path"], save_type="xml")


        ps.graph_process(paths["new_geo_path"], paths["new_xml_path"], paths["output_json_path"], paths["figure_graph_path"])

        #Moosas.saveModel(model, paths["new_rdf_path"], save_type="rdf")
        #Moosas.saveModel(model, paths["new_idf_path"], save_type="idf")
        
    except ValueError as e:
        print(f"ValueError: {e} - Modelname: {modelname}")
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e} - Modelname: {modelname}")
    except Exception as e:
        print(f"Unexpected error: {e} - Modelname: {modelname}")
  

for dirpath, dirnames, filenames in os.walk(input):
    for filename in filenames:
        if filename.endswith('.geo'):
            
            input_geo_path = os.path.join(dirpath, filename).replace('\\', '/')
            relative_path = os.path.relpath(input_geo_path, input)
            basename = os.path.splitext(relative_path)[0].replace('\\', '_')
            process_file(input_geo_path, basename)

