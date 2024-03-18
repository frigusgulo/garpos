import sys
import os 


module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "garpos_bin/"))
sys.path.append(module_dir)

print(module_dir)

# bin_path = "../bin/garpos_v101"

import garpos_bin as garpos 
from garpos_bin.garpos_v101.garpos_main import drive_garpos

import os
import sys

# from garpos.garpos_main import drive_garpos


obs_data = "/Users/franklyndunbar/Project/SeaFloorGeodesy/garpos/sample/obsdata/SAGA/SAGA.1903.kaiyo_k4-obs.csv"

svp_data = "/Users/franklyndunbar/Project/SeaFloorGeodesy/garpos/sample/obsdata/SAGA/SAGA.1903.kaiyo_k4-svp.csv"

site_config = "/Users/franklyndunbar/Project/SeaFloorGeodesy/garpos/sample/initcfg/SAGA/SAGA.1903.kaiyo_k4-initcfg.ini"

hyper_config = (
    "/Users/franklyndunbar/Project/SeaFloorGeodesy/garpos/sample/Settings-fix.ini"
)

suf = "test"

outdir = "/Users/franklyndunbar/Project/SeaFloorGeodesy/garpos/sample/output"

mc = 1  # max core

drive_garpos(site_config, hyper_config, outdir, suf, mc)
