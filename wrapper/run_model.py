import os
import sys
sys.path.append("/Users/franklyndunbar/Project/SeaFloorGeodesy/garpos/")

from typing import Tuple
from garpos_bin.garpos_v101.garpos_main import drive_garpos
from schemas import Observation,ObservationData,ModelResults,Site,ATDOffset,GarposFixed,GarposInput,InversionResults


def main(input:GarposInput,fixed:GarposFixed) -> Tuple[GarposInput,GarposFixed,InversionResults]:
    # Convert the observation and site data to the desired format
    tmp_dir = "/tmp/garpos"
    os.makedirs(tmp_dir, exist_ok=True)
    results_dir = os.path.join(tmp_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    input_path = os.path.join(tmp_dir, "observation.ini")
    fixed_path = os.path.join(tmp_dir, "settings.ini")

    input.to_dat_file(tmp_dir,input_path)
    fixed.to_dat_file(tmp_dir,fixed_path)

    rf = drive_garpos(input_path, fixed_path, results_dir, "test", 5)

    input_data = GarposInput.from_dat_file(rf)
    fixed_data = GarposFixed.from_dat_file(fixed_path)
    results = InversionResults.from_dat_file(rf)
    
    os.remove(results_dir)
    return input_data,fixed_data,results


if __name__ == "__main__":
    obs_file = "/Users/franklyndunbar/Project/SeaFloorGeodesy/garpos/sample/initcfg/SAGA/SAGA.1903.kaiyo_k4-initcfg.ini"
    settings_file = "/Users/franklyndunbar/Project/SeaFloorGeodesy/garpos/sample/Settings-fix.ini"
    input = GarposInput.from_dat_file(obs_file)
    fixed = GarposFixed.from_dat_file(settings_file)
    main(input,fixed)
