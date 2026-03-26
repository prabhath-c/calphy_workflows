import os
from typing import Dict, Any, Tuple

from ase.atoms import Atoms
from calphy import Calculation, Solid, Liquid
from calphy.routines import routine_fe, routine_ts
from calphy.postprocessing import gather_results
import pandas as pd


from helpers import _working_directory_context, _save_calphy_input_yaml, _run_calphy, gather_calphy_results, _build_calphy_config

def _run_calphy(input_class: Calculation) -> None:
    curr_wd = os.getcwd()
    with _working_directory_context(curr_wd):
        if input_class.reference_phase == "solid":
            job = Solid(calculation=input_class, simfolder=curr_wd)
        elif input_class.reference_phase == "liquid":
            job = Liquid(calculation=input_class, simfolder=curr_wd)
        else:
            raise ValueError("Unknown reference state")

        if input_class.mode == "fe":
            routine_fe(job)
        elif input_class.mode == "ts":
            routine_ts(job)
        else:
            raise ValueError("Unknown mode")

def gather_calphy_results(parent_directory: str) -> pd.DataFrame:
    with _working_directory_context(parent_directory):
        df = gather_results('.')
    return df

def calc_free_energy_with_calphy(
    input_structure: Atoms,
    potential_df: pd.DataFrame,
    calphy_parameters: Dict[str, Any],
    working_directory: str,
    user_dict: Dict[str, Any],
) -> Tuple[Calculation, pd.DataFrame]: 
    if not os.path.exists(working_directory):
        os.makedirs(working_directory)
    elif working_directory is None:
        working_directory = os.getcwd()
        print(f"No working directory provided. Using current directory {working_directory} as working directory.")

    with _working_directory_context(working_directory):
        input_class = _build_calphy_config(
            input_structure=input_structure,
            potential_df=potential_df,
            calphy_parameters=calphy_parameters
        )

        # _save_calphy_input_yaml(
        #     input_class=input_class, 
        #     folder_name=working_directory
        # )

        _run_calphy(input_class=input_class)

    abs_working_dir = os.path.abspath(working_directory)
    parent_dir = os.path.dirname(abs_working_dir)
    df = gather_calphy_results(parent_dir)

    return input_class, df