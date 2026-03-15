import os
from typing import Dict, Any

from ase.data import chemical_symbols, atomic_masses
from ase.atoms import Atoms
from calphy import Calculation, Solid, Liquid 
from calphy.routines import routine_fe, routine_ts
from calphy.postprocessing import gather_results
from contextlib import contextmanager
import pandas as pd
from pydantic import ValidationError
from pyiron_lammps.structure import structure_to_lammps, LammpsStructure
from ruamel.yaml import YAML

@contextmanager
def _working_directory_context(path: str):
    prev_cwd = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)

def _save_calphy_input_yaml(input_class: Calculation, folder_name: str):
    yaml = YAML()
    yaml.indent(mapping=2, sequence=2)

    input_data = {"calculations": [input_class.model_dump()]}
    with open(f"{folder_name}/my_input_file.yaml", "w") as fout:
        yaml.dump(input_data, fout)

def _write_structure(
    structure: Atoms, 
    potential_df: pd.DataFrame, 
    file_name: str, 
    working_directory: str
) -> None:
    """
    Write structure to file

    Args:
        structure: input structure
        file_name (str): output file name
        working_directory (str): output working directory

    Returns:
        None
    """
    lmp_structure = LammpsStructure()
    lmp_structure.potential = potential_df
    lmp_structure.atom_type = "atomic"

    # lmp_structure.el_eam_lst = list(lmp_structure.potential ["Species"][0])
    lmp_structure.el_eam_lst = lmp_structure.potential['Species'].to_list()[0]
    lmp_structure.structure = structure_to_lammps(structure)

    if not set(lmp_structure.structure.get_chemical_symbols()).issubset(
        set(lmp_structure.el_eam_lst)
    ):
        raise ValueError(
            "The selected potentials do not support the given combination of elements."
        )
    lmp_structure.write_file(file_name=file_name, cwd=working_directory)

def _ensure_potential(calphy_parameters: Dict[str, Any], potential_df: pd.DataFrame) -> Dict[str, Any]:

    if "pair_style" not in calphy_parameters or "pair_coeff" not in calphy_parameters:

        [pair_style, pair_coeff] = [
            line.replace("pair_style", "")
                .replace("pair_coeff", "")
                .strip() 
            for line in potential_df['Config'].tolist()[0]
            ]
        
        # update dict if missing
        if "pair_style" not in calphy_parameters:
            calphy_parameters["pair_style"] = pair_style
        if "pair_coeff" not in calphy_parameters:
            calphy_parameters["pair_coeff"] = pair_coeff
    
    return calphy_parameters

def _ensure_elements_and_masses(
    input_structure: Atoms, 
    potential_df: pd.DataFrame, 
    calphy_parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Ensure 'element' and 'mass' keys exist in calphy_parameters.
    If missing, compute them from pair_coeff and input_structure.
    """

    if "element" not in calphy_parameters or "mass" not in calphy_parameters:

        structure_symbols = list(set(input_structure.get_chemical_symbols()))

        element_symbols = potential_df["Species"].tolist()[0]
        # masses = list([atomic_masses[chemical_symbols.index(el)] for el in element_symbols])    
        
        masses = [
            atomic_masses[chemical_symbols.index(el)] if el in structure_symbols else 1.0
            for el in element_symbols
        ]

        if "element" not in calphy_parameters:
            calphy_parameters["element"] = element_symbols
        if "mass" not in calphy_parameters:
            calphy_parameters["mass"] = masses

    return calphy_parameters

def _create_input_class(input_parameters: Dict[str, Any]) -> Calculation:
    try:
        return Calculation.model_validate(input_parameters)
    except ValidationError as e:
        raise ValueError(f"Invalid parameters: {e}") from e

def _build_calphy_config(
    input_structure: Atoms,
    potential_df: pd.DataFrame,
    calphy_parameters: Dict[str, Any],
) -> Calculation:
    
    curr_wd = os.getcwd()
    if 'lattice' not in calphy_parameters:
        _write_structure(
            structure=input_structure, 
            potential_df=potential_df, 
            file_name='input_structure.data', 
            working_directory=curr_wd
        )
        
        lattice_file = f'{curr_wd}/input_structure.data'
        calphy_parameters["lattice"] = lattice_file

    ## FIXME: Check calphy pyiron job for handling elements, masses etc
    input_parameters = _ensure_potential(calphy_parameters, potential_df)
    input_parameters = _ensure_elements_and_masses(input_structure, potential_df, input_parameters)
    
    input_class = _create_input_class(input_parameters=input_parameters)

    return input_class

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
) -> tuple[Calculation, pd.DataFrame]: 
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