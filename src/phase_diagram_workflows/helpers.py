import os
from typing import Dict, Any, Tuple, Optional

from ase.data import chemical_symbols, atomic_masses
from ase.atoms import Atoms
from calphy import Calculation
from contextlib import contextmanager
import pandas as pd
from pydantic import ValidationError
from pyiron_lammps.structure import structure_to_lammps, LammpsStructure
from ruamel.yaml import YAML

@contextmanager
def _working_directory_context(path: str):
    """Context manager to temporarily change working directory.
    
    Changes to the specified directory, creates it if it doesn't exist,
    and ensures the original directory is restored on exit (even if errors occur).
    
    Parameters
    ----------
    path : str
        Target directory path
        
    Yields
    ------
    None
    
    Examples
    --------
    >>> with _working_directory_context('/tmp/workdir'):
    ...     # Code here runs in /tmp/workdir
    ...     os.getcwd()  # returns /tmp/workdir
    >>> os.getcwd()  # back to original directory
    """
    prev_cwd = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)

def _save_calphy_input_yaml(
    input_class: Calculation,
    folder_name: str,
    file_name: str = "my_input_file.yaml"
) -> None:
    """Save calphy Calculation object to YAML input file.
    
    Exports the calphy Calculation configuration to a YAML file that can be
    used directly with calphy command-line tools.
    
    Parameters
    ----------
    input_class : Calculation
        Calphy Calculation object to serialize
    folder_name : str
        Directory where YAML file will be written
    file_name : str, optional
        Name of the output YAML file (default: 'my_input_file.yaml')
        
    Returns
    -------
    None
    """
    yaml = YAML()
    yaml.indent(mapping=2, sequence=2)

    input_data = {"calculations": [input_class.model_dump()]}
    output_path = os.path.join(folder_name, file_name)
    with open(output_path, "w") as fout:
        yaml.dump(input_data, fout)

def _write_structure(
    structure: Atoms,
    potential_df: pd.DataFrame,
    file_name: str,
    working_directory: str
) -> None:
    """Write ASE structure to LAMMPS data file format.
    
    Uses pyiron_lammps to convert ASE Atoms object to LAMMPS format and
    writes it to the specified working directory. Validates that all elements
    in the structure are supported by the selected potential.
    
    Parameters
    ----------
    structure : Atoms
        ASE Atoms object to write
    potential_df : pd.DataFrame
        Potential DataFrame from pyiron_lammps
    file_name : str
        Output file name (will be written to working_directory)
    working_directory : str
        Output working directory
        
    Returns
    -------
    None
    
    Raises
    ------
    ValueError
        If structure contains elements not supported by the potential
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
    """Extract and ensure pair_style and pair_coeff parameters from potential.
    
    If pair_style or pair_coeff are missing from calphy_parameters, extracts them
    from the potential DataFrame (pyiron_lammps format).
    
    Parameters
    ----------
    calphy_parameters : Dict[str, Any]
        Calphy parameters dictionary to update
    potential_df : pd.DataFrame
        Potential DataFrame from pyiron_lammps with 'Config' column
        
    Returns
    -------
    Dict[str, Any]
        Updated calphy_parameters with pair_style and pair_coeff set
    """
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
    """Ensure 'element' and 'mass' keys exist in calphy parameters.
    
    If missing, computes them from the potential dataframe and input structure.
    
    Parameters
    ----------
    input_structure : Atoms
        ASE Atoms object with the structure
    potential_df : pd.DataFrame
        Potential DataFrame from pyiron_lammps
    calphy_parameters : Dict[str, Any]
        Calphy parameters dictionary to update
        
    Returns
    -------
    Dict[str, Any]
        Updated calphy_parameters with element and mass keys set
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
    """Create and validate a calphy Calculation object from input parameters.
    
    Parameters
    ----------
    input_parameters : Dict[str, Any]
        Dictionary of parameters for calphy Calculation
        
    Returns
    -------
    Calculation
        Validated calphy Calculation object
        
    Raises
    ------
    ValueError
        If parameters fail validation against calphy's Calculation model
    """
    try:
        return Calculation.model_validate(input_parameters)
    except ValidationError as e:
        raise ValueError(f"Invalid parameters: {e}") from e

def _build_calphy_config(
    input_structure: Atoms,
    potential_df: pd.DataFrame,
    calphy_parameters: Dict[str, Any],
    working_directory: Optional[str]
) -> Calculation:
    """Build complete calphy configuration from structure, potential, and parameters.
    
    Orchestrates the configuration building by:
    1. Writing the input structure to LAMMPS format (if not provided)
    2. Extracting potential parameters (pair_style, pair_coeff)
    3. Setting element types and atomic masses
    4. Creating and returning a validated Calculation object
    
    Parameters
    ----------
    input_structure : Atoms
        ASE Atoms object with the structure
    potential_df : pd.DataFrame
        Potential DataFrame from pyiron_lammps
    calphy_parameters : Dict[str, Any]
        Parameters for calphy calculation
    working_directory : Optional[str]
        Directory for writing files. If None, uses current directory
        
    Returns
    -------
    Calculation
        Configured and validated calphy Calculation object
    """
    if not os.path.exists(working_directory):
        os.makedirs(working_directory)
    elif working_directory is None:
        working_directory = os.getcwd()
    
    if 'lattice' not in calphy_parameters:
        _write_structure(
            structure=input_structure, 
            potential_df=potential_df, 
            file_name='input_structure.data', 
            working_directory=working_directory
        )
        
        lattice_file = f'{working_directory}/input_structure.data'
        calphy_parameters["lattice"] = lattice_file

    ## FIXME: Check calphy pyiron job for handling elements, masses etc
    input_parameters = _ensure_potential(calphy_parameters, potential_df)
    input_parameters = _ensure_elements_and_masses(input_structure, potential_df, input_parameters)
    
    input_class = _create_input_class(input_parameters=input_parameters)

    return input_class