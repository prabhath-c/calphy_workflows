import os
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

from ase.atoms import Atoms
from calphy import Calculation, Solid, Liquid
from calphy.routines import routine_fe, routine_ts
from calphy.postprocessing import gather_results
import pandas as pd

from .helpers import (
    _working_directory_context,
    _save_calphy_input_yaml,
    _build_calphy_config,
    _validate_input_structure,
    _validate_potential_df,
    _validate_calphy_parameters,
)

def _run_calphy(input_class: Calculation, lmp: Optional[Any] = None) -> None:
    """Execute calphy calculation based on the input configuration.

    Parameters
    ----------
    input_class : Calculation
        Calphy Calculation object with all parameters configured
    lmp : Optional[Any], optional
        Optional LAMMPS library object from pylammpsmpi with embedded executor.
        If provided, the calculation will use this lmp object instead of creating
        its own, enabling executor-based parallel execution.

    Raises
    ------
    ValueError
        If reference_phase is not 'solid' or 'liquid'
        If mode is not 'fe' (free energy) or 'ts' (temperature scaling)
    RuntimeError
        If calphy execution fails
    """
    # Use the working directory from the lattice path (this is where files are written)
    lattice_path = Path(input_class.lattice)
    working_directory = str(lattice_path.parent)

    with _working_directory_context(working_directory):
        try:
            if input_class.reference_phase == "solid":
                if lmp is not None:
                    job = Solid(calculation=input_class, simfolder=working_directory, lmp=lmp)
                else:
                    job = Solid(calculation=input_class, simfolder=working_directory)
            elif input_class.reference_phase == "liquid":
                if lmp is not None:
                    job = Liquid(calculation=input_class, simfolder=working_directory, lmp=lmp)
                else:
                    job = Liquid(calculation=input_class, simfolder=working_directory)
            else:
                raise ValueError(
                    f"Invalid reference_phase: {input_class.reference_phase}. "
                    "Must be 'solid' or 'liquid'"
                )

            if input_class.mode == "fe":
                routine_fe(job)
            elif input_class.mode == "ts":
                routine_ts(job)
            else:
                raise ValueError(
                    f"Invalid mode: {input_class.mode}. Must be 'fe' or 'ts'"
                )
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Calphy execution failed with {type(e).__name__}: {str(e)}"
            ) from e

def gather_calphy_results(parent_directory: str) -> pd.DataFrame:
    """Gather and return results from calphy calculations.

    Parameters
    ----------
    parent_directory : str
        Path to parent directory containing calphy calculation folders

    Returns
    -------
    pd.DataFrame
        DataFrame containing aggregated results from all calculations
    """
    with _working_directory_context(parent_directory):
        df = gather_results('.')
    return df

def calc_free_energy_with_calphy(
    input_structure: Atoms,
    potential_df: pd.DataFrame,
    calphy_parameters: Dict[str, Any],
    working_directory: Optional[str],
    lmp: Optional[Any] = None,
    metadata_dict: Optional[Dict[str, Any]] = None
) -> Tuple[Calculation, pd.DataFrame]:
    """Main function to calculate free energy using calphy with LAMMPS potentials.

    Orchestrates the entire workflow: configures calphy parameters, writes structure
    files in LAMMPS format, executes calphy calculations, and gathers results.

    Parameters
    ----------
    input_structure : Atoms
        ASE Atoms object representing the crystal structure
    potential_df : pd.DataFrame
        DataFrame containing potential information from pyiron_lammps
    calphy_parameters : Dict[str, Any]
        Dictionary with calphy parameters including:
        - mode: 'fe' (free energy) or 'ts' (temperature scaling)
        - temperature: float or list for temperature range
        - reference_phase: 'solid' or 'liquid'
        - n_equilibration_steps, n_switching_steps, n_print_steps
        - equilibration_control: thermostat type (e.g., 'nose-hoover')
        - queue: dict with cores and scheduler info
        - file_format: 'lammps-data'
    working_directory : str
        Directory where calculations will be run
    lmp : Optional[Any], optional
        Optional LAMMPS library object from pylammpsmpi with embedded executor.
        If provided, the calculation will use this lmp object instead of creating
        its own, enabling executor-based parallel execution.
    metadata_dict : Optional[Dict[str, Any]], optional
        Optional dictionary for storing user-defined metadata in executorlib's cache.
        Used when lmp is provided to enable result caching and retrieval.

    Returns
    -------
    Tuple[Calculation, pd.DataFrame]
        Tuple containing:
        - Calculation object: The calphy Calculation instance used
        - pd.DataFrame: Results DataFrame from gather_calphy_results()

    Examples
    --------
    # Basic usage without executor
    result = calc_free_energy_with_calphy(
        input_structure=structure,
        potential_df=potential_df,
        calphy_parameters=params,
        working_directory='output_dir'
    )

    # With executor
    executor = SingleNodeExecutor()
    lmp = LammpsLibrary(cores=1, executor=executor)
    result = calc_free_energy_with_calphy(
        input_structure=structure,
        potential_df=potential_df,
        calphy_parameters=params,
        working_directory='output_dir',
        lmp=lmp,
        metadata_dict={'project': 'my_project', 'version': '1.0'}
    )

    Raises
    ------
    ValueError
        If required parameters are missing or invalid. Type and value issues
        raised during validation are normalized to ValueError.
    RuntimeError
        If calculation execution fails
    """
    try:
        # Validate all inputs
        _validate_input_structure(input_structure)
        _validate_potential_df(potential_df)
        _validate_calphy_parameters(calphy_parameters)
        print("Input validation successful. Proceeding with calculation.")
    except (TypeError, ValueError) as e:
        raise ValueError(f"Input validation failed: {str(e)}") from e

    try:
        if working_directory is None:
            working_directory = os.getcwd()
            print(f"No working directory provided. Using current directory {working_directory} as working directory.")
        if not os.path.exists(working_directory):
            os.makedirs(working_directory)

        with _working_directory_context(working_directory):
            input_class = _build_calphy_config(
                input_structure=input_structure,
                potential_df=potential_df,
                calphy_parameters=calphy_parameters,
                working_directory=working_directory
            )

            # _save_calphy_input_yaml(
            #     input_class=input_class,
            #     folder_name=working_directory
            # )

            _run_calphy(input_class=input_class, lmp=lmp)

        abs_working_dir = os.path.abspath(working_directory)
        parent_dir = os.path.dirname(abs_working_dir)
        df = gather_calphy_results(parent_dir)

        return input_class, df

    except (ValueError, RuntimeError):
        raise
    except Exception as e:
        raise RuntimeError(
            f"Free energy calculation workflow failed: {type(e).__name__}: {str(e)}"
        ) from e