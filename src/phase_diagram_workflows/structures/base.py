from dataclasses import dataclass

from ase import Atoms

@dataclass
class StructureContainer:
    """
    A container for structures, which can be used to store multiple structures in a single object.
    """
    pristine_structure: Atoms