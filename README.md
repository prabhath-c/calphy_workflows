# Phase Diagram Workflows

🔬 **Advanced workflows for phase diagram calculations using Calphy and LAMMPS**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Calphy](https://img.shields.io/badge/built_with-calphy-orange)](https://github.com/ICAMS/calphy)

## 📋 **Overview**

**Phase Diagram Workflows** is a powerful Python package for computational materials science that provides robust workflows for calculating free energies and phase diagrams using [Calphy](https://github.com/phasesresearchlab/calphy) and LAMMPS.

The package offers both traditional serial execution and advanced parallel execution via executorlib, making it suitable for both small-scale testing and large-scale HPC calculations.

## 🚀 **Features**

### **Core Functionality**
- ✅ **Free Energy Calculations**: Solid, liquid, and interface free energy calculations
- 🚀 **Executor Support**: Parallel execution via executorlib for HPC environments

## 📦 **Installation**

### **From Source**
```bash
git clone https://github.com/prabhath-c/phase_diagram_workflows.git
cd phase_diagram_workflows
pip install -e .
```

## 🛠️ **Dependencies**

### **Core Dependencies**
- `ase` - Atomic Simulation Environment
- `calphy` - Calphy free energy calculation library
- `pandas` - Data analysis and manipulation
- `pyiron-lammps` - PyIron LAMMPS integration
- `ruamel.yaml` - YAML configuration support
- `pydantic` - Data validation and settings management

### **Optional Dependencies**
- `executorlib` - Parallel execution framework (executor feature)
- `pylammpsmpi` - MPI-enabled LAMMPS library (executor feature)

## 💻 **Usage**

### **Basic Usage**
```python
from phase_diagram_workflows import calculator

# Create a solid free energy calculation
solid_calc = calculator.Solid(
    structure=ase_atoms_object,
    temperature=300,
    pressure=0.0,
    potential="Al_eam.fs"
)

# Run the calculation
solid_calc.run()

# Get results
free_energy = solid_calc.get_free_energy()
```

### **Executor-Based Parallel Execution**
```python
from phase_diagram_workflows import calculator
from executorlib import SingleNodeExecutor
from pylammpsmpi import LammpsLibrary

# Initialize executor
executor = SingleNodeExecutor()
lmp_lib = LammpsLibrary(executor=executor)

# Run calculation with executor
result = calculator.run_calphy_with_executor(
    input_class=solid_calc,
    lmp=lmp_lib
)
```

## 📂 **Project Structure**
```
phase_diagram_workflows/
├── src/
│   └── phase_diagram_workflows/
│       ├── calculator.py      # Main calculation functions
│       ├── helpers.py         # Utility functions
│       └── __init__.py
├── notebooks/                # Example notebooks
├── tests/                    # Unit tests
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

## 🔬 **Examples**

Check out the example notebooks in the `notebooks/` directory:

- **[Al_free_energy_executor_demo.ipynb](notebooks/Al_free_energy_executor_demo.ipynb)** - Demonstrates executor integration

## 🤝 **Contributing**

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'feat: your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a pull request

## 📜 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 **Support**

For issues, questions, or feature requests:
- Open an issue on GitHub
- Contact: prabhath.chilakalapudi@mpi-susmat.de

---

**© 2026 Phase Diagram Workflows | MPI for Sustainable Materials**