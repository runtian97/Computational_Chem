import os
import glob
import torch
import numpy as np
from ase.io import read
from ase.calculators.orca import ORCA, OrcaProfile
from openbabel import pybel
from ase.units import Bohr


# ─────────────────────────────────────────────────────────────────────────────
#                         === ORCA Output Parsers ===
# ─────────────────────────────────────────────────────────────────────────────

def parse_3rd_rad_mnoment(output_file):
    """
    Parse MBIS third radial moments from ORCA output and convert from a.u.^3 to Å^3.
    """
    conversion_factor = 0.1481847  # (0.529177)^3
    charges = []
    with open(output_file, 'r') as f:
        lines = f.readlines()
        in_section = False
        for line in lines:
            if 'MBIS THIRD RADIAL MOMENT' in line:
                in_section = True
                continue
            if in_section:
                if 'Total SCF time' in line:
                    break
                parts = line.split()
                if len(parts) == 3:
                    try:
                        charges.append(float(parts[2]) * conversion_factor)
                    except ValueError:
                        continue
    return np.array(charges)


def parse_MBIS_charges(output_file):
    """
    Parse MBIS charges from ORCA output.
    """
    charges = []
    with open(output_file, 'r') as f:
        lines = f.readlines()
        in_section = False
        for line in lines:
            if 'MBIS ANALYSIS' in line:
                in_section = True
                continue
            if in_section:
                if 'TOTAL' in line:
                    break
                parts = line.split()
                if len(parts) == 5:
                    try:
                        charges.append(float(parts[2]))
                    except ValueError:
                        continue
    return np.array(charges)


def parse_hirshfeld_charges(output_file):
    """
    Parse Hirshfeld charges from ORCA output.
    """
    charges = []
    with open(output_file, 'r') as f:
        lines = f.readlines()
        in_section = False
        for line in lines:
            if 'HIRSHFELD ANALYSIS' in line:
                in_section = True
                continue
            if in_section:
                if 'TOTAL' in line:
                    break
                parts = line.split()
                if len(parts) == 4:
                    try:
                        charges.append(float(parts[2]))
                    except ValueError:
                        continue
    return np.array(charges)


def parse_dipole_moment(output_file):
    """
    Parse total dipole moment (in a.u.) from ORCA output and convert to Å·e.
    """
    conversion_factor = 0.529177
    with open(output_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Total Dipole Moment' in line:
                parts = line.split()
                return np.array([float(parts[4]), float(parts[5]), float(parts[6])]) * conversion_factor
    return np.zeros(3)


def parse_and_normalize_quadrupole(output_file):
    """
    Parse total quadrupole moment (TOT) from ORCA output and return de-traced, normalized version in Å².
    """
    for line in open(output_file):
        if line.startswith('TOT '):
            q = np.array([float(x) for x in line.split()[1:7]])
            m = q[:3].mean()
            q_detraced = np.concatenate([q[:3] - m, q[[3, 5, 4]]]) * Bohr ** 2  # rearrange: XY, YZ, XZ
            return q_detraced
    return np.zeros(6)


def parse_dispersion_and_single_point_E(output_file):
    """
    Parse dispersion and total single point energy (in eV) from ORCA output.
    """
    hartree_to_ev = 27.2113834
    E_disp, E_sp = None, None
    with open(output_file, 'r') as f:
        for line in f:
            if 'Dispersion correction' in line:
                try:
                    E_disp = float(line.split()[-1]) * hartree_to_ev
                except ValueError:
                    continue
            if 'FINAL SINGLE POINT ENERGY' in line:
                try:
                    E_sp = float(line.split()[-1]) * hartree_to_ev
                except ValueError:
                    continue
    return E_disp, E_sp


def parse_forces(output_file):
    """
    Parse Cartesian gradients (forces) from ORCA output, converted to eV/Å.
    """
    conversion = 51.42208619083232
    forces = []
    with open(output_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if 'CARTESIAN GRADIENT' in line:
                for j in range(i + 3, len(lines)):
                    parts = lines[j].split()
                    if len(parts) == 6:
                        forces.append([-float(x) * conversion for x in parts[-3:]])
                    else:
                        break
    return np.array(forces)


def parse_dispersion_grad(output_file):
    """
    Parse dispersion gradient forces from ORCA output, converted to eV/Å.
    """
    conversion = 51.42208619083232
    forces = []
    with open(output_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if 'DISPERSION GRADIENT' in line:
                for j in range(i + 3, len(lines)):
                    parts = lines[j].split()
                    if len(parts) == 6:
                        forces.append([-float(x) * conversion for x in parts[-3:]])
                    else:
                        break
    return np.array(forces)


# ─────────────────────────────────────────────────────────────────────────────
#                       === Main ORCA ASE Driver ===
# ─────────────────────────────────────────────────────────────────────────────

def ORCA_ASE(mol_sdf_path, device):
    """
    Run ORCA single-point calculation using ASE and extract computed properties.

    Parameters:
        mol_sdf_path (str): Path to the input SDF file.
        device (str): 'cpu' or 'cuda', not directly used but passed for compatibility.

    Returns:
        tuple: (dipole, energy, forces, quadrupole, hirshfeld_charges,
                mbis_charges, mbis_radial, energy_with_disp, forces_with_disp)
    """
    # Read molecule metadata
    mol = next(pybel.readfile('sdf', mol_sdf_path))
    charge = mol.charge
    multiplicity = mol.spin

    # Load molecule into ASE
    atoms = read(mol_sdf_path)

    # Define ORCA calculator
    profile = OrcaProfile(command='/Users/nickgao/orca_6_0_0/orca')
    calc = ORCA(
        label='orca',
        profile=profile,
        orcasimpleinput='b97-3c tightscf engrad SCFConvForced slowconv MBIS',
        orcablocks="""
        %PAL NPROCS 1 END
        %elprop
          Polar 1
          dipole true
          quadrupole true
        end
        %method
          MBIS_LARGEPRINT TRUE
        end
        %output
          PrintLevel mini
          Print[P_DFTD_GRAD] 1
          Print[P_Hirshfeld] 1
          Print[P_Mulliken] 1
          Print[P_Mbis] 1
        end
        """,
        charge=charge,
        mult=multiplicity
    )

    # Attach calculator and run calculation
    atoms.calc = calc
    atoms.get_potential_energy()
    out_file = 'orca.out'

    # Parse output
    dipole = parse_dipole_moment(out_file)
    E_disp, E_total = parse_dispersion_and_single_point_E(out_file)
    energy = E_total - E_disp
    energy_with_disp = E_total

    forces = parse_forces(out_file) - parse_dispersion_grad(out_file)
    forces_with_disp = parse_forces(out_file)

    quadrupole = parse_and_normalize_quadrupole(out_file)
    hirshfeld_charges = parse_hirshfeld_charges(out_file)
    mbis_charges = parse_MBIS_charges(out_file)
    mbis_radial = parse_3rd_rad_mnoment(out_file)

    # Clean intermediate ORCA files
    for f in glob.glob('orca*'):
        if not f.endswith('orca.out') and not f.endswith('orca.inp'):
            os.remove(f)

    return (dipole, energy, forces, quadrupole,
            hirshfeld_charges, mbis_charges, mbis_radial,
            energy_with_disp, forces_with_disp)


# ─────────────────────────────────────────────────────────────────────────────
#                              === CLI Test ===
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mol_sdf_path = '/Users/nickgao/Desktop/pythonProject/local_code_template/test_sdfs/CO.sdf'

    results = ORCA_ASE(mol_sdf_path, device=device)
    print("Quadrupole moment (normalized, Å²):", results[3])
