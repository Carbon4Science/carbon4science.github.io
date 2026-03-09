#!/mifs/dgd03153/apps/anaconda3/envs/chgnet040/bin/python
import argparse
import csv
import sys
from copy import deepcopy
import os

from ase.io import read
from ase import Atoms
from ase.data import chemical_symbols

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from joblib import Parallel, delayed

# ====== Argument Parsing ======
parser = argparse.ArgumentParser()
parser.add_argument('-tm', '--temperatures', type=int, nargs="+")
parser.add_argument('-l', '--label', type=str)
parser.add_argument('-fr', '--frequency', type=int, default=1)
parser.add_argument("-tt", "--totals", type=int, nargs="+", default=None, help="Total time in ps")
parser.add_argument("-i", "--ignores", type=int, nargs="+", default=None, help="Time to ignore in ps")
parser.add_argument("-ts", "--time_step", type=float, default=2)
parser.add_argument("-sk", "--step_skip", type=int, default=50)
parser.add_argument('-ei', '--elements_i', nargs="+")
parser.add_argument('-ej', '--elements_j', nargs="+")
parser.add_argument('-m', '--multi_rdf', action="store_true")
parser.add_argument("-r", "--rmax", type=float, default=4)
parser.add_argument("-dr", "--dr", type=float, default=0.05)
parser.add_argument('-p', '--plot', action="store_true")
parser.add_argument('-sh', '--shell', action="store_true")
parser.add_argument('--temp_par', action="store_true")
parser.add_argument('--rdf_par', action="store_true")
args = parser.parse_args()

FACTOR = 1000/args.time_step/args.step_skip/args.frequency

if args.rdf_par:
    print("Run in parallel RDF")
    from asetools.analysis.rdf_par import rdf
else:
    print("Run in serial RDF")
    from asetools.analysis.rdf import rdf

sys.path.append("/proj/external_group/snu_micc/dgd03153/scripts/md")
from find_solvation_radius import identify_cutoff_scipy
from utils import get_atoms_list

# ====== Matplotlib Styling ======
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.linewidth'] = 2 #2, 1.5
plt.rcParams['axes.labelsize'] = 25 #35, 20
plt.rcParams['axes.labelpad'] = 15 #25, 10
plt.rcParams['xtick.labelsize'] = 20 #25, 15
plt.rcParams['ytick.labelsize'] = 20 #25, 15
plt.rcParams['xtick.major.size'] = 10 #10, 7
plt.rcParams['xtick.major.width'] = 2 #2, 1.5
plt.rcParams['xtick.minor.size'] = 5 #5, 3
plt.rcParams['xtick.minor.width'] = 2 #2, 1
plt.rcParams['ytick.major.size'] = 10 #10, 7
plt.rcParams['ytick.major.width'] = 2 #2, 1.5
plt.rcParams['ytick.minor.size'] = 5 #5, 3
plt.rcParams['ytick.minor.width'] = 2 #2, 1.5
plt.rcParams['legend.fontsize'] = 20 #30, 15

# ====== Helper Functions ======
def get_element_indices(atoms, elements):
    if elements[0].isdigit(): # if indices were provided
        indices = [int(i) for i in elements]
    elif "-" in elements[0]: # if ranges were provided
        start, end = elements[0].split("-")
        indices = list(range(int(start), int(end)+1))
    else: # if symbols were provided
        indices = []
        for i, el in enumerate(atoms.get_chemical_symbols()):
            if el in elements:
                indices.append(i)
    return indices

def process_trajectory(atoms_list, elements_i, elements_j):
    """Read and process ASE trajectory file, returning filtered atoms list."""
    orig_atoms_list = deepcopy(atoms_list)
    new_atoms_list = []
    for atoms in orig_atoms_list:
        new_atoms = Atoms(cell=atoms.get_cell(), pbc=atoms.get_pbc())
        for i, atom in enumerate(atoms):
            if i in elements_i or i in elements_j:
                new_atoms.append(atom)
        new_atoms_list.append(new_atoms)
    return new_atoms_list

def calculate_rdf(atoms_list, elements_i, elements_j, rmax, dr):
    """Compute RDF and coordination number."""    
    r_gr = rdf(atoms_list, idx1=elements_i, idx2=elements_j, binwidth=dr, rmax=rmax)
    r_gr = np.array(r_gr)
    r, gr = r_gr[:, 0], r_gr[:, 1]
    # Compute Coordination Number
    #n_particles = len([atom for atom in atoms_list[0] if atom.number != real_elements_i[0]])
    n_particles = len(atoms_list[0])
    volume = atoms_list[0].get_volume()
    number_density = n_particles / volume  # 1/Å³
    cn_integrand = 4 * np.pi * number_density * gr * r**2
    cn = cumulative_trapezoid(cn_integrand, r, initial=0)
    return r, gr, cn
    
def save_rdf_data(filename, r, gr, cn):
    """Save RDF and coordination number data to CSV."""
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["r (Å)", "g(r)", "n(r)"])
        for r_i, rdf_i, cn_i in zip(r, gr, cn):
            writer.writerow([round(r_i, 3), rdf_i, cn_i])
    print(f"RDF data saved to {filename}")

def plot_rdf_cn(r, gr, cn, output_file):
    """Plot RDF and coordination number and save figure."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(r, gr, label="g(r)")
    ax2 = ax.twinx()
    ax2.plot(r, cn, linestyle="--", label="n(r)")
    ax.set_xlabel(r"r ($\mathrm{\AA}$)")
    ax.set_ylabel("g(r)")
    ax2.set_ylabel("n(r)")
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")
    print(f"Plot saved to {output_file}")

# ====== Execution Flow ======
def main(params):
    temperature = params[0]
    ignore = params[1]
    total = params[2]
    if ignore is not None:
        ignore = int(ignore*FACTOR)
    if total is not None:
        total = int(total*FACTOR) + 1
    traj_files, total_atoms = get_atoms_list(temperature=temperature, verbose=False, return_traj=True)
    
    if total_atoms:
        if args.frequency > 1:
            total_atoms = [atoms for i, atoms in enumerate(total_atoms) if i % args.frequency == 0]
        
        sampled_atoms = total_atoms[ignore:total]
        
        print(f"Temperature: {temperature} K\n"
              f"Found trajectory files: {traj_files}\n"
              f"Number of total structures: {len(total_atoms)}\n"
              f"Number of sampled structures: {len(sampled_atoms)}\n")
        
        # Get all element indices present in the trajectory
        unique_elements = sorted(set(sampled_atoms[0].get_chemical_symbols()))
        
        Is, Js = [], []
        
        if args.multi_rdf:
            for i, j in zip(args.elements_i, args.elements_j or [None] * len(args.elements_i)):
                Is.append([i])
                if j is None:
                    # If j is not specified, default to "all other elements"
                    Js.append([e for e in unique_elements if e not in Is[-1]])
                else:
                    Js.append([j])
        else:
            elements_i = args.elements_i
            elements_j = args.elements_j
            if elements_i is None:
                elements_i = unique_elements
            if elements_j is None:
                elements_j = unique_elements
            Is.append(elements_i)
            Js.append(elements_j)
        
        for i, (elements_i, elements_j) in enumerate(zip(Is, Js)):
            
            # Convert to the integer indices
            elements_i = get_element_indices(sampled_atoms[0], elements_i)
            elements_j = get_element_indices(sampled_atoms[0], elements_j)

            atoms_list = process_trajectory(sampled_atoms, elements_i, elements_j)
            
            # Back to the symbols
            if isinstance(elements_i[0], int):
                elements_i = (np.array(sampled_atoms[0].get_chemical_symbols())[elements_i]).tolist()
            if isinstance(elements_j[0], int):
                elements_j = (np.array(sampled_atoms[0].get_chemical_symbols())[elements_j]).tolist()
            r, gr, cn = calculate_rdf(atoms_list, elements_i, elements_j, args.rmax, args.dr)
        
            str_elements_i = "+".join(sorted(list(set(elements_i)))) if args.elements_i is not None else "All"
            str_elements_j = "+".join(sorted(list(set(elements_j)))) if args.elements_j is not None else "All"
            time = int(len(atoms_list)/FACTOR)
            name = f"{temperature}K_{time}ps_{str_elements_i}-{str_elements_j}_{args.rmax}A"
            if args.label is not None:
                name = name + f"_{args.label}"
        
            rdf_filename = f"RDF_{name}.csv"
            save_rdf_data(rdf_filename, r, gr, cn)
        
            if args.plot:
                plot_filename = f"RDF_{name}.png"
                plot_rdf_cn(r, gr, cn, plot_filename)
        
            if args.shell:
                cutoff = identify_cutoff_scipy(r, gr)
                print(f"Identified solvation shell cutoff for {name}: {cutoff:.3f} Å")

if args.totals is None:
    args.totals = [None for i in range(len(args.temperatures))]
if len(args.totals) == 1:
    args.totals = [args.totals[0] for i in range(len(args.temperatures))]
if args.ignores is None:
    args.ignores = [0 for i in range(len(args.temperatures))]
if len(args.ignores) == 1:
    args.ignores = [args.ignores[0] for i in range(len(args.temperatures))]

assert len(args.temperatures) == len(args.totals) == len(args.ignores)

params = [[temperature, ignore, total] for temperature, ignore, total in zip(args.temperatures, args.ignores, args.totals)]
if args.temp_par:
    n_jobs = len(args.temperatures)
    if n_jobs > os.cpu_count():
        n_jobs = os.cpu_count()
    results = Parallel(n_jobs=n_jobs, verbose=1)(delayed(main)(i) for i in params)
else:
    for param in params:
        main(param)
