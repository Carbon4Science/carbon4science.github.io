#!/home/junyoung/anaconda3/envs/chgnet040/bin/python
import warnings
for category in (UserWarning, DeprecationWarning):
    warnings.filterwarnings("ignore", category=category)
import os
from pathlib import Path
from glob import glob
import shutil
import argparse
import time
from datetime import timedelta
import numpy as np
from joblib import Parallel, delayed
import traceback
import platform

from ase import units
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
#from ase.optimize.optimize import AbnormalMDError

import torch
#from chgnet.model import CHGNetCalculator, CHGNet, MolecularDynamics
from chgnet.model.model_my import CHGNet
from chgnet.model.dynamics_my import CHGNetCalculator, MolecularDynamics

import sys

SYSTEM = platform.system()
if SYSTEM == "Windows":
    sys.path.append("D:/utils/")
elif SYSTEM == "Linux":
    sys.path.append("/mifs/dgd03153/scripts/md")
    sys.path.append("/mifs/dgd03153/continual")
from nvtnosehoover import NVTNoseHoover
#from nptnosehoover import NoseHooverNPT
from utils import parse_args, to_shell, get_logger
from lora_chgnet import LoRALinear

parser = argparse.ArgumentParser(description="Run MD simulations from CIF files using CHGNet.")
parser.add_argument(
    '-c', '--cifs', type=str, nargs="+",
    help="List of CIF file paths to be simulated. Provide one or more paths."
)
parser.add_argument(
    "-p", "--potentials", type=str, nargs="+", default=["0.3.0"],
    help="List of model directory to use for simulations. Defaults to ['0.3.0']."
)
parser.add_argument(
    "-sd", "--save_dirs", type=str, nargs="+", default=None,
    help="List of directory names to save the results. Defaults to the current directory."
)
parser.add_argument(
    "-e", "--ensembles", type=str, nargs="+", default=["nvt"],
    help="List of thermodynamic ensembles (e.g., nvt, npt) to use for each simulation. Defaults to ['nvt']."
)
parser.add_argument(
    "-uc", "--use_custom", action="store_true",
    help="Wether to use the ASE MD wrapper module provided by the developer, or the custom Nose-Hoover thermostat. "
         "Defaults to false, meaning that the former is used."
)
parser.add_argument(
    "-th", "--thermostat", type=str, default="Nose-Hoover",
    help="Thermostat to use. Defaults to Nose-Hoover. Available thermostats: Nose-Hoover, Berendsen, Berendsen_inhomogeneous, NPT_Berendsen."
       
)
parser.add_argument(
    '-tm', '--temperatures', type=int, nargs="+",
    help="List of temperatures for each CIF file, separated by '1' as a delimiter between structures. "
         "Example: 100 200 300 1 200 400 600 for two structures."
)
parser.add_argument(
    "-st", "--steps", type=int, default=[275000], nargs="+",
    help="List of step counts for each CIF file, separated by '1' as a delimiter if needed. "
         "Defaults to [275000]."
)
parser.add_argument(
    "-ts", "--timestep", type=float, default=2,
    help="Time step size in femtoseconds for the MD simulation. Defaults to 2 fs."
)
parser.add_argument(
    "-sk", "--step_skip", type=int, default=[50], nargs="+",
    help="Number of MD steps to skip between saving trajectory frames. Can be a list for each structure. "
         "Defaults to [50]."
)
parser.add_argument(
    "-se", "--serial", action="store_true",
    help="Run simulations serially instead of in parallel."
)
parser.add_argument(
    '-b', '--batch_size', type=int,
    help="Total number of CIFs per batch when distributing simulations."
)
parser.add_argument(
    '-i', '--batch_index', type=int,
    help="Index of the current batch (used with --batch_size)."
)
parser.add_argument(
    '-s', '--seed', type=int, default=42,
    help="Random seed in velocity initialization for reproducibility."
)
parser.add_argument(
    '-k', '--bulk_modulus', type=float,
    help="Bulk modulus to use in NPT simulations."
)
args = parser.parse_args()
 
class CustomMolecularDynamics:
    def __init__(self, model_dir, cif, save_dir):
        self.cif = cif
        self.atoms = read(cif)
        cif_name = os.path.basename(cif).replace(".cif", "")
        self.save_path = os.path.join(save_dir, f"{cif_name}-s{args.seed}")
        if not os.path.isdir(self.save_path):
            try:
                os.makedirs(self.save_path)
                print(f"Made {self.save_path}.")
            except:
                print(f"{self.save_path} already exists.")
                pass

        self.timestep = args.timestep
        if "H" in self.atoms.get_chemical_symbols() and self.timestep > 1:
            self.timestep = args.timestep = 1
            
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        if model_dir == "0.3.0":
            self.potential = CHGNet.load()
        else:
            try:
                self.potential = CHGNet.from_file(os.path.join(model_dir, "model.pth.tar")).to(device)
            except Exception:
                self.potential = torch.load(os.path.join(model_dir, "model.pt"), map_location=device)
        
        # Setting logger
        model_name = "_".join(Path(model_dir.replace("../", "").strip("/")).parts)
        save_name = "_".join(Path(self.save_path.replace(model_dir, "").replace("../", "").strip("/")).parts)
        self.logger = get_logger(f"MD_{model_name}_{save_name}.log")
        self.logger.info(f"System: {SYSTEM}")
        self.logger.info(f"Model: {model_dir}")
        for k_args, v_args in vars(args).items():
            self.logger.info(f'{k_args} = {v_args}')
        
        if all(
                [os.path.isfile(f"{model_dir}/test_f_mae.npy"),
                 os.path.isfile(f"{model_dir}/test_e_ae.npy"),
                 os.path.isfile(f"{model_dir}/test_z_per_structure.npy"),
                 os.path.isfile(f"{model_dir}/train_z_per_structure.npy")]
                ):
            print("Loading npy files...")
            self.test_f_mae = np.load(f"{model_dir}/test_f_mae.npy")
            self.test_e_ae = np.load(f"{model_dir}/test_e_ae.npy")
            self.test_z_per_structure = np.load(f"{model_dir}/test_z_per_structure.npy")
            self.train_z_per_structure = np.load(f"{model_dir}/train_z_per_structure.npy")
        else:
            self.test_f_mae = None
            self.test_e_ae = None
            self.train_z_per_structure = None
            self.test_z_per_structure = None
            
            
    def print_dyn(self, start_time):
        if args.use_custom:
            imd = self.md.get_number_of_steps()
            etot  = self.atoms.get_total_energy()
            temp_K = self.atoms.get_temperature()
            stress = self.atoms.get_stress(include_ideal_gas=True)/units.GPa
            stress_ave = (stress[0]+stress[1]+stress[2])/3.0
            volume = self.atoms.get_volume()
            elapsed_time = time.perf_counter() - start_time
            self.logger.info(f"{imd: >3}    {etot:.3f}    {temp_K:.2f}    {stress_ave:.3f}    {volume:.3f}    {elapsed_time:.3f}")
        else:
            imd = self.md.dyn.get_number_of_steps()
            etot  = self.atoms.get_total_energy()
            temp_K = self.atoms.get_temperature()
            stress = self.atoms.get_stress(include_ideal_gas=True)/units.GPa
            stress_ave = (stress[0]+stress[1]+stress[2])/3.0
            volume = self.atoms.get_volume()
            elapsed_time = time.perf_counter() - start_time
            self.logger.info(f"{imd: >3}    {etot:.3f}    {temp_K:.2f}    {stress_ave:.3f}    {volume:.3f}    {elapsed_time:.3f}")


    def heatup(
            self, 
            ensemble="nvt", 
            init_temp=100,
            target_temp=600, 
            steps=None, 
            nblock=50,
            ):
        self.traj_file = f"{self.save_path}/heatup_{target_temp}K.traj"
        self.log_file = f"{self.save_path}/heatup_{target_temp}K.log"
        elapsed_time = "00:00:00"
        heatup_exist = False
        if os.path.isfile(self.traj_file) and os.path.isfile(self.log_file):
            try:
                self.atoms = Trajectory(self.traj_file)[-1]
                print("Found existing heated up structures.")
                heatup_exist = True
            except:
                err_msg = traceback.format_exc()
                self.logger.error(f"Error in reading {self.traj_file}\n{err_msg}")
                self.logger.waning("Heatup again")
                os.remove(self.traj_file)
                os.remove(self.log_file)
        if not heatup_exist:
            if steps is None:
                steps = (target_temp-init_temp)*2
            if steps < 0:
                steps = 0
            total_time = steps*self.timestep/1000
            self.logger.info(f"Heatup (Target {target_temp} K, {total_time:.1f} ps ({steps} steps)).")
            if args.serial:
                self.logger.info("Step    Etot (eV)    T (K)    S (GPa)    V(A3)    elapsed_time (sec)")
            total_elapsed_time = 0
            for i in range(steps):
                if i % nblock == 0:
                    temperature = init_temp + int((target_temp - init_temp)*(i/steps))
                    if i == 0:
                        if args.seed:
                            rng = np.random.RandomState(args.seed)
                        else:
                            rng = None
                        MaxwellBoltzmannDistribution(self.atoms, temperature_K=init_temp, rng=rng)
                        Stationary(self.atoms)
                        ZeroRotation(self.atoms)
                    calc = CHGNetCalculator(model=self.potential)
                    if ensemble == "nve":
                        scale_factor = np.sqrt(temperature / self.atoms.get_temperature())
                        self.atoms.set_velocities(self.atoms.get_velocities()*scale_factor)
                        self.md = MolecularDynamics(
                            atoms=self.atoms,
                            model=calc,
                            ensemble="nve",
                            temperature=temperature,
                            timestep=self.timestep,
                            trajectory=self.traj_file,
                            logfile=self.log_file,
                            loginterval=10,
                            append_trajectory=True
                            )
                    else:
                        if args.use_custom:
                            self.atoms.calc = calc
                            self.md = NVTNoseHoover(
                                atoms=self.atoms, 
                                timestep=self.timestep*units.fs, 
                                temperature_K=temperature, 
                                trajectory=self.traj_file,
                                logfile=self.log_file,
                                loginterval=10,
                                append_trajectory=True
                            )
                            if args.serial:
                                self.md.attach(self.print_dyn, interval=nblock, start_time=time.perf_counter())
                        else:
                            self.md = MolecularDynamics(
                                atoms=self.atoms,
                                model=calc,
                                ensemble="nvt",
                                thermostat=args.thermostat,
                                temperature=temperature,
                                timestep=self.timestep,
                                trajectory=self.traj_file,
                                logfile=self.log_file,
                                loginterval=10,
                                append_trajectory=True
                                )
                            if args.serial:
                                self.md.attach({
                                    "function": self.print_dyn,
                                    "interval": nblock, 
                                    "start_time": time.perf_counter()
                                    })
                    start = time.perf_counter()
                    try:
                        self.md.run(nblock)
                    except Exception:
                        err_msg = traceback.format_exc()
                        self.logger.error(f"Error in heatup {self.traj_file}\n{err_msg}")
                    end = time.perf_counter()
                    total_elapsed_time += (end-start)
                    #self.atoms = Trajectory(self.traj_file)[-1]
            elapsed_time = str(timedelta(seconds=int(total_elapsed_time)))
            elapsed_time = (lambda x: '0'+x if len(x) < 8 else x)(elapsed_time)
            self.logger.info(f"{self.cif} heatup (Target {target_temp} K, {total_time:.1f} ps) completed. (Elapsed time: {elapsed_time})")

  
    def run_md(
            self, 
            ensemble="nvt",
            temperature=600, 
            steps=50000, 
            nblock=50):
        trajectories = glob(f"{self.save_path}/{ensemble}_{temperature}K-*.traj")
        logfiles = glob(f"{self.save_path}/{ensemble}_{temperature}K-*.log")
        trajectories = sorted(trajectories, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("-")[-1]))
        logfiles = sorted(logfiles, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("-")[-1]))
        total_time = steps*self.timestep/1000
        if trajectories and logfiles:
            try:
                self.atoms = Trajectory(trajectories[-1])[-1]
                print(f"Found existing trajectory files: {trajectories}")
            except:
                err_msg = traceback.format_exc()
                self.logger.error(f"{self.cif} Error in reading {trajectories[-1]}\n{err_msg}")
                self.logger.warning(f"Run {ensemble} again.")
                os.remove(trajectories[-1])
                os.remove(logfiles[-1])
                del trajectories[-1]
                del logfiles[-1]
                try:
                    self.atoms = Trajectory(trajectories[-1])[-1]
                except:
                    pass

            last_step = 0
            for logfile in logfiles:
                if SYSTEM == "Windows":
                    last_step += int(float(os.popen(f"powershell -command \"Get-Content \'{logfile}\' -Tail 1\"").read().split()[0])*1000/self.timestep)
                elif SYSTEM == "Linux":
                    last_step += int(float(os.popen(f"tail -n 1 {to_shell(logfile)}").read().split()[0])*1000/self.timestep)
            steps = steps - last_step      

        indices = [0] + [int(os.path.splitext(os.path.basename(i))[0].split("-")[-1]) for i in trajectories]
        self.traj_file = f"{self.save_path}/{ensemble}_{temperature}K-{indices[-1]+1}.traj"
        self.log_file = f"{self.save_path}/{ensemble}_{temperature}K-{indices[-1]+1}.log"

        if steps > 0: 
            calc = CHGNetCalculator(
                model=self.potential,
                save_dir=self.save_path,
                temperature=temperature,
                train_z_per_structure=self.train_z_per_structure,
                test_z_per_structure=self.test_z_per_structure,
                test_f_mae=self.test_f_mae,
                test_e_ae=self.test_e_ae,
                prefix=ensemble,
                )
            self.logger.info(f"Running {ensemble.upper()} MD ({temperature} K, {self.timestep} fs, {total_time:.1f} ps ({steps} steps left, {self.traj_file}))")
            if args.use_custom:
                self.atoms.calc = calc
                if ensemble == "nvt":
                    self.md = NVTNoseHoover(
                        atoms=self.atoms, 
                        timestep=self.timestep*units.fs, 
                        temperature_K=temperature, 
                        trajectory=self.traj_file,
                        logfile=self.log_file,
                        loginterval=nblock
                    )
                elif ensemble == "npt":
                    raise NotImplementedError("Nose Hoover NPT has a problem for now.")
                    #self.md = NoseHooverNPT(
                    #    atoms=self.atoms, 
                    #    timestep=self.timestep*units.fs, 
                    #    temperature=temperature,
                    #    trajectory=self.traj_file,
                    #    logfile=self.log_file,
                    #    loginterval=nblock,
                    #    append_trajectory=False,
                    #)
                if args.serial:
                    self.md.attach(self.print_dyn, interval=nblock, start_time=time.perf_counter())
                    self.logger.info("Step    Etot (eV)    T (K)    S (GPa)    V(A3)    elapsed_time (sec)")
            else:
                self.md = MolecularDynamics(
                    atoms=self.atoms,
                    model=calc,
                    ensemble=ensemble,
                    thermostat=args.thermostat,
                    temperature=temperature,
                    timestep=self.timestep,
                    bulk_modulus=args.bulk_modulus,
                    trajectory=self.traj_file,
                    logfile=self.log_file,
                    loginterval=nblock,
                    append_trajectory=False,       
                    )
                if args.serial:
                    self.md.attach({
                        "function": self.print_dyn,
                        "interval": nblock, 
                        "start_time": time.perf_counter()
                        })
                    self.logger.info("Step    Etot (eV)    T (K)    S (GPa)    V(A3)    elapsed_time (sec)")
            start = time.perf_counter()
            try:
                self.md.run(steps)
            except Exception:
                err_msg = traceback.format_exc()
                self.logger.error(f"Error in {ensemble.upper()} {self.traj_file}\n{err_msg}")
            end = time.perf_counter()
            elapsed_time = str(timedelta(seconds=int(end-start)))
            elapsed_time = (lambda x: '0'+x if len(x) < 8 else x)(elapsed_time)
            self.logger.info(f"{self.cif} {ensemble.upper()} ({temperature} K, {total_time:.1f} ps) completed. (Elapsed time: {elapsed_time} ({end-start} sec))")


def main(
        cif,
        model_dir,
        save_dir,
        ensemble,
        temperature,
        steps,
        step_skip,
        ):
    try:
        if not glob(os.path.join(save_dir, "failed/heatup_{temperature}K*")) and \
            not glob(os.path.join(save_dir, "failed/{temperature}K*")):
            md = CustomMolecularDynamics(model_dir=model_dir, cif=cif, save_dir=save_dir)
            md.heatup(init_temp=100, target_temp=temperature, nblock=50)
            md.run_md(ensemble=ensemble, temperature=temperature, steps=steps, nblock=step_skip)        
    except ValueError:
        #md.logger.error(f"Abnormal behavior was observed in MD simulations. ({cif}, {temperature} K)")
        err_msg = traceback.format_exc()
        md.logger.error(f"Error in MD simulations. ({cif}, {temperature} K)\n{err_msg}")
        #failed = os.path.join(save_dir, "failed")
        #if not os.path.isdir(failed):
        #    os.mkdir(failed)
        #shutil.move(md.traj_file, failed)
        #shutil.move(md.log_file, failed)

if args.cifs is None:
    cif_list = sorted(glob("*.cif"))
    if args.batch_size is not None and args.batch_index is not None:
        start = int((args.batch_index-1)*args.batch_size)
        end = int(args.batch_index*args.batch_size)
        cif_list = cif_list[start:end]
else:
    cif_list = args.cifs

if len(cif_list) != len(args.potentials):
    if len(cif_list) == 1:
        cif_list = [cif_list[0] for i in args.potentials]
        potential_list = args.potentials
    elif len(args.potentials) == 1:
        potential_list = [args.potentials[0] for i in cif_list]
else:
    potential_list = args.potentials

if args.save_dirs is not None:
    if len(args.save_dirs) == 1:
        save_dir_list = [args.save_dirs[0] for i in cif_list]
    else:
        save_dir_list = args.save_dirs
else:
    save_dir_list = ["." for cif in cif_list]
    
if len(args.ensembles) == 1:
    ensemble_list = [args.ensembles[0] for i in cif_list]
else:
    ensemble_list = args.ensembles
ensemble_list = [e.lower() for e in ensemble_list]

assert len(cif_list) == len(potential_list) == len(save_dir_list) == len(ensemble_list)

if len(args.steps) == 1:
    args.steps = [args.steps[0] if i != 1 else 1 for i in args.temperatures]
if len(args.step_skip) == 1:
    args.step_skip = [args.step_skip[0] if i != 1 else 1 for i in args.temperatures]

temperature_llist = parse_args(args.temperatures, len(cif_list))
steps_llist = parse_args(args.steps, len(cif_list))
step_skip_llist = parse_args(args.step_skip, len(cif_list))

params = []
for cif, potential, save_dir, ensemble, temperature_list, steps_list, step_skip_list in zip(
        cif_list, potential_list, save_dir_list, ensemble_list, temperature_llist, steps_llist, step_skip_llist
        ):
    for temperature, steps, step_skip in zip(temperature_list, steps_list, step_skip_list):
        params.append([cif, potential, save_dir, ensemble, temperature, steps, step_skip])
        
for param in params:
    print(param)

n_jobs = len(params)
if len(params) > os.cpu_count():
    n_jobs = os.cpu_count()

if __name__ == "__main__":
    if args.serial:
        print("Running in serial...")
        for param in params:
            main(*param)
    else:
        Parallel(n_jobs=n_jobs, verbose=1, batch_size=1)(delayed(main)(*param) for param in params)
