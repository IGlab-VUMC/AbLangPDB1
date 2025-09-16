# """
# ##################################################
# <Claude Instructions>
# 1. Create a method to generate structures given a dataframe

# </Claude Instructions>
# #################################################
# <Human Instructions>
# Computers Ignore this!
# How to Use:
# 1. Separately generate the structures with ABodyBuilder2:
# - C:\Users\clint\VUMC\IGLabBoxContent - Documents\Members\Clint Holt\General_Antibody_Stuff\defining_pub_clone\antibody_data\dms_data\ABodyBuilder2_structs
# - C:\Users\clint\VUMC\IGLabBoxContent - Documents\Members\Clint Holt\General_Antibody_Stuff\defining_pub_clone\antibody_data\sabdab\ABodyBuilder2
# - C:\Users\clint\VUMC\IGLabBoxContent - Documents\Members\Clint Holt\General_Antibody_Stuff\defining_pub_clone\antibody_data\sabdab\all_structures_new\abs_only_space2fmt_240910

# 2. Align all to 1 and save aligned files

# 3. Run this!
# </Human Instructions>
# """


import pandas as pd
import numpy as np
from Bio.PDB import PDBParser, Superimposer
from itertools import combinations
import os
from tqdm import tqdm
from glob import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from Bio.PDB import PDBParser, Superimposer, PDBIO, Select

# --- IMGT CDR Definitions ---
# These are the residue ranges for CDRs according to the IMGT numbering scheme.
# The format is (start_residue, end_residue). The numbering is inclusive.
IMGT_CDR_DEFINITIONS = {
    'H': { # Heavy Chain
        'CDR1': (27, 38),
        'CDR2': (56, 65),
        'CDR3': (105, 116)
    },
    'L': { # Light Chain
        'CDR1': (27, 38),
        'CDR2': (56, 65),
        'CDR3': (105, 116)
    }
}

def is_in_cdr(residue):
    """
    Checks if a Biopython residue object is within any defined IMGT CDR.

    Args:
        residue (Bio.PDB.Residue.Residue): The residue object to check.

    Returns:
        bool: True if the residue is in a CDR, False otherwise.
    """
    chain_id = residue.get_parent().id
    # We only consider standard H and L chains for this analysis
    if chain_id not in IMGT_CDR_DEFINITIONS:
        return False

    res_id = residue.get_id()
    res_num = res_id[1] # Residue number (integer part)

    for cdr_name, (start, end) in IMGT_CDR_DEFINITIONS[chain_id].items():
        if start <= res_num <= end:
            return True
    return False

def get_residue_identifier(residue):
    """
    Creates a unique string identifier for a residue (e.g., 'H27', 'L113A').

    Args:
        residue (Bio.PDB.Residue.Residue): The residue object.

    Returns:
        str: A unique identifier string.
    """
    chain_id = residue.get_parent().id
    res_id = residue.get_id()
    res_num = res_id[1]
    insertion_code = res_id[2].strip() # Get insertion code, remove whitespace
    return f"{chain_id}{res_num}{insertion_code}"


def extract_cdr_ca_coords(pdb_filepath):
    """
    Parses a PDB file and extracts alpha-carbon (CA) coordinates for CDR residues.

    Args:
        pdb_filepath (str): Path to the PDB file.

    Returns:
        dict: A dictionary mapping residue identifiers (e.g., 'H27A') to their
              3D CA coordinates (a numpy array). Returns an empty dict if
              the file cannot be parsed or has no CDR CA atoms.
    """
    parser = PDBParser(QUIET=True)
    cdr_ca_coords = {}
    
    if not os.path.exists(pdb_filepath):
        print(f"Warning: PDB file not found at {pdb_filepath}")
        return cdr_ca_coords

    try:
        structure = parser.get_structure('antibody', pdb_filepath)
        for model in structure:
            for chain in model:
                for residue in chain:
                    if is_in_cdr(residue):
                        # Check if the Alpha Carbon atom exists
                        if 'CA' in residue:
                            ca_atom = residue['CA']
                            res_id_str = get_residue_identifier(residue)
                            cdr_ca_coords[res_id_str] = ca_atom.get_coord()
            # We typically only care about the first model in a PDB file
            break
    except Exception as e:
        print(f"Error parsing PDB file {pdb_filepath}: {e}")

    return cdr_ca_coords
    
def calculate_rmsd(coords1, coords2):
    """
    Calculates the Root Mean Square Deviation (RMSD) between two sets of coordinates.
    Assumes the coordinates are already aligned and in the same order.

    Args:
        coords1 (np.ndarray): An (N, 3) array of coordinates.
        coords2 (np.ndarray): An (N, 3) array of coordinates.

    Returns:
        float: The calculated RMSD value.
    """
    if coords1.shape != coords2.shape:
        raise ValueError("Coordinate arrays must have the same shape.")
    if coords1.shape[0] == 0:
        return 0.0 # No common atoms, so distance is 0

    diff = coords1 - coords2
    squared_dist = np.sum(diff * diff, axis=1)
    mean_squared_dist = np.mean(squared_dist)
    return np.sqrt(mean_squared_dist)

def rmsd_to_similarity(rmsd_values, sigma=2.0):
    """
    Converts RMSD values to a similarity score between 0 and 1.
    
    Args:
        rmsd_values (float or np.ndarray): A single RMSD value or an array of values.
        sigma (float): A scaling factor that defines the RMSD at which 
                       the similarity score starts to drop off significantly. 
                       A common choice for protein structures is between 2.0 and 5.0 Å.

    Returns:
        float or np.ndarray: The corresponding similarity score(s).
    """
    return np.exp(-(np.square(rmsd_values / sigma)))

def calculate_pairwise_cdr_rmsd(df: pd.DataFrame, pdb_directory: str = ".", save_file: str = 'pairwise_cdr_comparisons'):
    """
    Main function to calculate pairwise RMSD for CDRs of antibodies in a DataFrame.
    Creates rectangular matrices comparing train vs val and train vs test datasets.

    Args:
        df (pd.DataFrame): DataFrame where each index corresponds to an antibody.
                          Must contain a 'DATASET' column with values 'TRAIN', 'TEST', 'VAL'.
        pdb_directory (str): The directory where PDB files (named as {index}.pdb) are located.
        save_file (str): Base filename for output files. Will create 4 files with specific extensions.

    Returns:
        tuple: (train_vs_val_rmsd, train_vs_test_rmsd) - Two 2D numpy arrays with RMSD values.
    """
    # Validate DATASET column exists
    if 'DATASET' not in df.columns:
        raise ValueError("DataFrame must contain a 'DATASET' column")
    
    # Split dataframe by dataset
    train_df = df[df['DATASET'] == 'TRAIN']
    val_df = df[df['DATASET'] == 'VAL'] 
    test_df = df[df['DATASET'] == 'TEST']
    
    # Validate that we have data for all required datasets
    if len(train_df) == 0:
        raise ValueError("No TRAIN data found in dataset")
    if len(val_df) == 0:
        print("Warning: No VAL data found in dataset")
    if len(test_df) == 0:
        print("Warning: No TEST data found in dataset")
    
    train_indices = train_df.index.tolist()
    val_indices = val_df.index.tolist()
    test_indices = test_df.index.tolist()
    
    all_indices = train_indices + val_indices + test_indices
    
    print(f"Dataset sizes - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    # 1. Extract CDR data for all antibodies first
    all_cdr_data = {}
    print("Step 1: Extracting CDR alpha-carbon coordinates from PDB files...")
    for idx in tqdm(all_indices, desc="Extracting CDR coordinates"):
        pdb_filepath = os.path.join(pdb_directory, f"{idx}.pdb")
        all_cdr_data[idx] = extract_cdr_ca_coords(pdb_filepath)
    print("Extraction complete.")

    def calculate_rmsd_matrix(indices1, indices2, desc="Calculating RMSD"):
        """Calculate RMSD matrix between two sets of indices."""
        rmsd_matrix = np.zeros((len(indices1), len(indices2)))
        
        total_comparisons = len(indices1) * len(indices2)
        progress_bar = tqdm(total=total_comparisons, desc=desc)
        
        for i, idx1 in enumerate(indices1):
            for j, idx2 in enumerate(indices2):
                cdr_data1 = all_cdr_data[idx1]
                cdr_data2 = all_cdr_data[idx2]

                # Find the intersection of residue identifiers
                common_residues = sorted(list(set(cdr_data1.keys()) & set(cdr_data2.keys())))

                if not common_residues:
                    print(f"Warning: No common CDR residues between {idx1} and {idx2}.")
                    progress_bar.update(1)
                    continue

                # Create ordered lists of coordinates based on the common residues
                coords1 = np.array([cdr_data1[res_id] for res_id in common_residues])
                coords2 = np.array([cdr_data2[res_id] for res_id in common_residues])

                # Calculate RMSD
                rmsd = calculate_rmsd(coords1, coords2)
                rmsd_matrix[i, j] = rmsd
                
                progress_bar.update(1)
                
        progress_bar.close()
        return rmsd_matrix

    # 2. Calculate train vs val RMSD matrix (only if val data exists)
    train_vs_val_rmsd = None
    train_vs_val_sim = None
    if len(val_indices) > 0:
        print("\nStep 2: Calculating train vs val RMSD matrix...")
        train_vs_val_rmsd = calculate_rmsd_matrix(train_indices, val_indices, "Train vs Val RMSD")
        train_vs_val_sim = rmsd_to_similarity(train_vs_val_rmsd)
    
    # 3. Calculate train vs test RMSD matrix (only if test data exists)
    train_vs_test_rmsd = None
    train_vs_test_sim = None
    if len(test_indices) > 0:
        print("\nStep 3: Calculating train vs test RMSD matrix...")
        train_vs_test_rmsd = calculate_rmsd_matrix(train_indices, test_indices, "Train vs Test RMSD")
        train_vs_test_sim = rmsd_to_similarity(train_vs_test_rmsd)
    
    print("Calculation complete.")
    
    # Save files only if data exists
    files_saved = []
    if train_vs_val_rmsd is not None and train_vs_val_sim is not None:
        np.save(save_file + "_rmsds_train_vs_val.npy", train_vs_val_rmsd)
        np.save(save_file + "_sims_train_vs_val.npy", train_vs_val_sim)
        files_saved.extend(["_rmsds_train_vs_val.npy", "_sims_train_vs_val.npy"])
    
    if train_vs_test_rmsd is not None and train_vs_test_sim is not None:
        np.save(save_file + "_rmsds_train_vs_test.npy", train_vs_test_rmsd)
        np.save(save_file + "_sims_train_vs_test.npy", train_vs_test_sim)
        files_saved.extend(["_rmsds_train_vs_test.npy", "_sims_train_vs_test.npy"])
    
    print(f"Saved {len(files_saved)} files with base name: {save_file}")
    print(f"Files: {files_saved}")
    
    return train_vs_val_rmsd, train_vs_test_rmsd


def create_structures(ab_df: pd.DataFrame, struct_folder: str, refine: bool = False) -> None:
    """
    Generate structures for all antibodies in the provided dataframe.
    Do this using ABodyBuilder2 in IMGT numbering scheme.
    Args:
        ab_df (pd.Dataframe) - Antibody dataframe. Necessary components: 
            - columns: "HC_AA" and "LC_AA" used for heavy and light chain amino acid sequences
            - index: File is save as {index}.pdb
        struct_folder (str) - The file path of the folder to write structures to.
        refine (bool) - Whether or not to run openmm on the predicted structure
    """
    from ImmuneBuilder import ABodyBuilder2

    # Get an object for modeling
    predictor = ABodyBuilder2() 

    # Loop over the full dataframe to do missing predictions
    num_saved = 0
    for i, row in tqdm(ab_df.sample(frac=1).iterrows()):
        fname = os.path.join(struct_folder, f"{i}.pdb")
        if not os.path.exists(fname):

            # Actually predict the structures and write to a pdb file
            seqs = {"H": row["HC_AA"], "L": row["LC_AA"]}
            ab = predictor.predict(seqs)
            if refine:
                ab.save_single_unrefined(fname) 
            else:
                ab.save(fname)
            num_saved += 1
            print("Generated", f"{i}.pdb")

    print(f"Generated and saved {num_saved} pdb files at {struct_folder}.")



class CaOnlySelect(Select):
    """A selector class to save only the alpha-carbon atoms of a structure."""
    def accept_atom(self, atom):
        return atom.get_id() == 'CA'

def _align_worker(args: tuple):
    """
    Worker function to align a single PDB file to a reference.

    This function is designed to be called by a parallel processing pool.

    Args:
        args (tuple): A tuple containing the mobile file path, reference file
                      path, output file path, and a boolean to save only C-alphas.
    
    Returns:
        tuple: A tuple containing the RMSD and a status message.
    """
    mobile_file, ref_file, out_file, save_ca_only = args

    if os.path.exists(out_file):
        return None, f"Skipped: {os.path.basename(out_file)} already exists."

    try:
        parser = PDBParser(QUIET=True)
        ref_struct = parser.get_structure("ref", ref_file)
        mobile_struct = parser.get_structure("mobile", mobile_file)

        # Create dictionaries mapping (chain_id, residue_id) to CA atoms
        ref_ca_map = {
            (res.get_parent().get_id(), res.get_id()): res['CA']
            for res in ref_struct.get_residues() if 'CA' in res
        }
        mobile_ca_map = {
            (res.get_parent().get_id(), res.get_id()): res['CA']
            for res in mobile_struct.get_residues() if 'CA' in res
        }
        
        # Find common residues to use for alignment
        common_keys = sorted(list(set(ref_ca_map.keys()) & set(mobile_ca_map.keys())))
        
        if not common_keys:
            return None, f"Error: No common CA atoms found for {os.path.basename(mobile_file)}."

        # Create ordered lists of atoms for the superimposer
        fixed_atoms = [ref_ca_map[key] for key in common_keys]
        moving_atoms = [mobile_ca_map[key] for key in common_keys]

        # Perform the alignment
        super_imposer = Superimposer()
        super_imposer.set_atoms(fixed_atoms, moving_atoms)
        
        # Apply the transformation to the entire mobile structure
        super_imposer.apply(mobile_struct.get_atoms())
        
        # Save the transformed structure
        io = PDBIO()
        io.set_structure(mobile_struct)

        if save_ca_only:
            # Replicates original script's behavior of saving only CA atoms
            io.save(out_file, CaOnlySelect())
        else:
            # Saves the entire aligned structure (recommended)
            io.save(out_file)
        
        return super_imposer.rms, f"Aligned: {os.path.basename(out_file)}"

    except Exception as e:
        return None, f"Error processing {os.path.basename(mobile_file)}: {e}"

def align_all_to_1_biopython(
    ref_ab_file: str,
    ab_folder_in: str,
    ab_folder_out: str,
    n_processors: int = 16,
    save_ca_only: bool = False
) -> None:
    """
    Align and save all antibodies in a directory to 1 reference antibody using Biopython.

    Args:
        ref_ab_file (str): File path to the reference PDB file. All other
            structures will be structurally aligned to this one.
        ab_folder_in (str): Folder containing PDB files to be aligned.
        ab_folder_out (str): Folder to save all aligned PDB files to.
        n_processors (int): Number of processors to use for parallel execution.
            Defaults to 16.
        save_ca_only (bool): If True, saves only the C-alpha atoms, replicating
            the behavior of the original PyMOL script. If False (default), saves
            the full aligned structure, which is generally more useful.
    """
    os.makedirs(ab_folder_out, exist_ok=True)
    
    pdb_files = glob(os.path.join(ab_folder_in, "*.pdb"))
    
    # Create a list of tasks for the process pool
    tasks = [
        (fin, ref_ab_file, os.path.join(ab_folder_out, os.path.basename(fin)), save_ca_only)
        for fin in pdb_files
    ]
    
    print(f"Aligning {len(pdb_files)} structures to {os.path.basename(ref_ab_file)} using {n_processors} processors...")

    results = []
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=n_processors) as executor:
        # Use tqdm for a progress bar
        for result in tqdm(executor.map(_align_worker, tasks), total=len(tasks)):
            results.append(result)
            
    # Print a summary of the results
    print("\nAlignment complete. Summary:")
    success_count = 0
    for rms, message in results:
        if "Aligned" in message:
            success_count += 1
            # print(f"  - {message} (RMSD: {rms:.2f})") # Uncomment for verbose output
        elif "Skipped" not in message:
            print(f"  - {message}")
    print(f"\nSuccessfully aligned {success_count} / {len(pdb_files)} structures.")


def align_all_to_1(ref_ab_file: str, ab_folder_in: str, ab_folder_out: str) -> None:
    """
    Align and save all antibodies in a directory to 1 reference antibody.
    
    Args:
        ref_ab_file (str) - File path to a reference antibody file.
            All antibodies will be structurally aligned to this one
        ab_folder_in (str) - Folder containing pdb files of all antibodies you want
            aligned to the ref ab. These all end with .pdb and no other .pdb files
            can exist in this directory.
        ab_folder_out (str) - Folder to save all aligned pdb files to.
    """
    from pymol import cmd
    from glob import glob
    import random
    

    # Load in reference antibody, get alpha carbon atoms
    cmd.load(ref_ab_file, object="ref")
    cmd.create("ref_ca", "ref and name ca")
    cmd.delete("ref")
    # One at a time load in the other unaligned antibodies
    pdb_files = list(glob(os.path.join(ab_folder_in, "*.pdb")))
    random.shuffle(pdb_files)
    for fin in tqdm(pdb_files):
        # Skip if the output file exists
        fout = os.path.join(ab_folder_out, os.path.basename(fin))
        if not os.path.exists(fout):

            # Load and get just alpha carbons
            cmd.load(fin, object="to_aln")
            cmd.create("to_aln_ca", "to_aln and name ca")
            # Align to ref
            cmd.align("to_aln_ca", "ref_ca")
            # Save
            cmd.save(fout, "to_aln_ca")
            # Delete that object
            cmd.delete("to_aln to_aln_ca")
        

# --- Example Usage ---
def create_dummy_data():
    """Creates dummy PDB files and a DataFrame for demonstration."""
    print("Creating dummy data for demonstration...")
    
    # Create DataFrame
    data = {'Name': ['Antibody_A', 'Antibody_B', 'Antibody_C'], 'Source': ['Mouse', 'Human', 'Mouse']}
    df = pd.DataFrame(data, index=[0, 1, 2])
    
    # PDB content templates
    pdb_line = "ATOM{a_id:>7}  CA  GLY {c_id}{r_id:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C\n"

    # --- 0.pdb: H-CDR1 residues H26, H27, H28 ---
    with open("0.pdb", "w") as f:
        f.write(pdb_line.format(a_id=1, c_id='H', r_id=26, x=1.0, y=1.0, z=1.0))
        f.write(pdb_line.format(a_id=2, c_id='H', r_id=27, x=2.0, y=2.0, z=2.0))
        f.write(pdb_line.format(a_id=3, c_id='H', r_id=28, x=3.0, y=3.0, z=3.0))

    # --- 1.pdb: H-CDR1 residues H27, H28, H29 ---
    # Coords are shifted relative to 0.pdb for a non-zero RMSD
    with open("1.pdb", "w") as f:
        f.write(pdb_line.format(a_id=1, c_id='H', r_id=27, x=2.5, y=2.5, z=2.5)) # Common with 0
        f.write(pdb_line.format(a_id=2, c_id='H', r_id=28, x=3.5, y=3.5, z=3.5)) # Common with 0
        f.write(pdb_line.format(a_id=3, c_id='H', r_id=29, x=4.0, y=4.0, z=4.0)) # Unique to 1 vs 0
        
    # --- 2.pdb: H-CDR1 residues H28, H29, and L-CDR3 residue L90 ---
    with open("2.pdb", "w") as f:
        f.write(pdb_line.format(a_id=1, c_id='H', r_id=28, x=3.0, y=4.0, z=5.0)) # Common with 0 and 1
        f.write(pdb_line.format(a_id=2, c_id='H', r_id=29, x=4.0, y=5.0, z=6.0)) # Common with 1
        f.write(pdb_line.format(a_id=3, c_id='L', r_id=90, x=10.0, y=10.0, z=10.0)) # Unique

    print("Dummy files '0.pdb', '1.pdb', '2.pdb' created.")
    return df

def run_dummy_practice():
    # Create dummy data for the example
    antibody_df = create_dummy_data()
    print("\nInput DataFrame:")
    print(antibody_df)
    
    # Run the main analysis function
    print("\nStarting analysis...")
    distance_matrix = calculate_pairwise_cdr_rmsd(antibody_df)
    
    # Print the results
    print("\n--- Results ---")
    print("Pairwise RMSD Matrix:")
    print(distance_matrix)
    
    # Clean up dummy files
    for i in range(3):
        if os.path.exists(f"{i}.pdb"):
            os.remove(f"{i}.pdb")
    print("\nCleaned up dummy files.")


def run_actual_rmsds():

    dms_df = pd.read_parquet('dms_embeddedby_ablang-heavy.parquet').set_index("NAME")
    calculate_pairwise_cdr_rmsd(df=dms_df,
                                pdb_directory = '../../../antibody_data/dms_data/ABodyBuilder2_structs_aln',
                                save_file='dms_pairwise_cdr')
    
    sabdab_df = pd.read_parquet('sabdab_embeddedby_ablang-heavy.parquet').set_index("NAME_x")
    calculate_pairwise_cdr_rmsd(df=sabdab_df,
                                pdb_directory = '../../../antibody_data/sabdab/ABB2_unrefined',
                                save_file='sabdab_pairwise_cdr')



def run_gen_abb2_structs():       ################ Warning currently not refining
    ab_df_f = 'sabdab_embeddedby_ablang-heavy.parquet'
    output_folder = '/mnt/c/Users/clint/VUMC/IGLabBoxContent - Documents/Members/Clint Holt/General_Antibody_Stuff/defining_pub_clone/antibody_data/sabdab/ABB2_unrefined'
    ab_df = pd.read_parquet(ab_df_f)
    ab_df.set_index("NAME_x", inplace=True)
    create_structures(ab_df, output_folder, refine=False)

    ab_df_f = 'dms_embeddedby_ablang-heavy.parquet'
    output_folder = '/mnt/c/Users/clint/VUMC/IGLabBoxContent - Documents/Members/Clint Holt/General_Antibody_Stuff/defining_pub_clone/antibody_data/dms_data/ABodyBuilder2_structs'
    ab_df = pd.read_parquet(ab_df_f)
    ab_df.set_index("NAME", inplace=True)
    create_structures(ab_df, output_folder, refine=False)

def run_aln_all():
    align_all_to_1_biopython(
        ref_ab_file='../../../antibody_data/dms_data/ABodyBuilder2_structs/1-57.pdb',
        ab_folder_in='../../../antibody_data/dms_data/ABodyBuilder2_structs',
        ab_folder_out='../../../antibody_data/dms_data/ABodyBuilder2_structs_aln',
        n_processors=16 # Using 4 for demonstration; can be increased
    )
    align_all_to_1_biopython(
        ref_ab_file='../../../antibody_data/sabdab/ABB2_unrefined/unaln/1ADQ_H_L_A.pdb',
        ab_folder_in='../../../antibody_data/sabdab/ABB2_unrefined/unaln',
        ab_folder_out='../../../antibody_data/sabdab/ABB2_unrefined',
        n_processors=16 # Using 4 for demonstration; can be increased
    )

if __name__ == '__main__':
    # run_gen_abb2_structs()
    # run_aln_all()
    run_actual_rmsds()
    # pass