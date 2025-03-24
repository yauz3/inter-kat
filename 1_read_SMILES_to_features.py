#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 22/02/2025
# Author: Sadettin Y. Ugurlu

import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from mordred import Calculator, descriptors
import PyBioMed
from PyBioMed.PyMolecule import moe
from jpype import startJVM, shutdownJVM, JClass
import jpype
import os

print("The import has been successfully finished!")
#exit()

current_dir = os.path.dirname(os.path.abspath(__file__))


# Set the path to the CDK JAR file
CDK_JAR_PATH = os.path.expanduser(f"{current_dir}/bin/cdk.jar")
print("it is found!")

# Start the JVM with CDK JAR
if not jpype.isJVMStarted():
    startJVM(jpype.getDefaultJVMPath(), "-ea", f"-Djava.class.path={CDK_JAR_PATH}")

# Load CDK classes
CDKDescCalc = JClass("org.openscience.cdk.qsar.DescriptorEngine")

# Load Mordred Calculator
mordred_calc = Calculator(descriptors, ignore_3D=True)


# 📌 Step 1: Read SMILES from CSV
def read_smiles(csv_file, smiles_column="SMILES"):
    df = pd.read_csv(csv_file)
    df["Mol"] = df[smiles_column].apply(lambda x: Chem.MolFromSmiles(x) if pd.notna(x) else None)
    return df


# 📌 Step 2: Compute RDKit Descriptors
def compute_rdkit_features(mol):
    if mol is None:
        return {}

    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "TPSA": Descriptors.TPSA(mol),
        "RotatableBonds": Descriptors.NumRotatableBonds(mol),
    }


# 📌 Step 3: Compute Mordred Descriptors
def compute_mordred_features(mol):
    if mol is None:
        return {}

    try:
        mordred_features = mordred_calc(mol)
        return dict(mordred_features)
    except:
        return {}


# 📌 Step 4: Compute PyBioMed Descriptors
def compute_pybiomed_features(mol):
    if mol is None:
        return {}

    try:
        m = Chem.MolFromSmiles(mol) # mol should be SMILES
        mol_des = moe.GetMOE(m)
        return mol_des
    except:
        return {}


# 📌 Step 5: Compute CDK Descriptors
def compute_cdk_features(smiles):
    if smiles is None:
        return {}

    try:
        cdk_calculator = CDKDescCalc("org.openscience.cdk.qsar.descriptors.molecular.MolecularWeightDescriptor")
        cdk_descriptors = cdk_calculator.calculate(smiles)
        return {"CDK_MolecularWeight": cdk_descriptors}
    except:
        return {}


# 📌 Step 6: Process Dataset & Save Features
def generate_features(csv_file, output_csv):
    df = read_smiles(csv_file)

    # Compute all descriptors
    df["RDKit_Features"] = df["Mol"].apply(compute_rdkit_features)
    df["Mordred_Features"] = df["Mol"].apply(compute_mordred_features)
    df["PyBioMed_Features"] = df["Mol"].apply(compute_pybiomed_features)
    df["CDK_Features"] = df["SMILES"].apply(compute_cdk_features)

    # Flatten feature dictionaries
    rdkit_df = pd.DataFrame(df["RDKit_Features"].tolist())
    mordred_df = pd.DataFrame(df["Mordred_Features"].tolist())
    pybiomed_df = pd.DataFrame(df["PyBioMed_Features"].tolist())
    cdk_df = pd.DataFrame(df["CDK_Features"].tolist())

    # Merge all features into final dataset
    final_df = pd.concat([df["SMILES"], rdkit_df, mordred_df, pybiomed_df, cdk_df], axis=1)
    # Save results
    final_df.to_csv(output_csv, index=False)
    print(f"Feature extraction complete. File saved as {output_csv}")


def merge_csv_on_smiles(file1, file2, output_file="merged_output.csv"):
    """
    Reads two CSV files and merges them based on the 'SMILES' column.
    The first file (file1) remains the primary reference (left-join).

    :param file1: Path to the first CSV file (main dataset)
    :param file2: Path to the second CSV file (dataset to merge)
    :param output_file: Path for the merged output CSV file
    :return: Merged DataFrame
    """
    # Read both CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    for i, row in df2.iterrows():
        for col in df2.columns:
            try:
                df2.at[i, col]=float(df2.at[i, col])
            except:
                if isinstance(row[col], str) and col != "SMILES":  # Exclude SMILES column
                    df2.at[i, col] = None
    # Merge on 'SMILES', keeping the order of the first file
    merged_df = df1.merge(df2, on="SMILES", how="left")

    # Save merged DataFrame
    merged_df.to_csv(output_file, index=False)
    print(f"Merged file saved as {output_file}")

    return merged_df


# training
"""generate_features("kat_isomeric_smiles_train_cleaned_2_150.csv", "molecular_features.csv")

merged_df = merge_csv_on_smiles("kat_isomeric_smiles_train_cleaned_2_150.csv", "molecular_features.csv", "kat_training_ready.csv")
"""
# test
"""generate_features("kat_isomeric_smiles_test_cleaned_no_overlapped_2_48.csv", "molecular_features_test.csv")

merged_df = merge_csv_on_smiles("kat_isomeric_smiles_test_cleaned_no_overlapped_2_48.csv", "molecular_features_test.csv", "kat_test_ready.csv")"""


generate_features("all_data_ready.csv", "all_features.csv")

merged_df = merge_csv_on_smiles("all_data_ready.csv", "all_features.csv", "all_features_ready.csv")