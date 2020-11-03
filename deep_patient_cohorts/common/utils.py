import re
from typing import List

import pandas as pd

POSITIVE = 1
NEGATIVE = -1
ABSTAIN = 0

IS_A_CID = "116680003"

# The textual descriptions of every child of the Snowmed Concept "Heart Diseases" (56265001)
HEART_DISEASES = [
    "Acute heart disease",
    "Heart valve disorder",
    "Valvular heart disease",
    "Heart valve disease",
    "Myxedema heart disease",
    "Myxoedema heart disease",
    "Hereditary coproporphyria",
    "Berger-Goldberg syndrome",
    "CPO deficiency",
    "CPRO deficiency",
    "Porphyria hepatica II",
    "Watson syndrome",
    "Cardiomegaly",
    "Cardiac hypertrophy",
    "Cor bovinum",
    "Coronary artery disease",
    "Cyanotic congenital heart disease",
    "Congenital heart disease",
    "Congenital anomaly of heart",
    "Senile cardiac amyloidosis",
    "AS transthyretin amyloidosis",
    "Cardiac amyloidosis",
    "Abnormal fetal heart beat first noted during labor AND/OR delivery in liveborn infant",
    "Abnormal fetal heart beat first noted during labor or delivery in liveborn infant",
    "Rheumatic heart disease",
    "Rheumatic carditis",
    "Rheumatic pancarditis",
    "Ventricular tachycardia",
    "Cardiac complication",
    "Mechanical complication due to heart valve prosthesis",
    "Conduction disorder of the heart",
    "Mulberry heart disease",
    "Disease of pericardium",
    "Pericardial disorder",
    "Myocardial disease",
    "Thyrotoxic heart disease",
    "Cardiopulmonary schistosomiasis",
    "Pancarditis",
    "Hypertensive heart disease",
    "Hypertensive cardiopathy",
    "Hypertensive cardiomegaly",
    "Hypertensive cardiovascular disease",
    "Failed attempted abortion with cardiac arrest AND/OR failure",
    "Failed attempted abortion with cardiac arrest or failure",
    "Cardiac sarcoidosis",
    "Heart disease in mother complicating pregnancy, childbirth AND/OR puerperium",
    "Heart disease, in mother complicating pregnancy, childbirth or puerperium",
    "Neoplasm of heart",
    "Neoplasm of mesothelial tissue of pericardium",
    "Cor pulmonale",
    "Acardiac monster",
    "Acardius",
    "Acardiacus",
    "Acardiac twins",
    "Injury of heart",
    "Cardiac edema",
    "Sudden cardiac death",
    "Functional cardiac disorder",
    "Endocardial disease",
    "Healed bacterial endocarditis",
    "Disorder of cardiac function",
    "Endocardiopathy",
    "Neoplasm of heart AND/OR pericardium",
    "Chronic heart disorder",
    "Chronic disorder of heart",
    "Chronic disease of heart",
    "Chronic heart disease",
    "Infectious disease of heart",
    "Structural disorder of heart",
    "Biopsy of lesion of wall of heart",
    "Malignant hypertensive heart disease",
    "Angina",
    "Cardiac angina",
    "Angina pectoris",
    "Stenocardia",
    "Anginal syndrome",
    "AP - Angina pectoris",
    "Ischemic heart disease - angina",
    "Ischaemic heart disease - angina",
    "Certain current complications following acute myocardial infarction",
    "Other forms of heart disease",
    "Sudden cardiac death, so described",
    "Cardiac septal defect, acquired",
    "[X]Other heart disorders in other diseases classified elsewhere",
    "Cardiac arrest following abortive pregnancy",
    "Cardiac failure following abortive pregnancy",
    "Obstetric anaesthesia with cardiac complications",
    "Obstetric anesthesia with cardiac complications",
    "Cardiac complications of anesthesia during labor and delivery",
    "Cardiac complications of anaesthesia during labour and delivery",
    "Eisenmenger's complex",
    "Atresia of cardiac vein",
    "Hemicardia",
    "Anomalous bands of heart",
    "Injury to heart and lung",
    "Sympathotonic orthostatic hypotension",
    "Athlete's heart",
    "Athletic heart syndrome",
    "Cardiac transplant disorder",
    "Type 2 dissection of thoracic aorta",
    "Aortic root dissection",
    "Type II dissection of thoracic aorta",
    "Leakage due to cardiac device",
    "Disorder of implantable defibrillator",
    "ICD - Disorder of implantable cardiac defibrillator",
    "Disorder of implantable cardiovertor",
    "Heart disease during pregnancy",
    "Mitral sub-valve apparatus fibrosis",
    "Mitral sub-valve apparatus calcification",
    "Aortic root congenital abnormality",
    "Right atrial abnormality",
    "RA - Right atrial abnormality",
    "Left atrial abnormality",
    "LA - Left atrial abnormality",
    "Right ventricle abnormality",
    "RV - Right ventricular abnormality",
    "Right ventricular abnormality",
    "SOV - Sinus of Valsalva abnormality",
    "Sinus of Valsalva abnormality",
    "Cardiac complications of care",
    "Disorder of cardiac pacemaker system",
    "Pulmonary heart disease",
    "Cardiac disease in pregnancy",
    "Cardiac failure developing in the perinatal period",
    "Neonatal cardiac failure",
    "Acute/subacute carditis",
    "Enlarge heart septal foramen",
    "Cardiac glycogen phosphorylase kinase deficiency",
    "Tuberculosis of heart",
    "Tumour of heart",
    "Tumor of heart",
    "CHD - Congenital heart disease",
    "Amyloid heart muscle disease",
    "Abnormal fetal heart beat first noted during labour AND/OR delivery in liveborn infant",
    "Cardiac complication of procedure",
    "Disorder of heart valve",
    "HHD - Hypertensive heart disease",
    "HCP - Hereditary coproporphyria",
    "Coproporphyrinogen oxidase deficiency",
    "Watson's syndrome",
    "CPO - Coproporphyrinogen oxidase deficiency",
    "Cardiac oedema",
    "VT - Ventricular tachycardia",
    "Cardiac arrhythmia",
    "Cardiac dysrhythmia",
    "Arrhythmia",
    "Cardiac dysrhythmias",
    "Cardiac arrhythmias",
    "Disorder of heart rhythm",
    "Disorder of heart conduction",
    "Disorder of pericardium",
    "Disorder of myocardium",
    "Disorder of heart muscle",
    "Complication of transplanted heart",
    "Carditis",
    "Abscess of cardiac septum",
    "Ventricular tachyarrhythmia",
    "Disease of the coronary arteries",
    "Disorder of coronary artery",
    "Ischemic heart disease",
    "Ischaemic heart disease",
    "IHD - Ischemic heart disease",
    "IHD - Ischaemic heart disease",
    "Disorder of cardiac ventricle",
    "Sinistrocardia",
    "Left sided heart",
    "Heart predominantly in left hemithorax",
    "Disorder of transplanted heart",
    "Fetal heart disorder",
    "Heart disease due to ionizing radiation",
    "Enlarged septal foramen of heart",
    "Heart disease due to ionising radiation",
    "Heart disease due to radiation",
    "Abnormal foetal heart beat first noted during labour AND/OR delivery in liveborn infant",
    "Eisenmenger complex",
    "Outflow tract abnormality in solitary indeterminate ventricle",
    "Congenital abnormality of middle cardiac vein",
    "Enlarged heart",
    "Cardiac abnormality due to heart abscess",
    "Disorder of left atrium",
    "Disorder of right atrium",
    "Right heart failure due to disorder of lung",
    "Right heart failure due to pulmonary disease",
    "Termination of pregnancy complicated by cardiac arrest and/or failure",
    "Induced termination of pregnancy complicated by cardiac arrest and/or failure",
    "Failed attempted termination of pregnancy with cardiac arrest and/or failure",
    "Foetal heart disorder",
    "Cardiac disorder due to typhoid fever",
    "Heart disease co-occurrent with human immunodeficiency virus infection",
    "Heart disease caused by ionizing radiation",
    "Heart disease caused by ionising radiation",
    "Heart disease caused by radiation",
    "Cardiac complication of anesthesia during the puerperium",
    "Cardiac complication of anaesthesia during the puerperium",
    "Acquired cardiac septal defect",
    "Make heart septal defect",
    "Dissection of ascending aorta",
    "Disorder of atrium following procedure",
    "Postprocedural atrial complication",
    "Stiff heart syndrome",
    "Wild-type transthyretin cardiac amyloidosis",
    "Disorder of cardiac atrium",
    "Atrial cardiopathy",
]

# The textual descriptions of every child of the Snowmed Concept "Cardiovascular Agents" (373247007)
CARDIAC_DRUGS = [
    "Capillary active drug",
    "Capillary protectant",
    "Vasoprotectant",
    "3-Methoxy-o-desmethylencainide",
    "Pirmenol",
    "Saluretic - chemical",
    "Capillary active agent",
    "Antiplatelet agent",
    "Hypotensive agent",
    "Diuretic",
    "Thrombolytic agent",
    "Antiarrhythmic agent",
    "Vasoconstrictor",
    "Inotropic agent",
    "Antivaricose agent",
    "Cardiac agent",
    "beta-Blocking agent",
    "Cardiac adrenergic blocking agent",
    "Antilipemic agent",
    "Alpha-adrenergic blocking agent",
    "Calcium channel blocker",
    "Antilipaemic agent",
    "Beta-adrenoceptor blocking agent",
    "Cardiac non-specific adrenergic blocking agent",
    "Lipid lowering agent",
    "Alpha-adrenoceptor blocking agent",
    "Capillary-active drug",
    "Cardioplegic solution agent",
    "Antihypertensive agent",
    "Ivabradine",
    "Ivabradine hydrochloride",
    "Adenosine A2 receptor agonist",
    "Adenosine A2 agonist",
    "Alpha adrenergic receptor antagonist",
    "Beta adrenergic receptor antagonist",
    "Substance with adenosine A2 receptor agonist mechanism of action",
    "Substance with alpha adrenergic receptor antagonist mechanism of action",
    "Substance with beta adrenergic receptor antagonist mechanism of action",
    "Substance with calcium channel blocker mechanism of action",
    "3-methoxy-o-desmethylencainide",
    "Modecainide",
    "Vasopressor",
]


def reformat_icd_code(icd_code: str, is_diag: bool = True) -> str:
    """Put a period in the right place because the MIMIC-3 data files exclude them.
    Generally, procedure ICD codes have dots after the first two digits, while diagnosis
    ICD codes have dots after the first three digits.
    Adopted from: https://github.com/jamesmullenbach/caml-mimic
    """
    icd_code = "".join(icd_code.split("."))
    if is_diag:
        if icd_code.startswith("E"):
            if len(icd_code) > 4:
                icd_code = icd_code[:4] + "." + icd_code[4:]
        else:
            if len(icd_code) > 3:
                icd_code = icd_code[:3] + "." + icd_code[3:]
    else:
        icd_code = icd_code[:2] + "." + icd_code[2:]
    return icd_code


def get_snowmed_children_descriptions(
    relationships_filepath: str, description_filepath: str, destination_id: str
) -> List[str]:
    """Given a parent concept in Snowmed (destination_id), returns the textual description of all
    its children concepts (i.e. those that have a "is a" relationship with the parent concept).
    Requires the filepaths to the relationship (relationships_filepath) and
    description (description_filepath) files from a Snowmed release.
    """
    relationships = pd.read_csv(
        relationships_filepath,
        sep="\t",
        usecols=["sourceId", "destinationId", "relationshipGroup", "typeId"],
        dtype={"sourceId": str, "destinationId": str, "relationshipGroup": str, "typeId": str},
    )
    descriptions = pd.read_csv(
        description_filepath,
        sep="\t",
        usecols=["conceptId", "term"],
        dtype={"conceptId": str, "term": str},
    )

    children_ids = (
        relationships[
            (relationships["destinationId"] == destination_id)
            & (relationships["typeId"] == IS_A_CID)
        ]["sourceId"]
        .unique()
        .tolist()
    )
    children_descriptions = descriptions[descriptions["conceptId"].isin(children_ids)]["term"]

    def _postprocess(text: str) -> str:
        text = re.sub(r"(\(substance\)|\(disorder\)|,?\s*NOS)", "", text)
        return text.strip()

    children_descriptions = children_descriptions.apply(_postprocess)

    return children_descriptions.unique().tolist()
