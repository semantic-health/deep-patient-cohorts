import pandas as pd
import os
from icd9 import ICD9
import logging

logger = logging.getLogger(__name__)

def rollup(code, level, icd_tree):
	"""
	Rolls up the ICD code to a level of hierarchy
	"""
	icd_str = str(code)
	if "." not in icd_str:
		if icd_str.startswith("E") and len(icd_str) > 4:
			code = icd_str[:4] + "." + icd_str[4:]
		elif len(icd_str) > 3:
			code = icd_str[:3] + "." + icd_str[3:]
	
	# find the code
	code_node = icd_tree.find(code)
	if code_node is None:
		# logger.info(f"COULD NOT FIND CODE : {code}")
		code_node = icd_tree.find(code[:-1])
		if code_node is None:
			# logger.info(f"Could not find parent code {code[:-1]}")
			code_node = icd_tree.find(code[:-2])
			if code_node is None:
				# logger.info(f"Could not find parent code {code[:-2]}")
				code_node = icd_tree.find(code[:-3])
				if code_node is None:
					logger.info(f"Could not even find a grand-grand-parent code {code[:-3]} of code {code} returning OTHER")
				return "OTHER"

	# Finds the parents for the code
	levels = code_node.parents

	# If the level is deeper than the original code return none
	if level + 1 > len(levels) - 1:
		return "NONE"

	# Selects the hierarchy level and returns the ICD code
	else:
		# + 1 is to avoid "ROOT"
		return(levels[level + 1].code)

def rollup_all_levels(code, icd_tree):
	"""
	Bad code that needs fixing
	"""
	levels = ["ICD_1", "ICD_2", "ICD_3", "ICD_4"] #, "ICD_5", "ICD_6", "ICD_7"
	output = []
	for i, level in enumerate(levels):
		output.append(rollup(code, i, icd_tree))
	return output

def rollup_multiple_codes(codes, icd_tree):
	"""
	Codes is a list of codes
	"""
	codes = str.split(codes, sep=",")
	# Make the storage
	levels = ["ICD_1", "ICD_2", "ICD_3", "ICD_4"] #, "ICD_5", "ICD_6", "ICD_7"
	output = {}

	# Make these into sets
	for level in levels:
		output[level] = set()
	
	# add the rolled up versions
	for code in codes:
		rolled_up_code = rollup_all_levels(code, icd_tree)
		for level, code in zip(levels, rolled_up_code):
			output[level].add(code)

	# convert back to a list
	for level in levels:
		output[level] = list(output[level])
	
	return output

def rollup_mimic(mimic_path, mimic_filename, codes_path):
	"""
	Rolls up all the codes in mimic data from our preprocessing
	"""
	mimic_data = pd.read_csv(os.path.join(mimic_path, mimic_filename))
	tree = ICD9(codes_path)
	mimic_data["DIAG_ICD_ALL"] = mimic_data.ICD_DIAG.apply(rollup_multiple_codes, icd_tree = tree)
	mimic_data = mimic_data.assign(**mimic_data.DIAG_ICD_ALL.apply(pd.Series))
	mimic_data.to_csv(os.path.join(mimic_path, "mimic_processed.csv"))

if __name__ == "__main__":
	import pdb; pdb.set_trace()
	tree = ICD9("codes.json")
	# mimic_rollup_test = pd.read_csv("mimic_rollup_test.csv")
	# mimic_rollup_test["ICD_3"] = mimic_rollup_test.ICD9_CODE.apply(rollup, level = 3, icd_tree = tree)
	# mimic_rollup_test["ALL_ICD"] = mimic_rollup_test.ICD9_CODE.apply(rollup_all_levels, icd_tree = tree)
	# mimic_rollup_test.assign(**mimic_rollup_test.ALL_ICD.apply(pd.Series))
	# for i, row in mimic_rollup_test.iterrows():
		# row.ICD_rolled_up = rollup(code = str(row.ICD9_CODE), level = 1, icd_tree = tree)
	rollup_mimic("/Users/michalmalyska/Desktop/Projects/semantic_health_public/deep-patient-cohorts/data", "mimic_real_test.csv", "/Users/michalmalyska/Desktop/Projects/semantic_health_public/deep-patient-cohorts/scripts/icd_rollup/codes.json")
