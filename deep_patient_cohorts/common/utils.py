POSITIVE = 1
NEGATIVE = -1
ABSTAIN = 0


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
