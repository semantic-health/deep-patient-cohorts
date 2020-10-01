import re
from typing import List, Union


class AgeClassifier:
    age_pattern = re.compile(
        r"(?:\[\**age\s*over\s*)(90)\s*\**]|(\d+)\s*(?:year\s*old|y.\s*o.|yo|year\s*old|year-old|-year-old|-year old)",
        re.IGNORECASE,
    )

    def __call__(self, texts: Union[str, List[str]]) -> Union[int, List[int]]:
        """Returns the extracted ages for each text in `texts` or -1 if no age was found."""
        if isinstance(texts, str):
            texts = [texts]

        matches = [self.age_pattern.findall(text) for text in texts]
        # The age of the patient tends to be the first age mentioned in the discharge summary.
        # There are two capturing groups, so we need to check which is the non-empty one.
        # -1 denotes a failed attempt to capture an age from the text.
        ages = [
            -1 if not match else int(match[0][0]) if match[0][0] else int(match[0][1])
            for match in matches
        ]
        return ages[0] if len(ages) == 1 else ages
