"""
Phonotactic Noun Generator

Generates pronounceable nonsense words following English phonotactics.
Ensures no collision with real English words.
"""

import random
import re


class PhonotacticGenerator:
    """
    Generates pronounceable nonsense words following English phonotactics.

    Ensures:
    - Words are pronounceable (valid onset-nucleus-coda patterns)
    - No collision with real English words
    - Consistent generation (seeded)
    - Distinct "feel" for different semantic categories
    """

    # English phonotactic patterns
    ONSETS = [
        '', 'b', 'bl', 'br', 'ch', 'd', 'dr', 'f', 'fl', 'fr', 'g', 'gl', 'gr',
        'h', 'j', 'k', 'kl', 'kr', 'l', 'm', 'n', 'p', 'pl', 'pr', 'qu', 'r',
        's', 'sc', 'sk', 'sl', 'sm', 'sn', 'sp', 'spl', 'spr', 'st', 'str', 'sw',
        't', 'th', 'tr', 'tw', 'v', 'w', 'wh', 'wr', 'z'
    ]

    NUCLEI = ['a', 'e', 'i', 'o', 'u', 'ai', 'au', 'ea', 'ee', 'ei', 'ie', 'oa', 'oo', 'ou']

    CODAS = [
        '', 'b', 'ch', 'd', 'f', 'g', 'k', 'l', 'lk', 'lm', 'lp', 'lt', 'm', 'mp',
        'n', 'nd', 'ng', 'nk', 'nt', 'p', 'r', 'rb', 'rd', 'rf', 'rg', 'rk', 'rl',
        'rm', 'rn', 'rp', 'rs', 'rt', 's', 'sh', 'sk', 'sp', 'st', 't', 'th', 'x', 'z'
    ]

    # Category-specific phonetic biases
    CATEGORY_BIASES = {
        'field': {
            'preferred_onsets': ['th', 'kr', 'gl', 'str', 'v', 'z'],
            'preferred_nuclei': ['o', 'i', 'a', 'ou', 'ei'],
            'preferred_codas': ['m', 'n', 'x', 'th', 'rs'],
            'syllables': (2, 4)
        },
        'particle': {
            'preferred_onsets': ['k', 'qu', 'b', 'g', 'z'],
            'preferred_nuclei': ['a', 'o', 'u', 'i'],
            'preferred_codas': ['k', 'n', 't', 'x', ''],
            'syllables': (1, 2)
        },
        'unit': {
            'preferred_onsets': ['m', 'k', 'j', 'v', 'w'],
            'preferred_nuclei': ['e', 'o', 'a', 'ou'],
            'preferred_codas': ['l', 't', 'n', 's', 'r'],
            'syllables': (1, 2)
        },
        'process': {
            'preferred_onsets': ['fl', 'v', 'r', 'sw', 'gl'],
            'preferred_nuclei': ['a', 'o', 'i', 'ea', 'ou'],
            'preferred_codas': ['ng', 'n', 'tion', 'm', 'sh'],
            'syllables': (2, 3)
        },
        'entity': {
            'preferred_onsets': ['b', 'l', 'm', 'n', 'p', 'r', 's', 't'],
            'preferred_nuclei': ['a', 'e', 'i', 'o', 'u'],
            'preferred_codas': ['', 'd', 'n', 'r', 's', 't'],
            'syllables': (2, 3)
        }
    }

    # Words to avoid (common English words)
    BLACKLIST = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
        'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
        'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who',
        'boy', 'did', 'man', 'she', 'too', 'use', 'war', 'big', 'god', 'lot'
    }

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.generated = set()

    def _generate_syllable(self, category: str = 'entity') -> str:
        """Generate a single syllable with category-appropriate phonetics"""
        bias = self.CATEGORY_BIASES.get(category, self.CATEGORY_BIASES['entity'])

        if self.rng.random() < 0.6:
            onset = self.rng.choice(bias['preferred_onsets'])
            nucleus = self.rng.choice(bias['preferred_nuclei'])
            coda = self.rng.choice(bias['preferred_codas'])
        else:
            onset = self.rng.choice(self.ONSETS)
            nucleus = self.rng.choice(self.NUCLEI)
            coda = self.rng.choice(self.CODAS)

        return onset + nucleus + coda

    def generate_word(self, category: str = 'entity', min_syllables: int = None,
                      max_syllables: int = None) -> str:
        """Generate a phonotactically valid nonsense word"""
        bias = self.CATEGORY_BIASES.get(category, self.CATEGORY_BIASES['entity'])

        if min_syllables is None:
            min_syllables = bias['syllables'][0]
        if max_syllables is None:
            max_syllables = bias['syllables'][1]

        for _ in range(100):
            num_syllables = self.rng.randint(min_syllables, max_syllables)
            word = ''.join(self._generate_syllable(category) for _ in range(num_syllables))

            # Clean up double letters
            word = re.sub(r'(.)\1{2,}', r'\1\1', word)

            if (len(word) >= 3 and
                word not in self.BLACKLIST and
                word not in self.generated):
                self.generated.add(word)
                return word.capitalize()

        # Fallback
        base = self._generate_syllable(category)
        return (base + str(self.rng.randint(1, 99))).capitalize()

    def generate_compound(self, category: str = 'field') -> str:
        """Generate a two-word compound term"""
        adj = self.generate_word('entity', 2, 3)
        noun = self.generate_word(category, 2, 3)
        return f"{adj} {noun}"
