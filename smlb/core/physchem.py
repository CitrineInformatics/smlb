"""Scientific data.

Scientific Machine Learning Benchmark: 
A benchmark of regression models in chem- and materials informatics.
(c) Matthias Rupp 2019, Citrine Informatics.

Chemical, physics and materials data.
"""

from smlb import params

_element_data = {
    1: {"Z": 1, "abbreviation": "H"},
    2: {"Z": 2, "abbreviation": "He"},
    3: {"Z": 3, "abbreviation": "Li"},
    4: {"Z": 4, "abbreviation": "Be"},
    5: {"Z": 5, "abbreviation": "B"},
    6: {"Z": 6, "abbreviation": "C"},
    7: {"Z": 7, "abbreviation": "N"},
    8: {"Z": 8, "abbreviation": "O"},
    9: {"Z": 9, "abbreviation": "F"},
    10: {"Z": 10, "abbreviation": "Ne"},
    11: {"Z": 11, "abbreviation": "Na"},
    12: {"Z": 12, "abbreviation": "Mg"},
    13: {"Z": 13, "abbreviation": "Al"},
    14: {"Z": 14, "abbreviation": "Si"},
    15: {"Z": 15, "abbreviation": "P"},
    16: {"Z": 16, "abbreviation": "S"},
    17: {"Z": 17, "abbreviation": "Cl"},
    18: {"Z": 18, "abbreviation": "Ar"},
    19: {"Z": 19, "abbreviation": "K"},
    20: {"Z": 20, "abbreviation": "Ca"},
    21: {"Z": 21, "abbreviation": "Sc"},
    22: {"Z": 22, "abbreviation": "Ti"},
    23: {"Z": 23, "abbreviation": "V"},
    24: {"Z": 24, "abbreviation": "Cr"},
    25: {"Z": 25, "abbreviation": "Mn"},
    26: {"Z": 26, "abbreviation": "Fe"},
    27: {"Z": 27, "abbreviation": "Co"},
    28: {"Z": 28, "abbreviation": "Ni"},
    29: {"Z": 29, "abbreviation": "Cu"},
    30: {"Z": 30, "abbreviation": "Zn"},
    31: {"Z": 31, "abbreviation": "Ga"},
    32: {"Z": 32, "abbreviation": "Ge"},
    33: {"Z": 33, "abbreviation": "As"},
    34: {"Z": 34, "abbreviation": "Se"},
    35: {"Z": 35, "abbreviation": "Br"},
    36: {"Z": 36, "abbreviation": "Kr"},
    37: {"Z": 37, "abbreviation": "Rb"},
    38: {"Z": 38, "abbreviation": "Sr"},
    39: {"Z": 39, "abbreviation": "Y"},
    40: {"Z": 40, "abbreviation": "Zr"},
    41: {"Z": 41, "abbreviation": "Nb"},
    42: {"Z": 42, "abbreviation": "Mo"},
    43: {"Z": 43, "abbreviation": "Tc"},
    44: {"Z": 44, "abbreviation": "Ru"},
    45: {"Z": 45, "abbreviation": "Rh"},
    46: {"Z": 46, "abbreviation": "Pd"},
    47: {"Z": 47, "abbreviation": "Ag"},
    48: {"Z": 48, "abbreviation": "Cd"},
    49: {"Z": 49, "abbreviation": "In"},
    50: {"Z": 50, "abbreviation": "Sn"},
    51: {"Z": 51, "abbreviation": "Sb"},
    52: {"Z": 52, "abbreviation": "Te"},
    53: {"Z": 53, "abbreviation": "I"},
    54: {"Z": 54, "abbreviation": "Xe"},
    55: {"Z": 55, "abbreviation": "Cs"},
    56: {"Z": 56, "abbreviation": "Ba"},
    57: {"Z": 57, "abbreviation": "La"},
    58: {"Z": 58, "abbreviation": "Ce"},
    59: {"Z": 59, "abbreviation": "Pr"},
    60: {"Z": 60, "abbreviation": "Nd"},
    61: {"Z": 61, "abbreviation": "Pm"},
    62: {"Z": 62, "abbreviation": "Sm"},
    63: {"Z": 63, "abbreviation": "Eu"},
    64: {"Z": 64, "abbreviation": "Gd"},
    65: {"Z": 65, "abbreviation": "Tb"},
    66: {"Z": 66, "abbreviation": "Dy"},
    67: {"Z": 67, "abbreviation": "Ho"},
    68: {"Z": 68, "abbreviation": "Er"},
    69: {"Z": 69, "abbreviation": "Tm"},
    70: {"Z": 70, "abbreviation": "Yb"},
    71: {"Z": 71, "abbreviation": "Lu"},
    72: {"Z": 72, "abbreviation": "Hf"},
    73: {"Z": 73, "abbreviation": "Ta"},
    74: {"Z": 74, "abbreviation": "W"},
    75: {"Z": 75, "abbreviation": "Re"},
    76: {"Z": 76, "abbreviation": "Os"},
    77: {"Z": 77, "abbreviation": "Ir"},
    78: {"Z": 78, "abbreviation": "Pt"},
    79: {"Z": 79, "abbreviation": "Au"},
    80: {"Z": 80, "abbreviation": "Hg"},
    81: {"Z": 81, "abbreviation": "Tl"},
    82: {"Z": 82, "abbreviation": "Pb"},
    83: {"Z": 83, "abbreviation": "Bi"},
    84: {"Z": 84, "abbreviation": "Po"},
    85: {"Z": 85, "abbreviation": "At"},
    86: {"Z": 86, "abbreviation": "Rn"},
    87: {"Z": 87, "abbreviation": "Fr"},
    88: {"Z": 88, "abbreviation": "Ra"},
    89: {"Z": 89, "abbreviation": "Ac"},
    90: {"Z": 90, "abbreviation": "Th"},
    91: {"Z": 91, "abbreviation": "Pa"},
    92: {"Z": 92, "abbreviation": "U"},
    93: {"Z": 93, "abbreviation": "Np"},
    94: {"Z": 94, "abbreviation": "Pu"},
    95: {"Z": 95, "abbreviation": "Am"},
    96: {"Z": 96, "abbreviation": "Cm"},
    97: {"Z": 97, "abbreviation": "Bk"},
    98: {"Z": 98, "abbreviation": "Cf"},
    99: {"Z": 99, "abbreviation": "Es"},
    100: {"Z": 100, "abbreviation": "Fm"},
    101: {"Z": 101, "abbreviation": "Md"},
    102: {"Z": 102, "abbreviation": "No"},
    103: {"Z": 103, "abbreviation": "Lr"},
    104: {"Z": 104, "abbreviation": "Rf"},
    105: {"Z": 105, "abbreviation": "Db"},
    106: {"Z": 106, "abbreviation": "Sg"},
    107: {"Z": 107, "abbreviation": "Bh"},
    108: {"Z": 108, "abbreviation": "Hs"},
    109: {"Z": 109, "abbreviation": "Mt"},
    110: {"Z": 110, "abbreviation": "Ds"},
    111: {"Z": 111, "abbreviation": "Rg"},
    112: {"Z": 112, "abbreviation": "Cn"},
    113: {"Z": 113, "abbreviation": "Nh"},
    114: {"Z": 114, "abbreviation": "Fl"},
    115: {"Z": 115, "abbreviation": "Mc"},
    116: {"Z": 116, "abbreviation": "Lv"},
    117: {"Z": 117, "abbreviation": "Ts"},
    118: {"Z": 118, "abbreviation": "Og"},
}


def element_data(element, property_):
    """Query chemical element data.

    Parameters:
        element: chemical element, given by either proton number (int) or abbreviation (str)
        property_: queried property; one of 'abbreviation', 'Z' (proton number)

    Returns:
        queried property

    Raises:
        InvalidParameterError: for invalid parameters
    """

    element = params.chemical_element(element)
    property_ = params.enumeration(property_, {"Z", "abbreviation"})
    return _element_data[element][property_]
