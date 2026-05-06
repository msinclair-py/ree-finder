"""Per-ion chemistry tables used during ion placement and QM setup.

The tables are keyed by the Amber/PDB ion code (uppercase, e.g. ``'CA'``,
``'ND'``):

- ``BINDING_ATOMS``: residue → list of side-chain atom names that can
  coordinate the given ion. Used by :func:`apps.place_ions` to pick which
  atoms contribute to the coordination-shell center-of-mass.
- ``USES_BACKBONE_O``: ions for which backbone carbonyl oxygens are also
  treated as candidate coordinators (hard- or borderline-Lewis-acid metals
  that bind backbone carbonyls in known structures).
- ``MIN_COORD``: minimum number of coordinating atoms required before an
  ion is actually placed; clusters below this threshold are dropped.
- ``ION_CHARGE`` / ``ION_SPIN``: formal charge and ground-state spin
  multiplicity guess written into the cluster JSON for downstream DFT
  (``ION_SPIN[ion] - 1`` is what PySCF wants for ``spin``).
"""
BINDING_ATOMS = {
    'CA': {
        'ARG': ['NH1', 'NH2'],
        'ASN': ['OD1'],
        'ASP': ['OD1', 'OD2'],
        'GLN': ['OE1'],
        'GLU': ['OE1', 'OE2'],
        'LYS': ['NZ'],
        'SER': ['OG'],
        'THR': ['OG1'],
        'TYR': ['OH'],
    },
    'CO': {
        'ASN': ['OD1'],
        'ASP': ['OD1', 'OD2'],
        'CYS': ['SG'],
        'GLN': ['OE1'],
        'GLU': ['OE1', 'OE2'],
        'HIS': ['ND1', 'NE2'],
        'MET': ['SD'],
        'SER': ['OG'],
        'THR': ['OG1'],
        'TYR': ['OH'],
    },
    'CU': {
        'ASN': ['OD1'],
        'ASP': ['OD1', 'OD2'],
        'CYS': ['SG'],
        'GLN': ['OE1'],
        'GLU': ['OE1', 'OE2'],
        'HIS': ['ND1', 'NE2'],
        'MET': ['SD'],
        'SER': ['OG'],
        'THR': ['OG1'],
        'TYR': ['OH'],
    },
    'FE': {
        'ASN': ['OD1'],
        'ASP': ['OD1', 'OD2'],
        'CYS': ['SG'],
        'GLN': ['OE1'],
        'GLU': ['OE1', 'OE2'],
        'HIS': ['ND1', 'NE2'],
        'MET': ['SD'],
        'SER': ['OG'],
        'THR': ['OG1'],
        'TYR': ['OH'],
    },
    'MG': {
        'ARG': ['NH1', 'NH2'],
        'ASN': ['OD1'],
        'ASP': ['OD1', 'OD2'],
        'GLN': ['OE1'],
        'GLU': ['OE1', 'OE2'],
        'HIS': ['ND1', 'NE2'],
        'LYS': ['NZ'],
        'SER': ['OG'],
        'THR': ['OG1'],
        'TYR': ['OH'],
    },
    'MN': {
        'ASN': ['OD1'],
        'ASP': ['OD1', 'OD2'],
        'CYS': ['SG'],
        'GLN': ['OE1'],
        'GLU': ['OE1', 'OE2'],
        'HIS': ['ND1', 'NE2'],
        'MET': ['SD'],
        'SER': ['OG'],
        'THR': ['OG1'],
        'TYR': ['OH'],
    },
    'ND': {
        'ASN': ['OD1'],
        'ASP': ['OD1', 'OD2'],
        'GLN': ['OE1'],
        'GLU': ['OE1', 'OE2'],
        'SER': ['OG'],
        'THR': ['OG1'],
        'TYR': ['OH'],
    },
    'ZN': {
        'ASN': ['OD1'],
        'ASP': ['OD1', 'OD2'],
        'CYS': ['SG'],
        'GLN': ['OE1'],
        'GLU': ['OE1', 'OE2'],
        'HIS': ['ND1', 'NE2'],
        'MET': ['SD'],
        'SER': ['OG'],
        'THR': ['OG1'],
        'TYR': ['OH'],
    },
}

USES_BACKBONE_O = {'CA', 'MG', 'ND'}

MIN_COORD = {
    'CA': 5,
    'CO': 4,
    'CU': 3,
    'FE': 4,
    'MG': 4,
    'MN': 4,
    'ND': 6,
    'ZN': 3,
}

ION_CHARGE = {'ZN': 2, 'MN': 2, 'FE': 3, 'CA': 2, 'MG': 2, 'CU': 2, 'CO': 2, 'ND': 3}
ION_SPIN = {'ZN': 1, 'MN': 6, 'FE': 6, 'CA': 1, 'MG': 1, 'CU': 2, 'CO': 4, 'ND': 4}
