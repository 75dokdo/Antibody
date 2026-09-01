"""Shared helpers for the GPC3 epitope work.

The single most important thing in this module is the numbering offset.

The structure files in ``data/structures`` are numbered by the MATURE protein
(Gln1 of the PDB is Gln25 of UniProt P51654 -- the 24-residue signal peptide is
absent).  Every published epitope and every patent we compared against uses
FULL-LENGTH UniProt numbering.  Comparing the two directly is off by 24
residues, which is enough to move an epitope call from one region to another.

    uniprot_number = mature_number + 24
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

from Bio.PDB import PDBParser, ShrakeRupley
from Bio.SeqUtils import seq1

warnings.filterwarnings("ignore")

#: UniProt P51654 numbering minus mature-chain numbering.
UNIPROT_OFFSET = 24

REPO_ROOT = Path(__file__).resolve().parent.parent
STRUCTURE_DIR = REPO_ROOT / "data" / "structures"
PATENT_DB = REPO_ROOT / "data" / "patents" / "gpc3_patent_epitopes.json"

#: Relative solvent accessibility above which we call a residue surface-exposed.
#: 0.20 is the conventional cut-off for "accessible enough to be part of a
#: conformational B-cell epitope".
RSA_EXPOSED_CUTOFF = 0.20

#: Tien et al. (2013) theoretical maximum solvent accessibility, in A^2.
MAX_ASA = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "E": 223.0, "Q": 225.0, "G": 104.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}


def to_uniprot(mature: int) -> int:
    """Mature-chain residue number -> UniProt P51654 residue number."""
    return mature + UNIPROT_OFFSET


def to_mature(uniprot: int) -> int:
    """UniProt P51654 residue number -> mature-chain residue number."""
    return uniprot - UNIPROT_OFFSET


@dataclass
class Residue:
    mature: int
    uniprot: int
    aa: str
    plddt: float
    sasa: float = 0.0
    rsa: float = 0.0

    @property
    def exposed(self) -> bool:
        return self.rsa >= RSA_EXPOSED_CUTOFF


def load_residues(pdb_path: Path, with_sasa: bool = False) -> list[Residue]:
    """Read a GPC3 structure file into a list of :class:`Residue`.

    ``B-factor`` in these AlphaFold-style models carries pLDDT, so we keep it
    as a per-residue model-confidence score.
    """
    structure = PDBParser(QUIET=True).get_structure("gpc3", str(pdb_path))
    model = structure[0]

    if with_sasa:
        # SASA must be computed on whatever structure is passed in.  For a
        # meaningful burial call this has to be the FULL ectodomain -- a
        # residue that is buried in the intact protein looks exposed when you
        # slice its region out into its own file.
        ShrakeRupley().compute(model, level="R")

    residues: list[Residue] = []
    for res in model["A"]:
        if res.id[0] != " ":
            continue
        aa = seq1(res.get_resname())
        atoms = list(res.get_atoms())
        plddt = sum(a.get_bfactor() for a in atoms) / len(atoms)
        entry = Residue(
            mature=res.id[1],
            uniprot=to_uniprot(res.id[1]),
            aa=aa,
            plddt=plddt,
        )
        if with_sasa:
            entry.sasa = res.sasa
            entry.rsa = res.sasa / MAX_ASA.get(aa, 200.0)
        residues.append(entry)
    return residues


def sequence_of(residues: list[Residue]) -> str:
    return "".join(r.aa for r in residues)


def expand(start: int, end: int) -> set[int]:
    return set(range(start, end + 1))


def epitope_uniprot_positions(epitope: dict) -> set[int]:
    """Every UniProt position an epitope entry lays claim to."""
    positions: set[int] = set()
    span = epitope.get("uniprot_residues")
    if span:
        positions |= expand(span["start"], span["end"])
    positions |= set(epitope.get("additional_uniprot_residues", []))
    return positions
