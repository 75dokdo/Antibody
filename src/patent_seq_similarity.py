"""BLAST-based sequence-similarity check against patent-claimed GPC3 epitopes.

The region scan in ``epitope_fto_scan.py`` answers "do my residue numbers
collide with a claimed epitope".  This answers the other half: "does my
sequence LOOK like a claimed epitope", which catches two cases the coordinate
scan misses --

  1. a stretch elsewhere in the antigen that happens to mimic a claimed
     epitope, so an existing antibody could cross-react into your region;
  2. a designed or ordered peptide/CDR handed in as raw sequence, with no
     residue numbering to compare at all.

Build the reference database, then query it:

    python3 src/patent_seq_similarity.py build
    python3 src/patent_seq_similarity.py query --seq PKDNEISTFHNL
    python3 src/patent_seq_similarity.py query --fasta my_epitopes.fasta
    python3 src/patent_seq_similarity.py selfscan --window 15 --step 5

``selfscan`` walks the whole GPC3 ectodomain in sliding windows and reports
every window that resembles a claimed epitope -- the fastest way to see the
sequence-level patent landscape of the antigen in one pass.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from gpc3_lib import PATENT_DB, REPO_ROOT, STRUCTURE_DIR, load_residues, to_mature

BLAST_DIR = REPO_ROOT / "data" / "patents" / "blastdb"
DB_FASTA = BLAST_DIR / "gpc3_patent_epitopes.fasta"
DB_NAME = BLAST_DIR / "gpc3_patent_epitopes"

# Short peptides need a permissive setup or blastp silently returns nothing:
# PAM30 + a large word-size-independent gap cost is the NCBI-recommended
# "blastp-short" regime.
SHORT_QUERY_OPTS = [
    "-task", "blastp-short",
    "-matrix", "PAM30",
    "-word_size", "2",
    "-gapopen", "9",
    "-gapextend", "1",
    "-evalue", "20000",
    "-max_target_seqs", "50",
]

OUTFMT = "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore"


def _require_blast() -> None:
    for tool in ("makeblastdb", "blastp"):
        if subprocess.run(["which", tool], capture_output=True).returncode != 0:
            sys.exit(f"error: {tool} not found. Install with: apt-get install -y ncbi-blast+")


def build_db() -> None:
    """Carve each claimed epitope out of the antigen sequence into a BLAST DB."""
    _require_blast()
    db = json.loads(PATENT_DB.read_text())
    residues = load_residues(STRUCTURE_DIR / "GPC3_ectodomain_full.pdb")
    by_uniprot = {r.uniprot: r.aa for r in residues}

    BLAST_DIR.mkdir(parents=True, exist_ok=True)
    records: list[str] = []

    for ep in db["epitopes"]:
        span = ep.get("uniprot_residues")
        if not span:
            continue
        # A whole-lobe entry is not a sequence motif; indexing it would make
        # every query match it and drown out the real signals.
        if span["end"] - span["start"] > 120:
            continue
        seq = "".join(
            by_uniprot.get(i, "X") for i in range(span["start"], span["end"] + 1)
        ).replace("X", "")
        if not seq:
            continue
        header = (f"{ep['id']}|{ep['antibody']}|uniprot_{span['start']}-{span['end']}"
                  f"|evidence_{ep.get('evidence')}|{';'.join(ep.get('patents', []))[:80]}")
        records.append(f">{header.replace(' ', '_')}\n{seq}")

        core = ep.get("core_uniprot_residues")
        if core:
            core_seq = "".join(
                by_uniprot.get(i, "") for i in range(core["start"], core["end"] + 1)
            )
            if core_seq:
                records.append(
                    f">{ep['id']}_CORE|{ep['antibody']}|uniprot_{core['start']}-{core['end']}"
                    f"|evidence_{ep.get('evidence')}|CORE_EPITOPE".replace(" ", "_")
                    + f"\n{core_seq}"
                )

    DB_FASTA.write_text("\n".join(records) + "\n")
    subprocess.run(
        ["makeblastdb", "-in", str(DB_FASTA), "-dbtype", "prot",
         "-out", str(DB_NAME), "-title", "GPC3_patent_epitopes"],
        check=True, capture_output=True,
    )
    print(f"built BLAST database: {DB_NAME}")
    print(f"  {len(records)} claimed-epitope sequences indexed")
    for rec in records:
        header, seq = rec.split("\n")
        print(f"    {header[1:60]:<60} {len(seq):>3} aa")


def run_blast(query_fasta: Path) -> list[dict]:
    _require_blast()
    if not Path(str(DB_NAME) + ".phr").exists():
        sys.exit("error: BLAST database not built. Run: python3 src/patent_seq_similarity.py build")

    proc = subprocess.run(
        ["blastp", "-query", str(query_fasta), "-db", str(DB_NAME),
         "-outfmt", OUTFMT, *SHORT_QUERY_OPTS],
        capture_output=True, text=True, check=True,
    )
    hits = []
    for line in proc.stdout.strip().splitlines():
        if not line:
            continue
        f = line.split("\t")
        hits.append({
            "query": f[0], "subject": f[1], "pident": float(f[2]),
            "length": int(f[3]), "qstart": int(f[6]), "qend": int(f[7]),
            "sstart": int(f[8]), "send": int(f[9]),
            "evalue": float(f[10]), "bitscore": float(f[11]),
        })
    return hits


def classify(hit: dict) -> str:
    """Turn an alignment into a similarity call."""
    ident_res = hit["pident"] / 100 * hit["length"]
    if hit["pident"] >= 90 and hit["length"] >= 8:
        return "CRITICAL"
    if hit["pident"] >= 70 and ident_res >= 7:
        return "HIGH"
    if hit["pident"] >= 50 and ident_res >= 5:
        return "MODERATE"
    return "LOW"


def report(hits: list[dict], show_low: bool = False) -> None:
    if not hits:
        print("  no similarity to any indexed claimed epitope")
        return
    order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3}
    rows = [(classify(h), h) for h in hits]
    if not show_low:
        rows = [r for r in rows if r[0] != "LOW"]
    if not rows:
        print("  only LOW-similarity hits (use --show-low to list them)")
        return
    rows.sort(key=lambda r: (order[r[0]], -r[1]["bitscore"]))

    print(f"  {'call':<10} {'query':<22} {'claimed epitope':<34} {'ident':<8} {'len':<5} {'E'}")
    print(f"  {'-'*10} {'-'*22} {'-'*34} {'-'*8} {'-'*5} {'-'*9}")
    for call, h in rows:
        subject = h["subject"].split("|")
        label = f"{subject[0]} {subject[2] if len(subject) > 2 else ''}"
        print(f"  {call:<10} {h['query'][:22]:<22} {label[:34]:<34} "
              f"{h['pident']:>5.1f}%  {h['length']:<5} {h['evalue']:.1e}")


def cmd_query(args: argparse.Namespace) -> None:
    if args.seq:
        tmp = Path(tempfile.mkstemp(suffix=".fasta")[1])
        tmp.write_text(f">query\n{args.seq.strip().upper()}\n")
        query = tmp
    elif args.fasta:
        query = Path(args.fasta)
    else:
        sys.exit("error: provide --seq or --fasta")

    print(f"query: {args.seq if args.seq else query}")
    report(run_blast(query), show_low=args.show_low)


def cmd_selfscan(args: argparse.Namespace) -> None:
    """Slide a window along the antigen and flag patent-similar stretches."""
    residues = load_residues(STRUCTURE_DIR / "GPC3_ectodomain_full.pdb")
    seq = "".join(r.aa for r in residues)

    lines = []
    for start in range(0, len(seq) - args.window + 1, args.step):
        window = seq[start:start + args.window]
        mature_start = residues[start].mature
        lines.append(f">m{mature_start}_u{residues[start].uniprot}\n{window}")

    tmp = Path(tempfile.mkstemp(suffix=".fasta")[1])
    tmp.write_text("\n".join(lines) + "\n")

    print(f"self-scan: {len(lines)} windows of {args.window} aa, step {args.step}")
    print("window ids are m<mature>_u<uniprot> = the window's first residue\n")
    report(run_blast(tmp), show_low=args.show_low)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build", help="build the claimed-epitope BLAST database")

    q = sub.add_parser("query", help="check one sequence or FASTA file")
    q.add_argument("--seq", help="raw amino-acid sequence")
    q.add_argument("--fasta", help="FASTA file of sequences to check")
    q.add_argument("--show-low", action="store_true")
    q.set_defaults(func=cmd_query)

    s = sub.add_parser("selfscan", help="sliding-window scan of the whole antigen")
    s.add_argument("--window", type=int, default=15)
    s.add_argument("--step", type=int, default=5)
    s.add_argument("--show-low", action="store_true")
    s.set_defaults(func=cmd_selfscan)

    args = parser.parse_args()
    if args.cmd == "build":
        build_db()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
