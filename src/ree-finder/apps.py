"""Per-stage compute kernels for the REE binding-site pipeline.

Each top-level function here corresponds to one Parsl stage in
:mod:`pipeline`: structure prediction (Boltz), ESM-2 / ESM-IF embeddings,
ESMBind ensemble inference, ion placement and Amber system construction,
MD-driven relaxation and cluster extraction, and DFT geometry optimization.
Functions are written to be self-contained so they can run on remote Parsl
workers.
"""
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import esm
import numpy as np

# Import torch before gpu4pyscf so the system CUDA stack gets loaded first;
# otherwise gpu4pyscf's bundled cuda-12.8 nvjitlink shadows it and breaks
# system torch's cusparse.
import torch
from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from boltz.main import predict
from dataset import MultimodalDataset
from esm import inverse_folding
from model import ESMBindMultiModal
from molecular_simulations.simulate import Simulator
from pyscf import gto, lib
from pyscf.geomopt.geometric_solver import optimize
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, EsmModel

try:
    import intel_extension_for_pytorch  # noqa: F401
except ImportError:
    pass

try:
    from gpu4pyscf import dft
except ImportError:
    from pyscf import dft

from defaults import BINDING_ATOMS, ION_CHARGE, ION_SPIN, MIN_COORD, USES_BACKBONE_O

# Resolve project-relative paths against this file rather than the worker's
# cwd, since parsl workers don't necessarily run from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent

# Amber's atomic_ions.lib uses uppercase residue/atom names for transition
# metals but mixed case ('Nd') for the lanthanides. tleap is case-sensitive,
# so the PDB we hand it has to match.
AMBER_ION_NAME = {'CA': 'CA', 'MG': 'MG', 'FE': 'FE', 'ND': 'Nd'}

def fold(
    header: str,
    sequence: str,
    out_dir: Path,
) -> Path:
    """Predict a protein structure with Boltz from a single amino-acid sequence.

    Writes a single-sequence Boltz YAML schema to ``out_dir/{header}.yaml``,
    invokes ``boltz.main.predict`` with MSA-server lookup and structural
    potentials enabled, and returns the path to the resulting PDB.

    Args:
        header: Identifier used both as the YAML/PDB stem and as the chain
            label in the schema.
        sequence: Amino-acid sequence to fold.
        out_dir: Directory for the schema and Boltz prediction outputs;
            created if missing.

    Returns:
        Path to the predicted PDB under
        ``out_dir/boltz_results_{header}/predictions/{header}/``.
    """
    import yaml

    weights_cache = PROJECT_ROOT / 'boltz_cache'
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    schema = {
        'version': 1,
        'sequences': [
            {'protein': {'id': 'A', 'sequence': sequence}}
        ],
    }
    schema_path = out_dir / f'{header}.yaml'
    schema_path.write_text(yaml.safe_dump(schema, sort_keys=False))

    predict.callback(
        data=schema_path,
        out_dir=out_dir,
        cache=weights_cache,
        output_format='pdb',
        use_potentials=True,
        use_msa_server=True,
    )

    return next(iter(out_dir.glob(f'boltz_results_{header}/predictions/{header}/*.pdb')))

def sequence_embeddings(
    header_labels: list[str],
    sequences: list[str],
    output_dir: Path,
    device: str,
    batch_size: int = 50,
    model_id: str = 'facebook/esm2_t33_650M_UR50D',
) -> Path:
    """Compute per-residue ESM-2 sequence embeddings and min-max normalize them.

    Tokenizes sequences in batches with the HuggingFace tokenizer for
    ``model_id``, runs the encoder on ``device``, strips the ``<cls>``/``<eos>``
    tokens, and rescales each embedding to ``[0, 1]`` using the precomputed
    per-feature min/max arrays under ``esm_normalization_constants/``. One
    ``.npy`` is written per protein, keyed by the corresponding header.

    Args:
        header_labels: Identifiers used as the output filenames.
        sequences: Amino-acid sequences in the same order as ``header_labels``.
        output_dir: Directory to write per-protein ``.npy`` embeddings;
            created if missing.
        device: Torch device string for the model (e.g. ``'cuda'``, ``'cpu'``).
        batch_size: Sequences per forward pass.
        model_id: HuggingFace model id for the ESM-2 encoder.

    Returns:
        The output directory containing the ``.npy`` files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    constants = PROJECT_ROOT / 'esm_normalization_constants'
    max_repr_esm = np.load(str(constants / 'esm_repr_max.npy'))
    min_repr_esm = np.load(str(constants / 'esm_repr_min.npy'))

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = EsmModel.from_pretrained(model_id).to(device).eval()  # type: ignore[arg-type]

    with torch.no_grad():
        for j in range(0, len(sequences), batch_size):
            partial_headers = header_labels[j:j + batch_size]
            partial_sequences = sequences[j:j + batch_size]
            batch = tokenizer(
                partial_sequences,
                return_tensors='pt',
                padding=True,
            ).to(device)

            hidden = model(**batch).last_hidden_state.detach().cpu().numpy()
            lengths = batch['attention_mask'].sum(1).detach().cpu().numpy()

            for header, h, seq_len in zip(
                partial_headers, hidden, lengths, strict=True
            ):
                # tokenizer prepends <cls> and appends <eos>; strip both
                seq_emb = h[1:seq_len - 1]
                seq_emb = (seq_emb - min_repr_esm) / (max_repr_esm - min_repr_esm)
                np.save(output_dir / f'{header}.npy', seq_emb)

    return output_dir

def structural_embeddings(
    header_labels: list[str],
    pdbs: list[str],
    output_dir: Path,
    device: str,
) -> Path:
    """Compute per-residue ESM-IF (inverse-folding) structural embeddings.

    Loads chain ``'A'`` from each PDB, extracts backbone coordinates, runs
    the ESM-IF1 encoder on ``device``, and min-max normalizes each embedding
    using the precomputed constants under ``esm_normalization_constants/``.
    One ``.npy`` is written per protein, keyed by the corresponding header.

    Args:
        header_labels: Identifiers used as the output filenames.
        pdbs: PDB paths in the same order as ``header_labels``.
        output_dir: Directory to write per-protein ``.npy`` embeddings;
            created if missing.
        device: Torch device string for the model.

    Returns:
        The output directory containing the ``.npy`` files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    constants = PROJECT_ROOT / 'esm_normalization_constants'
    max_repr_esm = np.load(str(constants / 'esm_if_repr_max.npy'))
    min_repr_esm = np.load(str(constants / 'esm_if_repr_min.npy'))

    torch_device = torch.device(device)
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model.eval()
    model = model.to(torch_device)

    with torch.no_grad():
        for header, pdb in zip(header_labels, pdbs, strict=True):
            structure = inverse_folding.util.load_structure(str(pdb), 'A')
            coords, _ = inverse_folding.util.extract_coords_from_structure(structure)
            emb = inverse_folding.util.get_encoder_output(
                model, alphabet, coords
            ).detach().cpu().numpy()

            emb = (emb - min_repr_esm) / (max_repr_esm - min_repr_esm)
            np.save(output_dir / f'{header}.npy', emb)

    return output_dir

def esmbind_inference(
    dataset: MultimodalDataset,
    ensemble_path: Path,
    device: str,
) -> dict[str, dict[str, np.ndarray]]:
    """Run the 5-fold ESMBind multimodal ensemble over a dataset of proteins.

    Loads ``fold_{1..5}.pt`` from ``ensemble_path``, restoring the SWA encoder
    and classifier weights into ``ESMBindMultiModal``. For each of the seven
    trained ligands (MG, FE, CU, CO, CA, MN, ZN), runs every protein in
    ``dataset`` through all five models and averages the per-residue sigmoid
    probabilities. Per-fold F1 and MCC thresholds are averaged across folds
    and returned alongside the predictions.

    Args:
        dataset: Multimodal dataset yielding ``(id, seq_feat, struct_feat)``
            tuples; its ``collate_fn`` is used for batching.
        ensemble_path: Directory containing ``fold_1.pt`` … ``fold_5.pt``
            checkpoints with ``swa_encoder``, ``swa_classifier``,
            ``threshold_f1``, and ``threshold_mcc`` keys.
        device: Torch device string for inference.

    Returns:
        Dict with averaged thresholds under keys ``'threshold_f1'`` and
        ``'threshold_mcc'`` (each a ``ligand -> float`` map), plus one entry
        per ligand mapping ``protein_id -> per-residue probability array``.
    """
    ligand_list = ['MG', 'FE', 'CU', 'CO', 'CA', 'MN', 'ZN'] # hard-coded

    # Architecture is pinned by the trained ESMBind weights:
    # feature_dim_1 = ESM-2 (1280), feature_dim_2 = ESM-IF (512).
    model_conf = SimpleNamespace(
        feature_dim_1=1280,
        feature_dim_2=512,
        hidden_dim=128,
        hidden_dim_1=256,
        hidden_dim_2=256,
        noise_level=0.1,
        dropout=0.2,
    )

    models = []
    f1_threshold_list, mcc_threshold_list = [], []
    for i in range(1, 6):
        model = ESMBindMultiModal(model_conf)
        checkpoint = torch.load(
            ensemble_path / f'fold_{i}.pt',
            map_location='cpu',
            weights_only=False,
        )

        model.params.encoder.load_state_dict(checkpoint['swa_encoder'], strict=False)
        model.params.classifier.load_state_dict(checkpoint['swa_classifier'], strict=False)
        f1_threshold_list.append(checkpoint['threshold_f1'])
        mcc_threshold_list.append(checkpoint['threshold_mcc'])
        model.training = False
        model.eval()
        model.to(device)
        models.append(model)

    threshold_f1, threshold_mcc = {}, {}
    keys = f1_threshold_list[0].keys()

    for key in keys:
        threshold_f1[key] = sum(d[key] for d in f1_threshold_list) / len(f1_threshold_list)
        threshold_mcc[key] = sum(d[key] for d in mcc_threshold_list) / len(mcc_threshold_list)

    predictions = {}
    predictions['threshold_f1'] = threshold_f1
    predictions['threshold_mcc'] = threshold_mcc

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=8,
        collate_fn=dataset.collate_fn,
    )

    for ligand in ligand_list:
        predictions[ligand] = {}
        with torch.no_grad():
            for batch_data in dataloader:
                _id, feats_1, feats_2, masks = batch_data
                feats_1 = feats_1.to(device)
                feats_2 = feats_2.to(device)
                masks = masks.to(device)

                outputs = []
                for model in models:
                    model.ligand = ligand  # type: ignore[assignment]
                    output = model(feats_1, feats_2, ligand)
                    output = torch.sigmoid(torch.masked_select(output, masks.bool()))
                    outputs.append(output)

                stacked = torch.stack(outputs).mean(0)
                predictions[ligand][_id[0]] = stacked.detach().cpu().numpy()

    return predictions

def place_ions(
    pdb_file: Path,
    output_pdb: Path,
    binding_residues: list,
    ion: str,
    chain_id: str = 'A',
    cluster_threshold: float = 7.,
) -> dict:
    """Place metal ions at the centers-of-mass of predicted coordination shells.

    Collects ion-coordinating side-chain atoms (and backbone O for ions in
    ``USES_BACKBONE_O``) from the residues flagged by the binding classifier,
    clusters their per-residue centers-of-mass agglomeratively by spatial
    proximity, and writes one ion per cluster that meets the minimum
    coordination number in ``MIN_COORD``. The written ion atom name follows
    Amber's ``atomic_ions.lib`` casing (e.g. ``Nd`` for lanthanides).

    Args:
        pdb_file: Input protein PDB.
        output_pdb: Path to write the PDB with placed ions appended as
            ``HETATM`` residues; parent directory is created if missing.
        binding_residues: Residue numbers (1-indexed) flagged by the binding
            classifier.
        ion: Ion code (e.g. ``'CA'``, ``'ND'``); keys into ``BINDING_ATOMS``,
            ``MIN_COORD``, and ``USES_BACKBONE_O``.
        chain_id: Chain to operate on.
        cluster_threshold: Agglomerative-clustering distance threshold (Å)
            for grouping coordinating atoms into a single ion site.

    Returns:
        Mapping of newly assigned ion residue id to the sorted list of
        protein residue ids that coordinate it. Empty if no cluster met the
        minimum coordination requirement.
    """
    spec = BINDING_ATOMS[ion]
    backbone_o = ion in USES_BACKBONE_O

    structure = PDBParser(QUIET=True).get_structure('p', pdb_file)
    chain = structure[0][chain_id]

    pred_set = set(binding_residues)
    placements = []
    for residue in chain:
        if residue.id[1] not in pred_set:
            continue

        names = spec.get(residue.resname, [])
        atoms = [a for a in residue if a.name in names]

        if backbone_o:
            atoms += [a for a in residue if a.name == 'O']

        if not atoms:
            continue

        coords = np.array([a.coord for a in atoms])
        masses = np.array([a.mass for a in atoms])
        com = (coords * masses[:, None]).sum(0) / masses.sum()
        placements.append((com, residue, atoms))

    if not placements:
        return {}

    positions = np.array([p[0] for p in placements])
    if len(placements) == 1:
        labels = np.array([0])
    else:
        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=cluster_threshold,
        ).fit(positions).labels_

    ion_to_group = {}
    next_resid = max((r.id[1] for r in chain), default=0) + 1
    for label in sorted(set(labels)):
        idxs = np.where(labels == label)[0]
        cluster_atoms = [a for i in idxs for a in placements[i][2]]

        if len(cluster_atoms) < MIN_COORD[ion]:
            continue

        coords = np.array([a.coord for a in cluster_atoms])
        masses = np.array([a.mass for a in cluster_atoms])
        ion_pos = (coords * masses[:, None]).sum(0) / masses.sum()

        amber_name = AMBER_ION_NAME[ion]
        new_res = Residue((f'H_{amber_name}', next_resid, ' '), amber_name, ' ')
        new_res.add(
            Atom(
                name=amber_name.rjust(4),
                coord=ion_pos,
                bfactor=0.,
                occupancy=1.,
                altloc=' ',
                fullname=amber_name.rjust(4),
                serial_number=next_resid,
                element=amber_name.upper(),
            )
        )

        chain.add(new_res)
        ion_to_group[next_resid] = sorted({placements[i][1].id[1] for i in idxs})
        next_resid += 1

    output_pdb.parent.mkdir(exist_ok=True, parents=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_pdb))

    return ion_to_group


def build_amber_system(
    metal_pdb: Path,
    out: Path,
    build: Path,
    ion: str,
    amberhome: Path,
    padding: float,
    salt_conc_M: float,
) -> tuple[Path, Path, Path]:
    """Build a solvated, neutralized Amber system around a protein-with-metal PDB.

    Generates a ``tleap`` script that loads the ff19SB protein force field and
    the OPC water model, sizes a cubic box from the ligated structure's spatial
    extent plus ``padding``, solvates with OPC, neutralizes, and adds NaCl at
    ``salt_conc_M`` (using the empirical 9.03 ions / 1000 Å³ at 0.15 M baseline).
    Then runs ``parmed`` to apply the Li/Merz 12-6-4 correction to the metal
    ion type (e.g. ``Nd3+``) consistent with OPC water.

    Args:
        metal_pdb: PDB containing the protein and placed ion(s).
        out: Directory to write ``system.prmtop``, ``system.inpcrd``, and
            ``system.pdb``.
        build: Working directory for ``tleap.in`` and ``parmed.in`` scripts.
        ion: Ion code used to build the parmed mask (e.g. ``'ND'`` →
            ``Nd3+``).
        amberhome: Path to the ``$AMBERHOME`` install (provides ``bin/tleap``,
            ``bin/parmed``, and parameter data files).
        padding: Solvent padding added to each side of the box (Å).
        salt_conc_M: Target NaCl concentration in molar.

    Returns:
        Tuple ``(prmtop, inpcrd, pdb)`` of paths under ``out``.

    Raises:
        subprocess.CalledProcessError: If ``tleap`` or ``parmed`` exit non-zero.
    """
    tleap = str(amberhome / 'bin' / 'tleap')
    parmed = str(amberhome / 'bin' / 'parmed')

    xs, ys, zs = [], [], []
    with open(metal_pdb) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                xs.append(float(line[30:38]))
                ys.append(float(line[38:46]))
                zs.append(float(line[46:54]))

    extent = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    dim = int(extent + 2 * padding)
    n_ions = round(dim**3 * 10e-6 * (salt_conc_M / 0.15) * 9.03)

    prmtop = out / 'system.prmtop'
    inpcrd = out / 'system.inpcrd'
    pdb_out = out / 'system.pdb'
    tleap_in = build / 'tleap.in'

    tleap_in.write_text(
        'source leaprc.protein.ff19SB\n'
        'source leaprc.water.opc\n'
        'loadamberparams frcmod.ionslm_1264_opc\n'
        f'PROT = loadpdb {metal_pdb}\n'
        f'setbox PROT centers\n'
        f'set PROT box {{{dim} {dim} {dim}}}\n'
        'solvatebox PROT OPCBOX {0 0 0}\n'
        'addions PROT Na+ 0\n'
        'addions PROT Cl- 0\n'
        f'addIonsRand PROT Na+ {n_ions} Cl- {n_ions}\n'
        f'savepdb PROT {pdb_out}\n'
        f'saveamberparm PROT {prmtop} {inpcrd}\n'
        'quit\n'
    )

    # tleap and parmed both look up data files relative to $AMBERHOME
    amber_env = {**os.environ, 'AMBERHOME': str(amberhome)}

    subprocess.run(
        [tleap, '-f', str(tleap_in)],
        cwd=str(build), check=True, capture_output=True, text=True,
        env=amber_env,
    )

    # parmed mask: select by atom type (@%) with the Li/Merz casing, e.g. Nd3+, Ca2+
    ion_type = f'{ion.capitalize()}{ION_CHARGE[ion]}+'
    parmed_in = build / 'parmed.in'
    parmed_in.write_text(
        f'loadRestrt {inpcrd}\n'
        'setOverwrite True\n'
        f'add12_6_4 @%{ion_type} watermodel OPC\n'
        f'outparm {prmtop} {inpcrd}\n'
    )

    subprocess.run(
        [parmed, '-i', str(parmed_in), '-p', str(prmtop)],
        cwd=str(amberhome), check=True, capture_output=True, text=True,
        env=amber_env,
    )

    return prmtop, inpcrd, pdb_out


def extract_clusters(
    sim,
    ion: str,
    ion_to_group: dict,
    output_dir: Path,
    water_cutoff: float,
) -> list[Path]:
    """Carve QM-ready coordination clusters out of a relaxed MD simulation.

    Reads the current positions from ``sim``, and for each placed ion writes
    an XYZ file containing the ion, the protein residues that coordinate it
    (per ``ion_to_group``), and any OPC waters whose oxygen lies within
    ``water_cutoff`` Å of the ion. A sidecar JSON records ion identity,
    formal charge, an initial spin-multiplicity guess (from ``ION_SPIN``),
    and the residues used.

    Args:
        sim: Object exposing ``simulation.context`` and ``simulation.topology``
            in the OpenMM API (typically ``Simulator`` after ``run()``).
        ion: Ion residue name to look up in the topology (e.g. ``'Nd'``).
        ion_to_group: Mapping from ion residue id to coordinating protein
            residue ids, as produced by :func:`place_ions`.
        output_dir: Directory to write ``cluster_{resid}.xyz`` and
            ``cluster_{resid}.json``; created if missing.
        water_cutoff: Distance cutoff (Å) for including OPC waters in the
            cluster. ``<= 0`` skips waters entirely.

    Returns:
        List of XYZ paths written, one per ion that had matching atoms in
        the topology.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    state = sim.simulation.context.getState(getPositions=True)
    positions = np.array(state.getPositions(asNumpy=True))

    by_resid: dict[tuple[str, int], list] = {}
    for atom in sim.simulation.topology.atoms():
        key = (atom.residue.name, int(atom.residue.id))
        by_resid.setdefault(key, []).append(atom)

    written = []
    for ion_resid, binding_resids in ion_to_group.items():
        ion_atoms = by_resid.get((ion, ion_resid))
        if not ion_atoms:
            continue
        ion_pos = positions[ion_atoms[0].index]

        atoms_to_write = list(ion_atoms)
        for resid in binding_resids:
            for (rn, rid), alist in by_resid.items():
                if rid == resid and rn not in [ion, 'OPC', 'Na+', 'Cl-']:
                    atoms_to_write.extend(alist)
                    break

        if water_cutoff > 0:
            for (rn, _), alist in by_resid.items():
                if rn != 'OPC':
                    continue
                wpos = positions[[a.index for a in alist]]
                if np.min(np.linalg.norm(wpos - ion_pos, axis=1)) < water_cutoff:
                    atoms_to_write.extend(alist)

        residue_summary = sorted({(a.residue.name, int(a.residue.id))
                                  for a in atoms_to_write})
        xyz_path = output_dir / f'cluster_{ion_resid}.xyz'
        json_path = output_dir / f'cluster_{ion_resid}.json'

        with open(xyz_path, 'w') as f:
            f.write(f'{len(atoms_to_write)}\n')
            f.write(
                f'{ion}{ion_resid} + '
                + ','.join(f'{rn}{rid}' for rn, rid in residue_summary
                           if not (rn == ion and rid == ion_resid))
                + '\n'
            )
            for a in atoms_to_write:
                el = a.element.symbol if a.element is not None else a.name[0]
                x, y, z = positions[a.index]
                f.write(f'{el:<3s} {x:12.6f} {y:12.6f} {z:12.6f}\n')

        with open(json_path, 'w') as f:
            json.dump({
                'ion': ion,
                'ion_resid': ion_resid,
                'ion_formal_charge': ION_CHARGE[ion],
                'spin_multiplicity_guess': ION_SPIN[ion],
                'binding_residues': list(binding_resids),
                'n_atoms': len(atoms_to_write),
                'water_cutoff_A': water_cutoff,
            }, f, indent=2)

        written.append(xyz_path)

    return written


def relax(
    pdb_file: Path,
    binding_residues: list,
    ion: str,
    output_dir: Path,
    water_cutoff: float,
    amberhome: Path,
    chain_id: str = 'A',
    cluster_threshold: float = 7.,
    padding: float = 10.,
    salt_conc_M: float = 0.15,
    sim_kwargs: dict | None = None,
) -> list[Path]:
    """Place ions, build an Amber system, run MD, and extract QM clusters.

    End-to-end relaxation step: calls :func:`place_ions` to seat ions at the
    classifier-flagged residues, :func:`build_amber_system` to solvate and
    parameterize, runs heat/equil/production MD with
    ``molecular_simulations.simulate.Simulator``, and finally
    :func:`extract_clusters` to write per-ion XYZ + JSON files.

    Args:
        pdb_file: Input protein PDB.
        binding_residues: Residue numbers (1-indexed) flagged by the
            binding classifier for this ion.
        ion: Ion code (e.g. ``'CA'``, ``'ND'``).
        output_dir: Root directory for the run; ``build/`` holds ``tleap``
            inputs, ``clusters/`` holds extracted XYZs.
        water_cutoff: Distance cutoff (Å) for waters included in clusters;
            forwarded to :func:`extract_clusters`.
        amberhome: Path to ``$AMBERHOME``.
        chain_id: Chain to operate on for ion placement.
        cluster_threshold: Agglomerative-clustering threshold (Å) forwarded
            to :func:`place_ions`.
        padding: Solvent padding (Å) forwarded to :func:`build_amber_system`.
        salt_conc_M: NaCl concentration (M) forwarded to
            :func:`build_amber_system`.
        sim_kwargs: Overrides for the ``Simulator`` constructor. Defaults to
            25k heat / 50k equil / 250k production steps when ``None``.

    Returns:
        List of XYZ cluster paths from :func:`extract_clusters`. Empty if
        no coordination shell met ``MIN_COORD`` for ``ion``.
    """
    out = output_dir.resolve()
    build = out / 'build'
    build.mkdir(exist_ok=True, parents=True)

    metal_pdb = build / 'protein_with_metal.pdb'
    ion_to_group = place_ions(
        pdb_file=pdb_file,
        output_pdb=metal_pdb,
        binding_residues=binding_residues,
        ion=ion,
        chain_id=chain_id,
        cluster_threshold=cluster_threshold,
    )

    if not ion_to_group:
        # ESMBind flagged residues but they didn't form a coordination shell
        # for this ion — skip this (protein, ion) combo silently.
        return []

    build_amber_system(
        metal_pdb=metal_pdb,
        out=out,
        build=build,
        ion=ion,
        amberhome=amberhome,
        padding=padding,
        salt_conc_M=salt_conc_M,
    )

    if sim_kwargs is None:
        sim_kwargs = {
            'heat_steps': 25_000,
            'equil_steps': 50_000,
            'prod_steps': 250_000,
        }
    sim = Simulator(out, **sim_kwargs)
    sim.run()

    return extract_clusters(
        sim=sim,
        ion=ion,
        ion_to_group=ion_to_group,
        output_dir=out / 'clusters',
        water_cutoff=water_cutoff,
    )

def geomopt(
    xyz_path: Path,
    json_path: Path,
    basis: str,
    functional: str,
    dispersion: str,
    max_steps: int=200,
    constraints: Path | None=None,
    num_threads: int=4,
    verbose: int=4,
    max_memory: int=160000,
    grid_level: int=3
):
    """Optimize the geometry of an extracted cluster with DFT, then re-evaluate.

    Reads atoms from ``xyz_path`` and charge/spin metadata from ``json_path``
    (as written by :func:`extract_clusters`), runs a restricted Kohn-Sham
    geometry optimization via ``pyscf.geomopt.geometric_solver.optimize``
    with the requested ``functional`` and ``dispersion`` correction, and
    performs a final single-point ``RKS`` evaluation at the optimized
    geometry. Writes ``{stem}_opt.xyz`` and ``{stem}_opt.json`` (the input
    metadata plus DFT settings, final energy, and convergence flag) next
    to the input. Uses the GPU PySCF build when available.

    Args:
        xyz_path: Cluster geometry in XYZ format.
        json_path: Sidecar JSON with at least ``ion_formal_charge`` and
            ``spin_multiplicity_guess`` keys (PySCF's ``spin`` is set to
            ``multiplicity - 1``).
        basis: PySCF basis set name (e.g. ``'def2-TZVP'``).
        functional: Exchange-correlation functional name.
        dispersion: Dispersion correction name passed to ``mf.disp``
            (e.g. ``'d3bj'``).
        max_steps: Geometry-optimization step limit.
        constraints: Optional path to a geomeTRIC constraints file.
        num_threads: PySCF thread count (``lib.num_threads``).
        verbose: PySCF verbosity level.
        max_memory: Per-process memory budget in MB.
        grid_level: DFT integration grid level.

    Returns:
        Tuple ``(opt_coords, e_final)`` where ``opt_coords`` is an
        ``(n_atoms, 3)`` array of optimized coordinates in Å and
        ``e_final`` is the final SCF energy in Hartree.
    """
    geom_str = ''
    elements = []
    with open(xyz_path) as f:
        n_atom = int(f.readline().strip())
        _ = f.readline()
        for _ in range(n_atom):
            parts = f.readline().split()
            elements.append(parts[0])
            geom_str += (
                f'{parts[0]}  {float(parts[1]):.8f}  '
                f'{float(parts[2]):.8f}  {float(parts[3]):.8f}\n'
            )

    lib.num_threads(num_threads)

    with open(json_path) as f:
        metadata = json.load(f)

    mol = gto.M(
        atom=geom_str,
        basis=basis,
        charge=metadata['ion_formal_charge'],
        spin=metadata['spin_multiplicity_guess'] - 1,
        verbose=verbose,
        max_memory=max_memory,
        symmetry=False
    )

    mf = dft.RKS(mol)
    mf.xc = functional
    mf.disp = dispersion
    mf.grids.level = grid_level
    mf.grids.build()

    mol_eq = optimize(
        mf,
        maxsteps=max_steps,
        constraints=constraints,
    )

    opt_coords = mol_eq.atom_coords(unit='Angstrom')

    mf_final = dft.RKS(mol_eq)
    mf_final.xc = functional
    mf_final.disp = dispersion
    mf_final.grids.level = grid_level
    mf_final.grids.build()
    e_final = mf_final.kernel()

    out_xyz = xyz_path.with_name(f'{xyz_path.stem}_opt.xyz')
    with open(out_xyz, 'w') as f:
        f.write(f'{n_atom}\n')
        f.write(f'{xyz_path.stem} optimized E={float(e_final):.8f} Ha\n')
        for el, (x, y, z) in zip(elements, opt_coords, strict=True):
            f.write(f'{el:<3s} {x:12.6f} {y:12.6f} {z:12.6f}\n')

    out_json = xyz_path.with_name(f'{xyz_path.stem}_opt.json')
    with open(out_json, 'w') as f:
        json.dump({
            **metadata,
            'energy_hartree': float(e_final),
            'basis': basis,
            'functional': functional,
            'dispersion': dispersion,
            'grid_level': grid_level,
            'converged': bool(getattr(mf_final, 'converged', False)),
        }, f, indent=2)

    return opt_coords, e_final
