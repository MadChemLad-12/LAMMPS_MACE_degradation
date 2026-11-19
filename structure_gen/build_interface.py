#!/usr/bin/env python
"""
Pt nanoparticle / graphene / Nafion / water builder for LAMMPS (MACE) simulations.

Every adjustable variable lives in Config (edit the defaults, or override in main()).
All functions receive cfg, so nothing is hard-coded further down.
Atom order in data.lammps: [graphene C + Pt] [Nafion copies] [waters (O,H,H)].
"""
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from ase import Atoms
from ase.build import graphene, make_supercell, molecule
from ase.cluster import Decahedron, Icosahedron, Octahedron, wulff_construction
from ase.io import read, write
from ase.neighborlist import neighbor_list


# =============================================================================
# CONFIG - everything you normally change is here
# =============================================================================
@dataclass
class Config:
    # ---- Pt nanoparticle ----------------------------------------------------
    np_shape: str = "octahedron"      # octahedron | decahedron | icosahedron | wulff
    np_sizes_nm: tuple = (0.8, 1.2)   # volume-equivalent diameters, nm
    oct_family: str = "any"           # octahedron only: any | truncated | cuboctahedron
    facet_down: str = "largest"           # 111 | 100 | largest  (facet facing the graphene)
    np_rotate_z: float = 0.0          # degrees about z (align NP with graphene lattice)
    wulff_surfaces: tuple = ((1, 1, 1), (1, 0, 0), (1, 1, 0))
    wulff_energies: tuple = (1.49, 1.73, 1.76)        # relative surface energies -> shape only
    a_pt: float = 3.92                # Pt lattice constant, A
    
    # ---- graphene support ---------------------------------------------------
    layers: int = 2                  # AB-stacked layers
    layer_spacing: float = 3.35       # A
    graphene_cells: tuple = None      # (nx, ny) orthogonal cells (2.46 x 4.26 A); None = auto from NP
    margin: float = 7.0               # graphene beyond the NP edge, A (NP-image gap = 2*margin)
    z_base: float = 5.0               # z of the bottom graphene layer, A (keeps it off the cell edge)

    # ---- NP / support contact ----------------------------------------------
    number: int = 1                   # number of NPs per graphene sheet
    embedded: bool = True             # True: NP sunk into the top layer; False: sits on top
    np_gap: float = 3.3               # not embedded: NP bottom this far ABOVE the top layer, A
    embed_depth: float = 1.2          # embedded: NP bottom this far BELOW the top layer, A
    embed_cutoff: float = 2.0         # embedded: remove top-layer C closer than this to any Pt, A

    # ---- adsorbate coverage ---
    coverage: float = 0.0          # ML = adsorbates / exposed surface Pt (see below)
    adsorbate: str = "O"           # O | OH
    ads_site: str = "mixed"          # fcc | hcp | hollow (fcc+hcp) | 4fold | bridge | ontop | mixed
    target: str = "all"            # all | top  (top = only sites above the particle's centre)
    ads_seed: int = 7              # different seed -> different arrangement (use for training diversity)
    ads_pt_clear: float = 2.6      # A, every Pt that does not bind the adsorbate stays this far away
    ads_c_min: float = 2.8         # A, minimum distance to graphene carbon
    ads_contact: float = 3.0       # A, surface Pt closer than this to graphene C is "in contact", not exposed
    ads_zmin: float = 1.0          # A, sites must be this far above the top graphene layer
    ads_probe: float = 1.8         # A, open-to-solution test point above the site
    ads_oo_min: float = 2.6        # A, minimum adsorbate-adsorbate distance (1 ML fcc is 2.77 A)
    ads_strict: bool = True   
    
    # ---- Nafion -------------------------------------------------------------
    nafion_xyz: str = "/home/h.jack/MACE_train/LAMMPS_MACE_degradation/structure_gen/Simple_nafion.xyz"
    #nafion_xyz: str = "/home/h.jack/MACE_train/LAMMPS_MACE_degradation/NAFION_FULL_SINGLE.xyz"
    #nafion_xyz: str = "/home/h.jack/MACE_train/LAMMPS_MACE_degradation/NAFION_FULL_SINGLE_bent.xyz"
    n_nafion: int = 6                 # 0 = no Nafion
    head_mode: str = "free"       # graphene | pt | free
    head_zmax: float = 7.0            # graphene mode: head atoms below z_top_layer + this, A
    head_shell: float = 5.0           # pt mode: head atoms within r_max + this of NP centroid, A
    nafion_density: float = 1.9       # g/cc, only used to estimate excluded volume

    # ---- water / packing ----------------------------------------------------
    tol: float = 2.5                  # Packmol tolerance, A
    water_above: float = 12.0         # water above NP top, A
    water_floor: float = 2.5          # water starts this far above the top graphene layer, A
    water_fill: float = 0.85          # fraction of 1 g/cc to pack (slack for NP / Nafion)
    vac: float = 10.0                 # vacuum above the liquid, A
    seed: int = 421
    run_packmol: bool = True

    # ---- output -------------------------------------------------------------
    outdir: str = "structure_gen/structures"
    specorder: tuple = ("Pt", "C", "O", "H", "F", "S")
    view: bool = False                # ase.visualize.view of each finished system


RHO_W = 0.0334          # water molecules / A^3 at 1 g/cc
A_CC = 1.42             # graphene C-C bond, A


# =============================================================================
# Nanoparticle
# =============================================================================
def vol_eq_nm(n_atoms, a_pt):
    """Diameter (nm) of a sphere with the same volume as n_atoms fcc Pt atoms."""
    return (3 * n_atoms * a_pt**3 / (2 * np.pi)) ** (1 / 3) / 10.0


def extent(atoms, a_pt):
    """Max atom-atom distance + one nearest-neighbour spacing (A): a 'physical' diameter."""
    p = atoms.positions
    return max(np.linalg.norm(p - q, axis=1).max() for q in p) + a_pt / np.sqrt(2)


def aspect(atoms):
    """sqrt(largest/smallest eigenvalue) of the position covariance: 1 = round."""
    w = np.linalg.eigvalsh(np.cov((atoms.positions - atoms.positions.mean(0)).T))
    return float(np.sqrt(w[-1] / max(w[0], 1e-9)))


def _candidates(cfg, n_target):
    """Yield (label, Atoms) for every particle of the requested shape family."""
    shape = cfg.np_shape.lower()
    if shape == "octahedron":
        for L in range(2, 16):
            for c in range((L - 1) // 2 + 1):
                if cfg.oct_family == "truncated" and not (c >= 1 and L == 3 * c + 1):
                    continue
                if cfg.oct_family == "cuboctahedron" and not (c >= 1 and L == 2 * c + 1):
                    continue
                yield f"Octahedron(L={L},c={c})", Octahedron("Pt", length=L, cutoff=c)
    elif shape == "decahedron":
        for p in range(2, 7):
            for q in range(1, 6):
                for r in range(0, 3):
                    yield f"Decahedron(p={p},q={q},r={r})", Decahedron("Pt", p=p, q=q, r=r)
    elif shape == "icosahedron":
        for n in range(2, 7):
            yield f"Icosahedron(shells={n})", Icosahedron("Pt", noshells=n)
    elif shape in ("wulff", "cluster"):
        # wulff_construction only roughly honours the requested size, so scan around it
        for n_req in range(max(4, int(0.8 * n_target)), int(1.25 * n_target) + 2, 2):
            cl = wulff_construction("Pt", list(cfg.wulff_surfaces), list(cfg.wulff_energies),
                                    n_req, "fcc", rounding="closest",
                                    latticeconstant=cfg.a_pt)
            yield f"Wulff(N_req={n_req})", cl
    else:
        raise ValueError(f"unknown np_shape {cfg.np_shape!r}")


def select_np(cfg, d_nm):
    """Pick the particle of the chosen shape whose volume-equivalent diameter is closest."""
    n_target = max(4, round(np.pi / 6 * (10 * d_nm) ** 3 / (cfg.a_pt**3 / 4)))
    best = None
    for label, cl in _candidates(cfg, n_target):
        a = Atoms(cl)
        a.set_pbc(False)
        if not 4 <= len(a) <= 600:
            continue
        # size error dominates; asphericity breaks ties between equally sized variants
        score = abs(vol_eq_nm(len(a), cfg.a_pt) - d_nm) + 0.02 * (aspect(a) - 1.0)
        if best is None or score < best[0]:
            best = (score, label, a)
    if best is None:
        raise RuntimeError("no candidate particles found")
    return best[1], best[2]


def facet_groups(pos):
    """Group convex-hull triangles into planar facets -> list of (outward normal, area)."""
    hull = ConvexHull(pos)
    groups = []
    for eq, simp in zip(hull.equations, hull.simplices):
        n = eq[:3] / np.linalg.norm(eq[:3])
        t = pos[simp]
        area = 0.5 * np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0]))
        for g in groups:
            if np.dot(g[0], n) > 0.999:
                g[1] += area
                break
        else:
            groups.append([n, area])
    return [(n, a) for n, a in groups]


def orient_facet_down(cfg, atoms):
    """Rotate so the chosen facet is horizontal and faces -z (toward the support)."""
    shape, want = cfg.np_shape.lower(), cfg.facet_down
    cubic = shape in ("octahedron", "wulff", "cluster")     # cubic axes along x,y,z
    pos = atoms.positions - atoms.positions.mean(0)
    groups = facet_groups(pos)

    def match(n):
        if want == "largest":
            return True
        a = np.sort(np.abs(n))
        if want == "111":
            if cubic:
                return np.allclose(a, 1 / np.sqrt(3), atol=0.05)
            return abs(n[2]) > 0.1 if shape == "decahedron" else True   # skip vertical (100) sides
        if want == "100":
            if cubic:
                return np.allclose(a, [0, 0, 1], atol=0.05)
            return shape == "decahedron" and abs(n[2]) < 0.1            # decahedron (100) sides
        raise ValueError(f"facet_down must be 111, 100 or largest, got {want!r}")

    cand = [(n, a) for n, a in groups if match(n)]
    if not cand:
        raise ValueError(f"{cfg.np_shape} has no {want} facet (try facet_down='largest')")
    n, _ = max(cand, key=lambda t: t[1])
    atoms.rotate(n, (0, 0, -1))
    atoms.rotate(cfg.np_rotate_z, "z")
    atoms.center()


def build_np(cfg, d_nm):
    label, atoms = select_np(cfg, d_nm)
    orient_facet_down(cfg, atoms)
    atoms.set_cell(None)
    atoms.set_pbc(False)
    return label, atoms


# =============================================================================
# Graphene support + NP placement / embedding
# =============================================================================
def build_graphene(cfg, np_width_xy):
    unit = graphene(size=(1, 1, 1))
    unit = make_supercell(unit, [[1, 0, 0], [1, 2, 0], [0, 0, 1]])   # 2.46 x 4.26 A
    ax, ay = unit.cell[0, 0], unit.cell[1, 1]
    if cfg.graphene_cells is None:
        L = max(np_width_xy) + 2 * cfg.margin                        # ~square sheet
        nx, ny = int(np.ceil(L / ax)), int(np.ceil(L / ay))
    else:
        nx, ny = cfg.graphene_cells
    sheet = unit.repeat((nx, ny, 1))
    sheet.set_pbc((True, True, False))
    gr = Atoms()
    for k in range(cfg.layers):
        layer = sheet.copy()
        if k % 2 == 1:                       # AB (Bernal) stacking: shift one C-C bond along y
            layer.positions[:, 1] += A_CC
            layer.wrap()
        layer.positions[:, 2] = cfg.z_base + k * cfg.layer_spacing
        layer.set_tags(k)
        gr += layer
    Lx, Ly = nx * ax, ny * ay
    gr.set_cell(np.diag([Lx, Ly, 0.0]))
    return gr, Lx, Ly


def assemble_solid(cfg, np_, gr, Lx, Ly):
    """Place the NP on / in the top layer; drop overlapping C when embedded."""
    z_top_layer = cfg.z_base + (cfg.layers - 1) * cfg.layer_spacing
    p = np_.positions.copy()
    p[:, 0] += Lx / 2 - 0.5 * (p[:, 0].min() + p[:, 0].max())
    p[:, 1] += Ly / 2 - 0.5 * (p[:, 1].min() + p[:, 1].max())
    z_bottom = z_top_layer - cfg.embed_depth if cfg.embedded else z_top_layer + cfg.np_gap
    p[:, 2] += z_bottom - p[:, 2].min()
    np_ = Atoms("Pt" * len(np_), positions=p)

    n_removed = 0
    if cfg.embedded:
        top = np.where(gr.get_tags() == cfg.layers - 1)[0]
        dmin = np.linalg.norm(gr.positions[top][:, None, :] - p[None, :, :], axis=2).min(axis=1)
        drop = top[dmin < cfg.embed_cutoff]
        n_removed = len(drop)
        keep = np.setdiff1d(np.arange(len(gr)), drop)
        gr = gr[keep]
        if cfg.layers > 1:
            low = np.where(gr.get_tags() < cfg.layers - 1)[0]
            dl = np.linalg.norm(gr.positions[low][:, None, :] - p[None, :, :], axis=2).min()
            if dl < 2.0:
                print(f"   WARNING: NP is {dl:.2f} A from a lower graphene layer - reduce embed_depth")

    solid = gr + np_
    if "tags" in solid.arrays:               # keep Packmol's xyz reader happy
        del solid.arrays["tags"]
    dpc = np.linalg.norm(gr.positions[:, None, :] - p[None, :, :], axis=2).min()
    return solid, np_, z_top_layer, n_removed, dpc

# ====================================
# add adsorbates
# ====================================

import itertools

ADS_TEMPLATES = {            # (element, offset along [tilt-axis, 0, normal]); first atom binds to Pt
    "O": [("O", (0.0, 0.0, 0.0))],
    "OH": [("O", (0.0, 0.0, 0.0)), ("H", (0.49, 0.0, 0.85))],     # O-H 0.98 A, tilted 30 deg
}
D_BOND = {"ontop": 2.05, "bridge": 2.05, "fcc": 2.05, "hcp": 2.05, "4fold": 2.1}   # Pt-O, A


def _surface_info(P, a_pt):
    """neighbour matrix, coordination numbers, surface mask and local outward normals of the Pt atoms"""
    d = np.linalg.norm(P[:, None] - P[None], axis=2)
    nn = (d > 0.1) & (d < 1.15 * a_pt / np.sqrt(2))
    cn = nn.sum(1)
    surf = cn <= 9
    nrm = np.zeros_like(P)
    for i in range(len(P)):
        v = P[i] - P[nn[i]].mean(0) if cn[i] else P[i] - P.mean(0)
        nrm[i] = v / (np.linalg.norm(v) + 1e-12)
    return d, nn, cn, surf, nrm


def _candidate_sites(P, a_pt, kinds):
    d, nn, cn, surf, nrm = _surface_info(P, a_pt)
    S = np.where(surf)[0]
    cen = P.mean(0)
    sites = []

    def add(kind, atoms, n=None):
        idx = list(atoms)
        c = P[idx].mean(0)
        navg = nrm[idx].mean(0)
        navg /= np.linalg.norm(navg) + 1e-12
        if n is None:
            n = navg
            if kind == "bridge":           # normal must be perpendicular to the bond or O ends up off-centre
                b = P[idx[1]] - P[idx[0]]
                b /= np.linalg.norm(b)
                n = n - (n @ b) * b
                n /= np.linalg.norm(n) + 1e-12
        elif n @ navg < 0:                 # geometric facet normal, oriented like the local outward normals
            n = -n
        R = np.linalg.norm(P[idx] - c, axis=1).mean()
        h = np.sqrt(max(D_BOND[kind] ** 2 - R ** 2, 0.25 ** 2))
        q = c + n * h
        if np.abs(np.linalg.norm(P[idx] - q, axis=1) - D_BOND[kind]).max() > 0.3:
            return                          # distorted site (corner/edge geometry): O would not sit symmetrically
        sites.append(dict(kind=kind, atoms=tuple(atoms), pos=q, n=n))

    if "ontop" in kinds:
        for i in S:
            add("ontop", (i,))
    if "bridge" in kinds:
        for i, j in itertools.combinations(S, 2):
            if nn[i, j]:
                add("bridge", (i, j))
    if kinds & {"fcc", "hcp"}:
        for i, j, k in itertools.combinations(S, 3):
            if nn[i, j] and nn[j, k] and nn[i, k]:
                n = np.cross(P[j] - P[i], P[k] - P[i])
                n /= np.linalg.norm(n) + 1e-12
                navg = nrm[[i, j, k]].mean(0)
                if n @ navg < 0:
                    n = -n
                c = P[[i, j, k]].mean(0)
                below = c - n * (a_pt / np.sqrt(3))                 # one layer down along the FACET normal
                kind = "hcp" if np.linalg.norm(P - below, axis=1).min() < 0.9 else "fcc"
                if kind in kinds:
                    add(kind, (i, j, k), n)
    if "4fold" in kinds:
        seen = set()
        for i in S:
            for j, k in itertools.combinations(np.where(nn[i])[0], 2):
                if j in S and k in S and abs(d[j, k] - a_pt) < 0.35:
                    for l in np.where(nn[j] & nn[k])[0]:
                        if l != i and l in S and abs(d[i, l] - a_pt) < 0.35:
                            key = frozenset((i, j, k, l))
                            quad = P[[i, j, k, l]]
                            flat = np.linalg.svd(quad - quad.mean(0), compute_uv=False)[-1]
                            if key not in seen and flat < 0.4:
                                seen.add(key)
                                n = np.linalg.svd(quad - quad.mean(0))[2][-1]
                                add("4fold", (i, j, k, l), n)
    return sites, surf


def _site_ok(site, allpos, is_pt, is_c, z_min, cfg):
    q, n = site["pos"], site["n"]
    d = np.linalg.norm(allpos - q, axis=1)
    support = np.zeros(len(allpos), bool)
    support[list(site["pt_index"])] = True
    if (d[is_pt & ~support] < cfg.ads_pt_clear).any():       # other Pt too close (buried / crowded)
        return False
    if (d[is_c] < cfg.ads_c_min).any():                      # on or next to the support
        return False
    if q[2] < z_min:
        return False
    probe = q + n * cfg.ads_probe                            # must be open to the solution
    if (np.linalg.norm(allpos - probe, axis=1) < 2.2).any():
        return False
    return True


def _pick(xyz, n, oo_min, rng, tries=400):
    """choose n sites with mutual distance >= oo_min; alternates spread-out and random greedy packing"""
    m = len(xyz)
    D = np.linalg.norm(xyz[:, None] - xyz[None], axis=2)
    best = []
    for t in range(tries):
        if t % 2 == 0:                                       # farthest-first from a random start
            ch = [int(rng.integers(m))]
            while len(ch) < n:
                ok = np.where((D[:, ch] >= oo_min).all(1))[0]
                if not len(ok):
                    break
                ch.append(int(ok[np.argmax(D[np.ix_(ok, ch)].min(1))]))
        else:                                                # random greedy (packs dense layers better)
            ch = []
            for s in rng.permutation(m):
                if all(D[s, c] >= oo_min for c in ch):
                    ch.append(int(s))
                    if len(ch) == n:
                        break
        if len(ch) > len(best):
            best = ch
        if len(best) == n:
            break
    return best


def add_adsorbate(cfg, solid, n_gr, n_pt, verbose=True):
    """Return (solid + adsorbates, info). `solid` = graphene C (first n_gr atoms) + Pt (next n_pt atoms)
    as built by assemble_solid; adsorbates are appended AFTER the Pt atoms."""
    info = dict(n_exposed=0, n_sites=0, n_requested=0, n_placed=0, achieved_ml=0.0)
    if cfg.coverage == 0.0:
        print("Coverage of 0.0 ML chosen. Adding 0 adsorbates")
        return solid, info
    if cfg.adsorbate not in ADS_TEMPLATES:
        raise SystemExit(f"adsorbate '{cfg.adsorbate}' not in {list(ADS_TEMPLATES)}")
    if cfg.target not in ("all", "top"):
        raise SystemExit("target must be 'all' or 'top'")
    kinds_all = {"fcc", "hcp", "4fold", "bridge", "ontop"}
    kinds = {"mixed": kinds_all, "hollow": {"fcc", "hcp"}}.get(cfg.ads_site, {cfg.ads_site})
    if not kinds <= kinds_all:
        raise SystemExit(f"ads_site '{cfg.ads_site}' unknown")
    pos = solid.positions
    pt_idx = np.arange(n_gr, n_gr + n_pt)
    P = pos[pt_idx]
    is_pt = np.zeros(len(solid), bool)
    is_pt[pt_idx] = True
    is_c = np.zeros(len(solid), bool)
    is_c[:n_gr] = True
    z_top_layer = pos[:n_gr, 2].max()
    z_min = z_top_layer + cfg.ads_zmin

    def usable(sites):
        out = []
        for s in sites:
            s["pt_index"] = tuple(int(pt_idx[a]) for a in s["atoms"])
            if _site_ok(s, pos, is_pt, is_c, z_min, cfg):
                out.append(s)
        return out

    # exposed surface Pt = surface atoms (Pt-Pt CN <= 9) that are not in contact with the graphene
    _, _, _, surf, _ = _surface_info(P, cfg.a_pt)
    dC = np.linalg.norm(P[:, None] - pos[:n_gr][None], axis=2).min(1)
    n_exp = int((surf & (dC >= cfg.ads_contact)).sum())
    sites, _ = _candidate_sites(P, cfg.a_pt, kinds)
    sites = usable(sites)
    if cfg.target == "top":
        sites = [s for s in sites if s["pos"][2] > P[:, 2].mean()]
    n_req = int(np.floor(cfg.coverage * n_exp + 0.5))
    info.update(n_exposed=n_exp, n_sites=len(sites), n_requested=n_req)
    if n_req == 0 or not sites:
        raise SystemExit(f"no adsorbate placed: coverage {cfg.coverage} x {n_exp} exposed Pt = {n_req}, "
                         f"{len(sites)} valid '{cfg.ads_site}' sites")
    rng = np.random.default_rng(cfg.ads_seed)
    pick = _pick(np.array([s["pos"] for s in sites]), n_req, cfg.ads_oo_min, rng)
    info["n_placed"] = len(pick)
    info["achieved_ml"] = len(pick) / n_exp
    if len(pick) < n_req:
        msg = (f"requested {n_req} {cfg.adsorbate} ({cfg.coverage} ML of {n_exp} exposed Pt) but only {len(pick)} fit "
               f"({len(sites)} valid '{cfg.ads_site}' sites, O-O >= {cfg.ads_oo_min} A): max {info['achieved_ml']:.3f} ML. "
               "Try ads_site='mixed', a smaller ads_oo_min or a lower coverage.")
        if cfg.ads_strict:
            raise SystemExit("ERROR: " + msg)
        print("   WARNING: " + msg)
    new = Atoms()
    tmpl = ADS_TEMPLATES[cfg.adsorbate]
    kinds_used = {}
    for k in pick:
        s = sites[k]
        n = s["n"]
        a = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(n, a)
        phi = rng.uniform(0, 2 * np.pi)
        t = np.cos(phi) * a + np.sin(phi) * b                # random azimuth for OH
        for el, (x, _, z) in tmpl:
            new += Atoms(el, positions=[s["pos"] + x * t + z * n])
        kinds_used[s["kind"]] = kinds_used.get(s["kind"], 0) + 1
    out = solid + new
    if "tags" in out.arrays:
        del out.arrays["tags"]
    if verbose:
        dpt = np.linalg.norm(new.positions[:, None] - P[None], axis=2).min()
        dc = np.linalg.norm(new.positions[:, None] - pos[:n_gr][None], axis=2).min()
        print(f"   adsorbates: {len(pick)} x {cfg.adsorbate} on {n_exp} exposed Pt = {info['achieved_ml']:.3f} ML "
              f"(asked {cfg.coverage}); sites {kinds_used}; closest Pt {dpt:.2f} A, closest C {dc:.2f} A")
    return out, info

# =============================================================================
# Nafion + Packmol
# =============================================================================
def head_indices(naf):
    """1-based Packmol indices of each sulfonate head: S plus the O atoms bonded to it."""
    sym = naf.get_chemical_symbols()
    idx = set()
    for iS in (k for k, s in enumerate(sym) if s == "S"):
        d = naf.get_distances(iS, range(len(naf)))
        idx |= {iS} | {k for k, s in enumerate(sym) if s == "O" and d[k] < 1.9}
    if not idx:
        raise ValueError("no sulfur found in the Nafion file")
    return sorted(i + 1 for i in idx)


def head_block(cfg, idx, pt, z_top_layer):
    ids = " ".join(map(str, idx))
    if cfg.head_mode == "graphene":
        c = f"below plane 0. 0. 1. {z_top_layer + cfg.head_zmax:.2f}"
    elif cfg.head_mode == "pt":
        cen = pt.positions.mean(axis=0)
        r = np.linalg.norm(pt.positions - cen, axis=1).max() + cfg.head_shell
        c = f"inside sphere {cen[0]:.2f} {cen[1]:.2f} {cen[2]:.2f} {r:.2f}"
    elif cfg.head_mode == "free":
        return ""
    else:
        raise ValueError("head_mode must be graphene, pt or free")
    return f"  atoms {ids}\n    {c}\n  end atoms\n"

def pack_system(cfg, solid, pt, z_top_layer, Lx, Ly, workdir):
    tol = cfg.tol
    naf = read(cfg.nafion_xyz) if cfg.n_nafion > 0 else None
    z_floor = z_top_layer + cfg.water_floor
    z_top = pt.positions[:, 2].max() + cfg.water_above
    v_excl = np.pi / 6 * (extent(pt, cfg.a_pt) + tol) ** 3
    if naf is not None:
        L_naf = max(np.linalg.norm(naf.positions - q, axis=1).max() for q in naf.positions)
        z_top = max(z_top, z_floor + L_naf + 2.0)           # room for the whole chain
        v_excl += cfg.n_nafion * 1.6605 * naf.get_masses().sum() / cfg.nafion_density
    n_w = int(cfg.water_fill * RHO_W * (Lx * Ly * (z_top - z_floor) - v_excl))

    write(workdir / "solid.xyz", solid)
    write(workdir / "water.xyz", molecule("H2O"))
    box = f"{tol/2} {tol/2} {z_floor} {Lx - tol/2} {Ly - tol/2} {z_top}"
    naf_block = ""
    if naf is not None:
        naf_block = (f"structure {Path(cfg.nafion_xyz).resolve()}\n"
                     f"  number {cfg.n_nafion}\n  inside box {box}\n"
                     f"{head_block(cfg, head_indices(naf), pt, z_top_layer)}end structure\n\n")
    (workdir / "pack.inp").write_text(
        f"tolerance {tol}\nfiletype xyz\nseed {cfg.seed}\noutput liquid.xyz\n\n"
        f"structure solid.xyz\n  number 1\n  fixed 0. 0. 0. 0. 0. 0.\nend structure\n\n"
        f"{naf_block}"
        f"structure water.xyz\n  number {n_w}\n  inside box {box}\nend structure\n")

    if cfg.run_packmol:
        subprocess.run("packmol < pack.inp > pack.log", shell=True, check=True, cwd=workdir)
    system = read(workdir / "liquid.xyz")
    system.set_cell(np.diag([Lx, Ly, z_top + cfg.vac]))
    system.set_pbc(True)
    return system, naf, n_w


def check(system, n_solid, n_naf, len_naf):
    """Inter-molecular contacts < 1.5 A (periodic). Solid = one group, Nafion/water per molecule."""
    n = len(system)
    mol = np.zeros(n, dtype=int)
    for k in range(n_naf):
        mol[n_solid + k * len_naf: n_solid + (k + 1) * len_naf] = 1 + k
    start = n_solid + n_naf * len_naf
    mol[start:] = 1000 + np.arange(n - start) // 3          # O,H,H per water
    i, j, d = neighbor_list("ijd", system, 1.5)
    bad = mol[i] != mol[j]
    return int(bad.sum()), (float(d[bad].min()) if bad.any() else None)


# =============================================================================
# Driver
# =============================================================================
def build_one(cfg, d_nm):
    label, np0 = build_np(cfg, d_nm)
    width = np.ptp(np0.positions[:, :2], axis=0)
    gr, Lx, Ly = build_graphene(cfg, width)
    solid, pt, z_top_layer, n_removed, dpc = assemble_solid(cfg, np0, gr, Lx, Ly)
    n_gr = len(solid) - len(pt)
    solid, ads_info = add_adsorbate(cfg, solid, n_gr, len(pt))
    tag = f"{cfg.np_shape}_{len(pt):03d}pt" + (f"_{cfg.adsorbate}{ads_info['achieved_ml']:.3f}ML_s{cfg.ads_seed}" if ads_info["n_placed"] else "")
    workdir = Path(cfg.outdir) / tag
    workdir.mkdir(parents=True, exist_ok=True)

    system, naf, n_w = pack_system(cfg, solid, pt, z_top_layer, Lx, Ly, workdir)
    n_naf, len_naf = (cfg.n_nafion, len(naf)) if naf is not None else (0, 0)
    nbad, dmin = check(system, len(solid), n_naf, len_naf)

    write(workdir / "data.lammps", system, format="lammps-data", atom_style="atomic",
          specorder=list(cfg.specorder), masses=True)
    if cfg.view:
        from ase.visualize import view
        view(system)

    print(f"{d_nm:5.2f} {len(pt):5d} {vol_eq_nm(len(pt), cfg.a_pt):6.2f} "
          f"{extent(pt, cfg.a_pt)/10:6.2f} {n_removed:5d} {dpc:6.2f} "
          f"{Lx:6.1f}x{Ly:5.1f} {len(system):7d}   contacts<1.5A: {nbad}"
          + (f" (min {dmin:.2f})" if dmin else "") + f"   [{label}]")
    print(f"Files written to {workdir}/data.lammps ")
    print(f"    and {workdir}/liquid.xyz ")
    print("")


def main():
    cfg = Config()          # override here if you like, e.g. Config(np_shape="icosahedron")
    sizes = cfg.np_sizes_nm
    if isinstance(sizes, (int, float)):
        sizes = [float(sizes)]
    else:
        sizes = [float(x) for x in sizes]

    print(f"{'d_req':>5} {'N_Pt':>5} {'d_vol':>6} {'d_ext':>6} {'C_rm':>5} "
          f"{'minPtC':>6} {'cell':>12} {'atoms':>7}")
    for d in sizes:
        build_one(cfg, d)


if __name__ == "__main__":
    main()