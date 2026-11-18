# =============================================================================
# Adsorbates on the Pt nanoparticle (paste into build_interface.py, below assemble_solid)
# =============================================================================
# Why ase.build.add_adsorbate does not work here: it is written for a flat slab. It drops the
# adsorbate `height` above the HIGHEST atom of whatever Atoms object you give it, at one (x, y)
# position, along z, and it knows nothing about facets, edges, the Pt/graphene contact line or
# the other atoms. On a nanoparticle that puts O above the top vertex, inside the particle, or
# on the support. The functions below find the sites FROM the particle's own geometry and reject
# the ones that are buried, face the graphene or crowd carbon.
#
# Config fields to add:
#     coverage: float = 0.0          # ML = adsorbates / exposed surface Pt (see below)
#     adsorbate: str = "O"           # O | OH
#     ads_site: str = "fcc"          # fcc | hcp | hollow (fcc+hcp) | 4fold | bridge | ontop | mixed
#     target: str = "all"            # all | top  (top = only sites above the particle's centre)
#     ads_seed: int = 7              # different seed -> different arrangement (use for training diversity)
#     ads_pt_clear: float = 2.6      # A, every Pt that does not bind the adsorbate stays this far away
#     ads_c_min: float = 2.8         # A, minimum distance to graphene carbon
#     ads_contact: float = 3.0       # A, surface Pt closer than this to graphene C is "in contact", not exposed
#     ads_zmin: float = 1.0          # A, sites must be this far above the top graphene layer
#     ads_probe: float = 1.8         # A, open-to-solution test point above the site
#     ads_oo_min: float = 2.6        # A, minimum adsorbate-adsorbate distance (1 ML fcc is 2.77 A)
#     ads_strict: bool = True        # raise if the requested coverage cannot be placed
#
# Coverage definition (state this in the paper): ML = N_adsorbate / N_exposed, where N_exposed is
# the number of surface Pt atoms (Pt-Pt coordination <= 9) that are not in contact with the
# graphene (Pt-C distance >= ads_contact). Small particles give coarse coverage steps: check the
# printed 'achieved' value.
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