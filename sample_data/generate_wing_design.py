"""
Generate 3D wing aerodynamic design space data.

Physics modeled (6 inputs, 4 outputs):
  - Oswald span efficiency factor via Raymer's approximation (AR, taper, sweep)
  - Induced drag: CL^2 / (pi * AR * e)
  - Korn's equation for drag-divergence Mach (with quarter-chord sweep effect)
  - Cubic wave drag rise above drag-divergence Mach
  - Profile drag: Torenbeek form factor * Prandtl-Schlichting skin friction
    with mild compressibility correction above M=0.40
  - L/D ratio from CL_design / (CD_induced + CD_wave + CD_profile)
    computed using noisy drag values (as in real flight test or CFD data)

Design intent: sharply different sensitivities per output
  CD_induced: dominated by aspect_ratio and CL_design
  CD_wave:    dominated by mach, sweep_deg, thickness_chord
  CD_profile: dominated by thickness_chord (and mildly by mach)
  LD_ratio:   integrates all of the above

Sampling: Latin Hypercube-style (6 independent shuffles, 200 rows)

Run from the project root:
    conda activate base
    python sample_data/generate_wing_design.py

Outputs: sample_data/wing_design_space.csv
"""

import math
import random

random.seed(42)


def lhs_col(lo, hi, n):
    """
    Latin Hypercube column: n evenly-spaced level centers across [lo, hi],
    independently shuffled.
    """
    step = (hi - lo) / n
    vals = [lo + step * (i + 0.5) for i in range(n)]
    random.shuffle(vals)
    return vals


def oswald_factor(ar, taper, sweep_deg):
    """
    Span efficiency (Oswald) factor using Raymer's approximation.
    Accounts for non-elliptic lift distribution due to taper and sweep.
    Optimal taper ratio for unswept wing is ~0.35.
    Clamped to 0.55 (realistic lower bound for swept tapered wings).
    """
    # Raymer's baseline elliptic correction
    e_base = 1.78 * (1.0 - 0.045 * ar ** 0.68) - 0.64
    # Taper penalty: wings with taper far from 0.35 lose efficiency
    taper_factor = 1.0 - 0.002 * ar * (taper - 0.35) ** 2
    # Sweep penalty: tip vortex structure degrades slightly with sweep
    sweep_factor = 1.0 - 0.0004 * sweep_deg
    return max(0.55, e_base * taper_factor * sweep_factor)


def drag_divergence_mach(sweep_deg, thickness_chord, cl):
    """
    Korn's equation for drag-divergence Mach with quarter-chord sweep.
    Sweep effectively reduces the normal Mach component, raising Mdd.
    """
    sweep_rad = math.radians(sweep_deg)
    cos_s = math.cos(sweep_rad)
    # kappa_A = 0.87 representative of supercritical-style sections
    mdd = (0.87 - thickness_chord / 2.0 - cl / 6.0) / (cos_s + 1e-9)
    return max(0.50, min(0.95, mdd))


def compute_cd_induced(ar, cl, taper, sweep_deg):
    """Induced drag from Prandtl's lifting line with Oswald efficiency."""
    e = oswald_factor(ar, taper, sweep_deg)
    return cl ** 2 / (math.pi * ar * e)


def compute_cd_wave(mach, mdd):
    """Cubic wave drag rise above drag-divergence Mach (abrupt onset)."""
    return 20.0 * max(0.0, mach - mdd) ** 3


def compute_cd_profile(thickness_chord, mach):
    """
    Profile drag from Torenbeek form factor and turbulent skin friction.
    Form factor grows strongly with thickness (1 + 2.7*t + 100*t^4).
    Compressibility mildly increases skin friction above M=0.40.
    Factor of 2 accounts for upper and lower airfoil surfaces.
    """
    ff = 1.0 + 2.7 * thickness_chord + 100.0 * thickness_chord ** 4
    cf = 0.0038   # Prandtl-Schlichting turbulent skin friction at Re~5M
    comp_factor = 1.0 + 0.08 * max(0.0, mach - 0.40) ** 2
    return 2.0 * cf * ff * comp_factor


def generate_row(ar, sweep_deg, taper, tc, cl_design, mach):
    # Compute noiseless drag components
    mdd = drag_divergence_mach(sweep_deg, tc, cl_design)
    cdi_clean = compute_cd_induced(ar, cl_design, taper, sweep_deg)
    cdw_clean = compute_cd_wave(mach, mdd)
    cdp_clean = compute_cd_profile(tc, mach)

    # Multiplicative noise: scales with signal magnitude (realistic for CFD uncertainty)
    cdi = cdi_clean * (1.0 + random.gauss(0, 0.05))        # +-5%
    cdw = max(0.0, cdw_clean * (1.0 + random.gauss(0, 0.10)))  # +-10%, clamp >= 0
    cdp = cdp_clean * (1.0 + random.gauss(0, 0.03))        # +-3%

    # L/D computed from noisy drag total (as real data would have it)
    cd_total = cdi + cdw + cdp
    ld = cl_design / max(cd_total, 1e-6)

    return (
        round(ar, 3),
        round(sweep_deg, 3),
        round(taper, 4),
        round(tc, 4),
        round(cl_design, 4),
        round(mach, 4),
        round(cdi, 6),
        round(cdw, 6),
        round(cdp, 6),
        round(ld, 3),
    )


# Latin Hypercube sampling: 200 rows, 6 independent parameter sweeps
n = 200
ars     = lhs_col(5.0,   15.0,  n)   # wing aspect ratio
sweeps  = lhs_col(0.0,   45.0,  n)   # quarter-chord sweep [deg]
tapers  = lhs_col(0.3,    1.0,  n)   # taper ratio (tip chord / root chord)
tcs     = lhs_col(0.08,   0.18, n)   # thickness-to-chord ratio
cl_vals = lhs_col(0.3,    1.2,  n)   # design lift coefficient
machs   = lhs_col(0.30,   0.85, n)   # cruise Mach number

rows = []
for i in range(n):
    rows.append(generate_row(ars[i], sweeps[i], tapers[i], tcs[i], cl_vals[i], machs[i]))

# Write CSV
output_path = 'sample_data/wing_design_space.csv'
with open(output_path, 'w') as f:
    f.write('aspect_ratio,sweep_deg,taper_ratio,thickness_chord,CL_design,mach,'
            'CD_induced,CD_wave,CD_profile,LD_ratio\n')
    for row in rows:
        f.write(','.join(str(v) for v in row) + '\n')

print(f'Generated {len(rows)} rows -> {output_path}')
print('Suggested usage:')
print('  Features (inputs):  aspect_ratio, sweep_deg, taper_ratio,')
print('                      thickness_chord, CL_design, mach')
print('  Targets (outputs):  CD_induced, CD_wave, CD_profile, LD_ratio')
print('  Recommended model:  GPR with ARD RBF or Matern kernel')
print('  Note: ARD will reveal AR+CL dominate CD_induced; mach+sweep+t/c dominate CD_wave')
