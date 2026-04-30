"""
Generate transonic drag divergence data for a NACA 0012 airfoil.

Physics modeled:
  - Prandtl-Glauert compressibility correction below M=0.80
  - Shock-induced lift reduction above M=0.80
  - Shevell drag-divergence Mach (Mdd) dependent on CL -- nonlinear coupling
  - Cubic wave drag rise: 20*(M-Mdd)^3 -- much steeper than the baseline quadratic
  - Mach tuck: strong nose-down CM that develops above M_crit=0.68

Run from the project root:
    conda activate base
    python sample_data/generate_transonic.py

Outputs: sample_data/transonic_naca0012.csv
"""

import math
import random

random.seed(42)


def prandtl_glauert(mach):
    """Prandtl-Glauert compressibility correction factor (subsonic)."""
    return 1.0 / math.sqrt(max(1e-6, 1.0 - mach ** 2))


def drag_divergence_mach(cl):
    """
    Shevell's equation for NACA 0012 drag-divergence Mach number.
    Higher CL -> lower Mdd -> earlier wave drag onset.
    Clamped to physically meaningful range [0.60, 0.83].
    """
    mdd = 0.83 - cl / 6.0
    return max(0.60, min(0.83, mdd))


def generate_row(alpha_deg, mach, reynolds):
    pg = prandtl_glauert(mach)
    re_factor = 1.0 + 0.020 * math.log10(reynolds / 1e6)

    # Lift: Prandtl-Glauert correction + Re correction
    # Above M=0.80, shocks limit lift (shock stall begins)
    cl_shock_factor = 1.0 - 0.4 * max(0.0, mach - 0.80)
    cl_noiseless = 0.1097 * alpha_deg * pg * re_factor * cl_shock_factor

    # Drag divergence Mach depends on noiseless CL (physical coupling)
    mdd = drag_divergence_mach(cl_noiseless)

    # Profile drag: skin friction decreases slightly with higher Re
    cd_min = 0.0055 - 0.0002 * math.log10(reynolds / 1e6)

    # Induced (lift-dependent) drag
    cd_lift = 0.009 * cl_noiseless ** 2

    # Wave drag: cubic rise above Mdd (steeper and more abrupt than quadratic)
    cd_wave = 20.0 * max(0.0, mach - mdd) ** 3

    cd_noiseless = cd_min + cd_lift + cd_wave

    # Pitching moment: Mach tuck -- shock on upper surface creates nose-down moment
    # Quadratic growth above M_crit=0.68, simulating aft shift of aerodynamic center
    cm_noiseless = -0.004 - 0.0005 * alpha_deg - 0.04 * max(0.0, mach - 0.68) ** 2

    # Add realistic measurement / CFD scatter
    cl = cl_noiseless + random.gauss(0, 0.006)
    cd = cd_noiseless + random.gauss(0, 0.0004)
    cm = cm_noiseless + random.gauss(0, 0.001)

    return (
        round(alpha_deg, 1),
        round(mach, 2),
        int(reynolds),
        round(cl, 4),
        round(cd, 5),
        round(cm, 4),
    )


# Full-factorial Design of Experiment
# 6 alpha x 8 mach x 2 reynolds = 96 rows
# Mach spacing is denser near the drag rise region (0.72-0.81)
alpha_levels = [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0]
mach_levels  = [0.55, 0.62, 0.68, 0.72, 0.75, 0.78, 0.81, 0.85]
re_levels    = [1_000_000, 3_000_000]

rows = []
for alpha in alpha_levels:
    for mach in mach_levels:
        for re in re_levels:
            rows.append(generate_row(alpha, mach, re))

# Write CSV
output_path = 'sample_data/transonic_naca0012.csv'
with open(output_path, 'w') as f:
    f.write('alpha_deg,mach,reynolds,CL,CD,CM\n')
    for row in rows:
        f.write(','.join(str(v) for v in row) + '\n')

print(f'Generated {len(rows)} rows -> {output_path}')
print('Suggested usage:')
print('  Features (inputs):  alpha_deg, mach, reynolds')
print('  Targets (outputs):  CL, CD, CM')
print('  Recommended model:  GPR with RBF or Matern kernel')
print('  Note: Linear regression will fail -- CD has a cubic nonlinearity at drag divergence')
