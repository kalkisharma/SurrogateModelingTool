"""
Generate NACA 4-digit airfoil family aerodynamic performance data.

Physics modeled (5 inputs, 3 outputs):
  - Zero-lift angle as function of camber magnitude and chordwise position
  - Prandtl-Glauert compressibility correction
  - Soft stall nonlinearity (CL cap above ~1.3 -- smooth, not hard clamp)
  - Parabolic drag polar referenced to design CL (minimum drag at design camber)
  - Profile drag: thickness increases drag; camber slightly reduces friction at design CL
  - Compressibility drag onset shifted by airfoil thickness (thicker = earlier onset)
  - Pitching moment: camber position determines nose-down moment magnitude

Sampling: Latin Hypercube-style (5 independent shuffles, 150 rows)
Reynolds number fixed at 1 million (not a column).

Run from the project root:
    conda activate base
    python sample_data/generate_naca4digit.py

Outputs: sample_data/naca4digit_family.csv
"""

import math
import random

random.seed(42)


def prandtl_glauert(mach):
    """Prandtl-Glauert compressibility correction factor (subsonic)."""
    return 1.0 / math.sqrt(max(1e-6, 1.0 - mach ** 2))


def lhs_col(lo, hi, n):
    """
    Latin Hypercube column: n evenly-spaced level centers across [lo, hi],
    independently shuffled.
    """
    step = (hi - lo) / n
    vals = [lo + step * (i + 0.5) for i in range(n)]
    random.shuffle(vals)
    return vals


def generate_row(max_camber, camber_pos, thickness, alpha_deg, mach):
    pg = prandtl_glauert(mach)

    # Zero-lift angle: depends on camber magnitude and chordwise position
    # At camber_pos=40%, alpha_L0 = -1.2 * max_camber exactly
    # The (1 + 0.5*(pos-40)/40) term shifts +/-25% across pos range [20%, 60%]
    alpha_L0 = -1.2 * max_camber * (1.0 + 0.5 * (camber_pos - 40.0) / 40.0)

    # Lift: thin airfoil theory + compressibility
    cl_lin = 0.110 * (alpha_deg - alpha_L0) * pg

    # Soft stall cap: reduces CL smoothly above ~1.3 without a hard cutoff
    # At CL=1.3: correction=0; at CL=1.5: CL becomes 1.5*(1-0.1)=1.35
    cl = cl_lin * (1.0 - max(0.0, cl_lin - 1.3) / 2.0)

    # Design lift coefficient: the CL at which this airfoil has minimum drag
    cl_minD = 0.12 * max_camber

    # Profile drag: base skin friction + thickness penalty + camber friction benefit
    # sqrt(max_camber + eps) avoids sqrt(0) while still giving zero camber at camber=0
    cd_min = (0.004
              + 0.00013 * thickness
              - 0.0001 * math.sqrt(max_camber + 1e-9))

    # Induced drag: parabolic polar referenced to design CL, not zero-lift
    cd_lift = 0.009 * (cl - cl_minD) ** 2

    # Compressibility drag: thicker airfoils have lower critical Mach
    # Threshold shifts from M=0.40 (t/c=0) down toward M=0.28 (t/c=24%)
    mach_threshold = 0.40 - thickness / 200.0
    cd_comp = 0.004 * max(0.0, mach - mach_threshold) ** 2

    cd = cd_min + cd_lift + cd_comp

    # Pitching moment about quarter chord (thin airfoil theory, camber-line integral)
    # Symmetric camber at 50% chord -> CM ~ 0
    # Forward camber (pos < 50%) -> negative (nose-down) moment
    cm = (-math.pi / 2.0 * (max_camber / 100.0) * (1.0 - 2.0 * camber_pos / 100.0)
          - 0.0004 * alpha_deg)

    # Add CFD / wind-tunnel scatter
    cl += random.gauss(0, 0.010)
    cd += random.gauss(0, 0.0003)
    cm += random.gauss(0, 0.001)

    return (
        round(max_camber, 3),
        round(camber_pos, 3),
        round(thickness, 3),
        round(alpha_deg, 3),
        round(mach, 4),
        round(cl, 4),
        round(cd, 5),
        round(cm, 4),
    )


# Latin Hypercube sampling: 150 rows, 5 independent parameter sweeps
n = 150
cambers   = lhs_col(0.0,   9.0,  n)   # max camber [% chord]
positions = lhs_col(20.0, 60.0,  n)   # chordwise position of max camber [% chord]
thicks    = lhs_col(6.0,  24.0,  n)   # max thickness [% chord]
alphas    = lhs_col(-5.0, 12.0,  n)   # angle of attack [deg]
machs     = lhs_col(0.1,   0.5,  n)   # Mach number

rows = []
for i in range(n):
    rows.append(generate_row(cambers[i], positions[i], thicks[i], alphas[i], machs[i]))

# Write CSV
output_path = 'sample_data/naca4digit_family.csv'
with open(output_path, 'w') as f:
    f.write('max_camber_pct,camber_pos_pct,thickness_pct,alpha_deg,mach,CL,CD,CM\n')
    for row in rows:
        f.write(','.join(str(v) for v in row) + '\n')

print(f'Generated {len(rows)} rows -> {output_path}')
print('Suggested usage:')
print('  Features (inputs):  max_camber_pct, camber_pos_pct, thickness_pct, alpha_deg, mach')
print('  Targets (outputs):  CL, CD, CM')
print('  Recommended model:  GPR with ARD Matern kernel')
print('  Note: ARD will show camber drives CL/CM; thickness drives CD; alpha has broad influence')
