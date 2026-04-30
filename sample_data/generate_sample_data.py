"""
Generate sample CFD aerodynamic data for the Surrogate Modeling Tool.

Simulates a NACA 0012 airfoil polar using semi-empirical relations:
  - Prandtl-Glauert compressibility correction (subsonic)
  - Parabolic drag polar
  - Wave drag onset above Mach 0.4
  - Reynolds number effect on skin friction drag

Run from the project root:
    conda activate base
    python sample_data/generate_sample_data.py

Outputs: sample_data/naca0012_airfoil.csv
"""

import math
import random

random.seed(42)


def prandtl_glauert(mach):
    return 1.0 / math.sqrt(max(1e-6, 1.0 - mach ** 2))


def generate_row(alpha_deg, mach, reynolds):
    pg = prandtl_glauert(mach)
    re_factor = 1.0 + 0.020 * math.log10(reynolds / 1e6)

    # Lift: thin airfoil theory + compressibility + Re correction
    cl_inc = 0.1097 * alpha_deg          # 2π/57.3 * alpha (per degree)
    cl = cl_inc * pg * re_factor

    # Drag: parabolic polar + skin friction Re effect + wave drag
    cd_min = 0.006 - 0.0003 * math.log10(reynolds / 5e5)
    cd_lift = 0.009 * cl ** 2
    cd_wave = 0.06 * max(0.0, mach - 0.40) ** 2
    cd = cd_min + cd_lift + cd_wave

    # Pitching moment about quarter chord (symmetric airfoil ≈ constant)
    cm = -0.004 - 0.0005 * alpha_deg

    # Add realistic measurement noise (±1.5% CL, ±3% CD, ±5% CM)
    cl += random.gauss(0, 0.008)
    cd += random.gauss(0, 0.0003)
    cm += random.gauss(0, 0.0008)

    return round(alpha_deg, 1), round(mach, 2), int(reynolds), \
           round(cl, 4), round(cd, 5), round(cm, 4)


# Structured Design of Experiment
# 8 alpha × 4 mach × 3 Reynolds = 96 rows
alpha_levels = [-5.0, -2.0, 0.0, 2.0, 5.0, 8.0, 11.0, 14.0]
mach_levels  = [0.10, 0.20, 0.35, 0.50]
re_levels    = [500_000, 1_000_000, 2_000_000]

rows = []
for alpha in alpha_levels:
    for mach in mach_levels:
        for re in re_levels:
            rows.append(generate_row(alpha, mach, re))

# Write CSV
output_path = 'sample_data/naca0012_airfoil.csv'
with open(output_path, 'w') as f:
    f.write('alpha_deg,mach,reynolds,CL,CD,CM\n')
    for row in rows:
        f.write(','.join(str(v) for v in row) + '\n')

print(f'Generated {len(rows)} rows -> {output_path}')
print('Suggested usage:')
print('  Features (inputs):  alpha_deg, mach, reynolds')
print('  Targets (outputs):  CL, CD, CM')
print('  Recommended model:  GPR with RBF or Matérn kernel')
