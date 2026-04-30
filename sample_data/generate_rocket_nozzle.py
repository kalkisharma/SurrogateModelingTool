"""
Generate rocket nozzle performance data using isentropic flow theory.

Physics modeled (5 inputs, 3 outputs):
  - Supersonic exit Mach from area ratio: Newton-Raphson solve of isentropic A/A* relation
  - Isentropic exit pressure and temperature from chamber stagnation state
  - Vacuum thrust coefficient (Cf_vac) from integrated momentum and pressure thrust
  - Actual Cf corrected for ambient back-pressure on nozzle exit area
  - Characteristic exhaust velocity (c*) from propellant energy content
  - Specific impulse: Isp = Cf * c* / g0
  - Exit velocity: Ve = Me * sqrt(gamma * R * Te)

Sampling:
  - chamber_pressure, area_ratio, chamber_temp, gamma: Latin Hypercube (linear)
  - ambient_pressure: log-uniform [0.001, 1.0] bar -- covers sea level to near-vacuum

R_specific = 400 J/(kg*K): representative for hydrocarbon/LOX propellants

Run from the project root:
    conda activate base
    python sample_data/generate_rocket_nozzle.py

Outputs: sample_data/rocket_nozzle.csv
"""

import math
import random

random.seed(42)

# Specific gas constant representative of hydrocarbon rocket propellants [J/(kg*K)]
R_SPECIFIC = 400.0


def lhs_col(lo, hi, n):
    """
    Latin Hypercube column: n evenly-spaced level centers across [lo, hi],
    independently shuffled.
    """
    step = (hi - lo) / n
    vals = [lo + step * (i + 0.5) for i in range(n)]
    random.shuffle(vals)
    return vals


def exit_mach(area_ratio, gamma, tol=1e-8):
    """
    Newton-Raphson solve for supersonic exit Mach from the isentropic
    area-Mach relation:

        A/A* = (1/Me) * ((2/(g+1)) * (1 + (g-1)/2 * Me^2))^((g+1)/(2*(g-1)))

    Initial guess Me=2.0 places the solver on the supersonic branch.
    The max(1.01, Me) clamp prevents drift onto the subsonic branch.
    """
    exp = (gamma + 1.0) / (2.0 * (gamma - 1.0))

    def area_mach_func(me):
        t = 1.0 + (gamma - 1.0) / 2.0 * me ** 2
        return (1.0 / me) * (2.0 / (gamma + 1.0) * t) ** exp - area_ratio

    me = 2.0   # supersonic initial guess
    for _ in range(50):
        f = area_mach_func(me)
        dme = 1e-6
        df = (area_mach_func(me + dme) - f) / dme
        me -= f / (df + 1e-30)
        me = max(1.01, me)   # stay on supersonic branch
        if abs(f) < tol:
            break
    return me


def nozzle_performance(pc_bar, area_ratio, pa_bar, tc_k, gamma):
    """
    Compute nozzle performance from chamber conditions and nozzle geometry.

    Returns:
        Cf   -- thrust coefficient (dimensionless)
        Isp  -- specific impulse [s]
        Ve   -- nozzle exit velocity [m/s]
    """
    pc = pc_bar * 1e5   # chamber total pressure [Pa]
    pa = pa_bar * 1e5   # ambient static pressure [Pa]

    me = exit_mach(area_ratio, gamma)

    # Isentropic stagnation-to-static ratios at nozzle exit
    t_ratio = 1.0 + (gamma - 1.0) / 2.0 * me ** 2
    pe = pc / t_ratio ** (gamma / (gamma - 1.0))   # exit static pressure [Pa]
    te = tc_k / t_ratio                             # exit static temperature [K]

    # Exit velocity from isentropic relations
    ve = me * math.sqrt(gamma * R_SPECIFIC * te)

    # Vacuum thrust coefficient (momentum thrust + pressure thrust, normalized by Pc*At)
    cf_vac = math.sqrt(
        2.0 * gamma ** 2 / (gamma - 1.0)
        * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (gamma - 1.0))
        * (1.0 - (pe / pc) ** ((gamma - 1.0) / gamma))
    ) + (pe / pc) * area_ratio

    # Ambient back-pressure subtracts from effective thrust on exit plane
    cf = cf_vac - (pa / pc) * area_ratio
    cf = max(0.1, cf)   # physical lower bound

    # Characteristic exhaust velocity: depends only on propellant and chamber conditions
    c_star = (math.sqrt(R_SPECIFIC * tc_k / gamma)
              * ((gamma + 1.0) / 2.0) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0))))

    # Specific impulse [s]
    isp = cf * c_star / 9.81

    return round(cf, 4), round(isp, 1), round(ve, 1)


def generate_row(pc_bar, area_ratio, pa_bar, tc_k, gamma):
    cf, isp, ve = nozzle_performance(pc_bar, area_ratio, pa_bar, tc_k, gamma)

    # Add measurement / model uncertainty
    cf  = max(0.0, cf  + random.gauss(0, 0.008))   # +-0.8% of typical Cf~1.5
    isp = max(0.0, isp + random.gauss(0, 1.5))     # +-1.5 s
    ve  = max(0.0, ve  + random.gauss(0, 5.0))     # +-5 m/s

    return (
        round(pc_bar, 2),
        round(area_ratio, 3),
        round(pa_bar, 5),
        round(tc_k, 1),
        round(gamma, 4),
        round(cf, 4),
        round(isp, 1),
        round(ve, 1),
    )


# Sampling: 120 rows, 5 parameters
n = 120

# 4 parameters: Latin Hypercube (linear spacing)
pc_vals    = lhs_col(20.0,   200.0,  n)   # chamber pressure [bar]
ar_vals    = lhs_col(5.0,    50.0,   n)   # nozzle area ratio (exit/throat)
tc_vals    = lhs_col(2500.0, 3500.0, n)   # chamber temperature [K]
gamma_vals = lhs_col(1.15,   1.35,   n)   # ratio of specific heats

# Ambient pressure: log-uniform [0.001, 1.0] bar
# log10 range [-3, 0] gives equal density per decade (vacuum to sea level)
amb_vals = [10 ** random.uniform(math.log10(0.001), 0.0) for _ in range(n)]

rows = []
for i in range(n):
    rows.append(generate_row(pc_vals[i], ar_vals[i], amb_vals[i], tc_vals[i], gamma_vals[i]))

# Write CSV
output_path = 'sample_data/rocket_nozzle.csv'
with open(output_path, 'w') as f:
    f.write('chamber_pressure_bar,area_ratio,ambient_pressure_bar,'
            'chamber_temp_K,gamma,thrust_coeff_Cf,specific_impulse_s,exit_velocity_ms\n')
    for row in rows:
        f.write(','.join(str(v) for v in row) + '\n')

print(f'Generated {len(rows)} rows -> {output_path}')
print('Suggested usage:')
print('  Features (inputs):  chamber_pressure_bar, area_ratio, ambient_pressure_bar,')
print('                      chamber_temp_K, gamma')
print('  Targets (outputs):  thrust_coeff_Cf, specific_impulse_s, exit_velocity_ms')
print('  Recommended model:  GPR with RBF or Matern kernel')
print('  Note: consider log-transforming ambient_pressure_bar before training')
