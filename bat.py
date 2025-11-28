# final_cumulative_model.py
# Python 3.8+ recommended

from decimal import Decimal, getcontext
import csv
from tqdm import tqdm

# ------------------------
# Precision for Decimal
getcontext().prec = 120
# ------------------------
LN2 = Decimal(2).ln()
NEG_INF = Decimal('-Infinity')

# ------------------------
# PARAMETERS (edit these)
params = {
    'T': 100000,  # generations

    # initial exclusive-category counts (Decimals)
    'init_counts': {
        'G_only': Decimal('10'),
        'S_only': Decimal('10'),
        'H_only': Decimal('10'),
        'GS':     Decimal('0'),
        'GH':     Decimal('0'),
        'SH':     Decimal('0'),
        'GSH':    Decimal('0'),
        'N':      Decimal('1000000'),
    },

    # base reproduction and epsilons
    'C_N_base': Decimal('1'),       # base reproduction for normals
    'eps_G_base': Decimal('0.00000001'),
    'eps_S_base': Decimal('0.00000001'),
    'eps_H_base': Decimal('0.00000001'),

    # base mortality (per-category bases)
    'delta_N_base': Decimal('0.5'),
    'delta_G_base': Decimal('0.5'),
    'delta_S_base': Decimal('0.5'),
    'delta_H_base': Decimal('0.5'),

    # exponential multipliers
    'a_repro': Decimal('0.1'),   # M(n) = 1 + a*(1 - exp(-k*n))
    'k_repro': Decimal('0.000001'),
    'h_mort':  Decimal('0.000001'),  # D(n) = exp(-h*n)

    # whether to use double-epsilon factor in C (C = C_N_base + 2*eps)
    'double_eps_in_C': True,

    # output
    'output_csv': 'bat_population_summary.csv',
    'LOG_OUT_THRESHOLD': Decimal('230.25850929940456'),  # ln(1e100)
}
# ------------------------

# ---------- helper functions ----------
def safe_log(x: Decimal) -> Decimal:
    if x <= 0:
        return NEG_INF
    return x.ln()

def safe_exp(l: Decimal) -> Decimal:
    if l == NEG_INF:
        return Decimal(0)
    return l.exp()

def logsumexp(logs):
    logs = [l for l in logs]
    if not logs:
        return NEG_INF
    m = max(logs)
    if m == NEG_INF:
        return NEG_INF
    s = Decimal(0)
    for l in logs:
        s += (l - m).exp()
    return m + s.ln()

def log_Fn(n: int) -> Decimal:
    # log(2^n - 1) = n*ln2 + ln(1 - 2^{-n})
    neg = - (Decimal(n) * LN2)
    r = safe_exp(neg)   # 2^{-n}
    if r == 0:
        return Decimal(n) * LN2
    one_minus_r = Decimal(1) - r
    return Decimal(n) * LN2 + safe_log(one_minus_r)

# log(1 - e^{logx}) safe (logx <= 0)
def log_one_minus_exp(logx: Decimal) -> Decimal:
    if logx == NEG_INF:
        return Decimal(0)  # log(1)
    val = safe_exp(logx)
    one_minus = Decimal(1) - val
    if one_minus <= 0:
        return NEG_INF
    return safe_log(one_minus)

# ---------- Pascal-based log p formulas ----------
def compute_log_p_single(logPG, logPN, n):
    logFn = log_Fn(n)
    termsA = []
    if logPG != NEG_INF:
        termsA.append(2 * logPG)  # P_G^2
    if logPG != NEG_INF and logPN != NEG_INF:
        termsA.append(logPG + logPN + logFn)
    logA = logsumexp(termsA) if termsA else NEG_INF
    logB = 2 * logPN if logPN != NEG_INF else NEG_INF
    logDen = logsumexp([logA, logB])
    if logDen == NEG_INF:
        return NEG_INF
    return logA - logDen

def compute_log_p_pair(logPG, logPS, logPN, n):
    logFn = log_Fn(n)
    termsC = []
    if logPG != NEG_INF and logPS != NEG_INF:
        termsC.append(logPG + logPS)
        if logPN != NEG_INF:
            termsC.append(logPG + logPS + logPN + logFn)
    logC = logsumexp(termsC) if termsC else NEG_INF
    logD = 2 * logPN if logPN != NEG_INF else NEG_INF
    logDen = logsumexp([logC, logD])
    if logDen == NEG_INF:
        return NEG_INF
    return logC - logDen

def compute_log_p_triple(logPG, logPS, logPH, logPN, n):
    logFn = log_Fn(n)
    termsT = []
    if logPG != NEG_INF and logPS != NEG_INF and logPH != NEG_INF:
        termsT.append(logPG + logPS + logPH)
        if logPN != NEG_INF:
            termsT.append(logPG + logPS + logPH + logPN + logFn)
    logT = logsumexp(termsT) if termsT else NEG_INF
    logD = 2 * logPN if logPN != NEG_INF else NEG_INF
    logDen = logsumexp([logT, logD])
    if logDen == NEG_INF:
        return NEG_INF
    return logT - logDen

# ---------- mapping and C computation ----------
def compute_marginal_logs_from_exclusive(log_counts):
    # marginals numerator (log) for carriers of each allele
    logG_components = [log_counts.get(k, NEG_INF) for k in ('G_only','GS','GH','GSH') if log_counts.get(k, NEG_INF) != NEG_INF]
    logS_components = [log_counts.get(k, NEG_INF) for k in ('S_only','GS','SH','GSH') if log_counts.get(k, NEG_INF) != NEG_INF]
    logH_components = [log_counts.get(k, NEG_INF) for k in ('H_only','GH','SH','GSH') if log_counts.get(k, NEG_INF) != NEG_INF]
    logG_num = logsumexp(logG_components) if logG_components else NEG_INF
    logS_num = logsumexp(logS_components) if logS_components else NEG_INF
    logH_num = logsumexp(logH_components) if logH_components else NEG_INF
    return logG_num, logS_num, logH_num

def compute_C_map(C_N_base, epsG, epsS, epsH, double_eps=True, mode='additive'):
    # produce C for exclusive categories using additive mapping and double epsilon factor if requested
    C = {}
    if mode != 'additive':
        raise NotImplementedError("Only 'additive' implemented")
    # single-locus: C_N_base + (2*eps) if double_eps True else + eps
    mul = Decimal(2) if double_eps else Decimal(1)
    C['G_only'] = C_N_base + mul * epsG
    C['S_only'] = C_N_base + mul * epsS
    C['H_only'] = C_N_base + mul * epsH
    C['GS']     = C_N_base + mul * (epsG + epsS)
    C['GH']     = C_N_base + mul * (epsG + epsH)
    C['SH']     = C_N_base + mul * (epsS + epsH)
    C['GSH']    = C_N_base + mul * (epsG + epsS + epsH)
    C['N']      = C_N_base
    return C

# ---------- initialization ----------
categories = ['G_only','S_only','H_only','GS','GH','SH','GSH','N']
log_counts = {}
for cat in categories:
    val = params['init_counts'].get(cat, Decimal('0'))
    log_counts[cat] = safe_log(val)

history = []

# helper: category delta mapping (min of relevant base deltas * D_n)
def category_delta(cat, delta_G_base, delta_S_base, delta_H_base, delta_N_base, D_n):
    deltas = []
    if cat in ('G_only','GS','GH','GSH'):
        deltas.append(delta_G_base)
    if cat in ('S_only','GS','SH','GSH'):
        deltas.append(delta_S_base)
    if cat in ('H_only','GH','SH','GSH'):
        deltas.append(delta_H_base)
    if cat == 'N':
        deltas.append(delta_N_base)
    base = min(deltas) if deltas else delta_N_base
    return base * D_n

# main loop
T = int(params['T'])
LOG_OUT_THRESHOLD = params['LOG_OUT_THRESHOLD']

for n in tqdm(range(1, T+1), desc="Generations", unit="gen"):
    # dynamic multipliers
    a = params['a_repro']; k = params['k_repro']; h = params['h_mort']
    M_n = Decimal(1) + a * (Decimal(1) - (-(k * Decimal(n))).exp())   # multiplicative factor for eps
    D_n = (-(h * Decimal(n))).exp()                                   # mortality multiplier

    # eps^{(n)} = eps_base * M_n
    epsG_n = params['eps_G_base'] * M_n
    epsS_n = params['eps_S_base'] * M_n
    epsH_n = params['eps_H_base'] * M_n

    # survivors (log)
    log_survivor = {}
    for cat in categories:
        log_count_prev = log_counts.get(cat, NEG_INF)
        delta_cat_n = category_delta(cat,
            params['delta_G_base'], params['delta_S_base'], params['delta_H_base'], params['delta_N_base'], D_n)
        s_cat_n = Decimal(1) - delta_cat_n
        if s_cat_n <= 0 or log_count_prev == NEG_INF:
            log_survivor[cat] = NEG_INF
        else:
            log_survivor[cat] = log_count_prev + safe_log(s_cat_n)

    Bat_alive_log = logsumexp([log_survivor[cat] for cat in categories])
    if Bat_alive_log == NEG_INF:
        print(f"Extinction at generation {n}")
        break

    # marginals after mortality (log)
    logG_num, logS_num, logH_num = compute_marginal_logs_from_exclusive(log_survivor)
    logP_G = logG_num - Bat_alive_log if logG_num != NEG_INF else NEG_INF
    logP_S = logS_num - Bat_alive_log if logS_num != NEG_INF else NEG_INF
    logP_H = logH_num - Bat_alive_log if logH_num != NEG_INF else NEG_INF
    logP_N = log_survivor['N'] - Bat_alive_log if log_survivor['N'] != NEG_INF else NEG_INF

    # compute joint probabilities (log) using Pascal formulas
    log_pG = compute_log_p_single(logP_G, logP_N, n)
    log_pS = compute_log_p_single(logP_S, logP_N, n)
    log_pH = compute_log_p_single(logP_H, logP_N, n)

    if n == 2:
        log_pGS = log_pGH = log_pSH = NEG_INF
        log_pGSH = NEG_INF
    else:
        log_pGS = compute_log_p_pair(logP_G, logP_S, logP_N, n)
        log_pGH = compute_log_p_pair(logP_G, logP_H, logP_N, n)
        log_pSH = compute_log_p_pair(logP_S, logP_H, logP_N, n)
        log_pGSH = compute_log_p_triple(logP_G, logP_S, logP_H, logP_N, n)

    # p_N as Pascal-single for P_N against P_other
    logP_other = log_one_minus_exp(logP_N)
    log_pN = compute_log_p_single(logP_N, logP_other, n)

    # convert joint log p's to Decimal probabilities (safe)
    def log_to_prob(lg):
        if lg == NEG_INF:
            return Decimal('0')
        val = safe_exp(lg)
        # clamp tiny negatives
        if val < 0:
            return Decimal('0')
        return val

    pG = log_to_prob(log_pG)
    pS = log_to_prob(log_pS)
    pH = log_to_prob(log_pH)
    pGS = log_to_prob(log_pGS)
    pGH = log_to_prob(log_pGH)
    pSH = log_to_prob(log_pSH)
    pGSH = log_to_prob(log_pGSH)
    pN_joint = log_to_prob(log_pN)

    # compute exclusive probabilities via inclusion-exclusion (normal arithmetic)
    # exactly G only = pG - pGS - pGH + pGSH
    G_only_p = pG - pGS - pGH + pGSH
    S_only_p = pS - pGS - pSH + pGSH
    H_only_p = pH - pGH - pSH + pGSH
    GS_only_p = pGS - pGSH
    GH_only_p = pGH - pGSH
    SH_only_p = pSH - pGSH
    GSH_only_p = pGSH
    N_p = pN_joint  # joint p_N is already probability of being normal (exclusive)

    # fix tiny negative numerical artifacts
    def clamp_zero(x):
        return x if x > Decimal('0') else Decimal('0')
    G_only_p = clamp_zero(G_only_p)
    S_only_p = clamp_zero(S_only_p)
    H_only_p = clamp_zero(H_only_p)
    GS_only_p = clamp_zero(GS_only_p)
    GH_only_p = clamp_zero(GH_only_p)
    SH_only_p = clamp_zero(SH_only_p)
    GSH_only_p = clamp_zero(GSH_only_p)
    N_p = clamp_zero(N_p)

    # Normalize exclusives so sum = 1 (to correct rounding)
    p_list = [G_only_p, S_only_p, H_only_p, GS_only_p, GH_only_p, SH_only_p, GSH_only_p, N_p]
    sum_p = sum(p_list)
    if sum_p == 0:
        # emergency fallback: put all mass into N
        G_only_p = S_only_p = H_only_p = GS_only_p = GH_only_p = SH_only_p = GSH_only_p = Decimal('0')
        N_p = Decimal('1')
        sum_p = Decimal('1')
    # normalize
    G_only_p /= sum_p
    S_only_p /= sum_p
    H_only_p /= sum_p
    GS_only_p /= sum_p
    GH_only_p /= sum_p
    SH_only_p /= sum_p
    GSH_only_p /= sum_p
    N_p /= sum_p

    # compute offspring counts: Off_X = p_excl * Bat_alive * C_X^{(n)}
    # build C_map with eps_n and double_eps factor
    if params['double_eps_in_C']:
        # C = C_N_base + 2*eps (per allele)
        C_map_n = compute_C_map(params['C_N_base'], epsG_n, epsS_n, epsH_n, double_eps=True)
    else:
        C_map_n = compute_C_map(params['C_N_base'], epsG_n, epsS_n, epsH_n, double_eps=False)

    Bat_alive = safe_exp(Bat_alive_log)
    # Offspring counts (Decimal)
    Off = {}
    Off['G_only'] = G_only_p * Bat_alive * C_map_n['G_only']
    Off['S_only'] = S_only_p * Bat_alive * C_map_n['S_only']
    Off['H_only'] = H_only_p * Bat_alive * C_map_n['H_only']
    Off['GS']     = GS_only_p * Bat_alive * C_map_n['GS']
    Off['GH']     = GH_only_p * Bat_alive * C_map_n['GH']
    Off['SH']     = SH_only_p * Bat_alive * C_map_n['SH']
    Off['GSH']    = GSH_only_p * Bat_alive * C_map_n['GSH']
    Off['N']      = N_p * Bat_alive * C_map_n['N']

    total_offspring = sum(Off.values())

    # Update counts: X_next = survivor + offspring
    counts_next = {}
    for cat in categories:
        surv = safe_exp(log_survivor.get(cat, NEG_INF))
        counts_next[cat] = surv + Off.get(cat, Decimal('0'))

    # Bat_total after update
    Bat_total = sum(counts_next.values())

    # Prepare output (only: counts, frequencies (exclusive), Bat_total, Bat_alive, total_offspring)
    def safe_render_value(x):
        # if x is zero
        if x == 0:
            return '0'
        # take log if too big
        logx = safe_log(x)
        if logx == NEG_INF:
            return '0'
        if logx > params['LOG_OUT_THRESHOLD']:
            return f'log:{str(logx)}'
        # else return decimal as string
        return format(x, 'f')

    row = {}
    row['gen'] = n
    row['Bat_total'] = safe_render_value(Bat_total)
    row['Bat_alive'] = safe_render_value(Bat_alive)
    row['Total_offspring'] = safe_render_value(total_offspring)

    # counts and frequencies (exclusive)
    excl_counts = {
        'G_only': counts_next['G_only'],
        'S_only': counts_next['S_only'],
        'H_only': counts_next['H_only'],
        'GS':     counts_next['GS'],
        'GH':     counts_next['GH'],
        'SH':     counts_next['SH'],
        'GSH':    counts_next['GSH'],
        'N':      counts_next['N'],
    }
    # frequencies as decimal fraction (count / Bat_total)
    freqs = {}
    if Bat_total == 0:
        for cat in excl_counts:
            freqs[cat] = Decimal('0')
    else:
        for cat in excl_counts:
            freqs[cat] = excl_counts[cat] / Bat_total

    # populate row
    for cat in ['G_only','S_only','H_only','GS','GH','SH','GSH','N']:
        row[f'Count_{cat}'] = safe_render_value(excl_counts[cat])
        # frequency as decimal string
        row[f'Freq_{cat}'] = format(freqs[cat], 'f')

    history.append(row)

    # set log_counts for next generation: convert counts_next to logs safely
    log_counts = {}
    for cat in categories:
        log_counts[cat] = safe_log(counts_next[cat]) if counts_next[cat] > 0 else NEG_INF

# end generations

# write CSV with selected columns
if history:
    fieldnames = ['gen','Bat_total','Bat_alive','Total_offspring']
    for cat in ['G_only','S_only','H_only','GS','GH','SH','GSH','N']:
        fieldnames.append(f'Count_{cat}')
        fieldnames.append(f'Freq_{cat}')
    with open(params['output_csv'], 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in history:
            writer.writerow(r)
    print("Simulation complete. CSV saved to", params['output_csv'])
else:
    print("No data (extinction).")
