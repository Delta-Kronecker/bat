import csv
from decimal import Decimal, getcontext
from tqdm import tqdm

# تنظیم دقت بالا برای محاسبات Decimal
getcontext().prec = 15

# ------------------------------------------------------------
# 0. پارامترهای ورودی مدل (همه Decimal)
# ------------------------------------------------------------

NUM_GENERATIONS = 10000
SAVE_INTERVAL = 100

C0 = Decimal('1.0')
LAMBDA = Decimal('0.5')

LOCI = ['G', 'S', 'H']
INITIAL_MU0 = {'G': Decimal('1.0'), 'S': Decimal('1.0'), 'H': Decimal('1.0')}
FIXED_EPSILON = {'G': Decimal('0.00000002'), 'S': Decimal('0.00000001'), 'H': Decimal('0.00000003')}
INITIAL_ALPHA = {'G': Decimal('0.000002'), 'S': Decimal('0.000001'), 'H': Decimal('0.000003')}

OUTPUT_FILENAME = 'co_evolution_GSH_genotype_frequency.csv'


# ------------------------------------------------------------
# 1. آماده‌سازی و توابع کمکی
# ------------------------------------------------------------

def get_all_genotypes():
    genotypes_state_G = ['GG', 'Gg', 'gg']
    genotypes_state_S = ['SS', 'Ss', 'ss']
    genotypes_state_H = ['HH', 'Hh', 'hh']

    all_genotypes = []

    for g in genotypes_state_G:
        for s in genotypes_state_S:
            for h in genotypes_state_H:
                all_genotypes.append(g + s + h)

    def has_dominant_allele(genotype_str, locus_char):
        if locus_char == 'G':
            state = genotype_str[0:2]
        elif locus_char == 'S':
            state = genotype_str[2:4]
        elif locus_char == 'H':
            state = genotype_str[4:6]

        if state in ['gg', 'ss', 'hh']:
            return 0
        else:
            return 1

    return all_genotypes, has_dominant_allele


ALL_GENOTYPES, HAS_DOMINANT_ALLELE = get_all_genotypes()


def get_initial_population():
    # مقادیر اولیه ارائه شده توسط کاربر، تبدیل به Decimal
    N_i_float = {
        'GGSSHH': 0, 'GGSSHh': 0, 'GGSShh': 0,
        'GGSsHH': 0, 'GGSsHh': 0, 'GGSshh': 0,
        'GGssHH': 0, 'GGssHh': 0, 'GGsshh': 0,

        'GgSSHH': 0, 'GgSSHh': 0, 'GgSShh': 0,
        'GgSsHH': 0, 'GgSsHh': 0, 'GgSshh': 0,
        'GgssHH': 0, 'GgssHh': 0, 'Ggsshh': 1,

        'ggSSHH': 0, 'ggSSHh': 0, 'ggSShh': 0,
        'ggSsHH': 0, 'ggSsHh': 0, 'ggSshh': 1,
        'ggssHH': 0, 'ggssHh': 1, 'ggsshh': 99999999997
    }

    N_i = {g: Decimal(str(count)) for g, count in N_i_float.items()}

    return N_i


# ------------------------------------------------------------
# 2. تعریف کلاس SimulationState (با محاسبه فراوانی)
# ------------------------------------------------------------

class SimulationState:
    def __init__(self, generation, N_total):
        self.generation = generation
        self.N_i = {}
        self.N_Total = N_total
        self.mu0 = {}
        self.alpha = {}
        self.mu_bar = {}
        self.p = {}
        self.C_bar = Decimal('0.0')
        self.f_i = {}

    def calculate_frequencies(self):
        if self.N_Total != Decimal('0'):
            for g in ALL_GENOTYPES:
                self.f_i[g] = self.N_i.get(g, Decimal('0')) / self.N_Total
        else:
            for g in ALL_GENOTYPES:
                self.f_i[g] = Decimal('0')

    def to_dict(self):
        self.calculate_frequencies()
        data = {'Generation': self.generation, 'N_Total': str(self.N_Total)}

        for X in LOCI:
            data[f'mu0_{X}'] = str(self.mu0.get(X, Decimal('0.0')))
            data[f'alpha_{X}'] = str(self.alpha.get(X, Decimal('0.0')))
            data[f'mu_bar_{X}'] = str(self.mu_bar.get(X, Decimal('0.0')))
            data[f'p_{X}'] = str(self.p.get(X, Decimal('0.0')))

        data['C_bar'] = str(self.C_bar)

        for g in ALL_GENOTYPES:
            data[f'f_{g}'] = str(self.f_i.get(g, Decimal('0.0')))

        return data


# ------------------------------------------------------------
# 3. توابع فازهای مدل
# ------------------------------------------------------------

def phase_1_analysis(current_state, previous_mu_bar):
    new_alpha = {}

    for X in LOCI:
        if current_state.generation == 0:
            mu0_t = INITIAL_MU0[X]
            mu_bar_prev = INITIAL_MU0[X]
        else:
            mu0_t = previous_mu_bar[X]
            mu_bar_prev = previous_mu_bar[X]

        current_state.mu0[X] = mu0_t

        N_has_X = Decimal('0')
        for genotype_str, count in current_state.N_i.items():
            N_has_X += count * Decimal(HAS_DOMINANT_ALLELE(genotype_str, X))
        N_no_X = current_state.N_Total - N_has_X

        mu_bar_t = (N_no_X * mu0_t + N_has_X * (mu0_t + FIXED_EPSILON[X])) / current_state.N_Total
        current_state.mu_bar[X] = mu_bar_t

        if mu_bar_prev == Decimal('0'):
            progress_ratio = Decimal('1.0')
        else:
            progress_ratio = mu_bar_t / mu_bar_prev

        alpha_t = current_state.alpha[X]
        new_alpha[X] = alpha_t * progress_ratio

    return new_alpha


def phase_2_reproduction(current_state):
    p_t = {}

    for X in LOCI:
        A_X = Decimal('0')
        T_X = Decimal('2') * current_state.N_Total

        for genotype_str, count in current_state.N_i.items():
            allele_count = Decimal('0')
            if X == 'G':
                state = genotype_str[0:2]
            elif X == 'S':
                state = genotype_str[2:4]
            elif X == 'H':
                state = genotype_str[4:6]

            if state in ['GG', 'SS', 'HH']:
                allele_count = Decimal('2')
            elif state in ['Gg', 'Ss', 'Hh']:
                allele_count = Decimal('1')

            A_X += count * allele_count

        p_t[X] = A_X / T_X
        current_state.p[X] = p_t[X]

    N_i_offspring = {}

    for genotype_str in ALL_GENOTYPES:
        f_i = Decimal('1.0')

        for X in LOCI:
            p = p_t[X]
            q = Decimal('1') - p

            if X == 'G':
                state = genotype_str[0:2]
            elif X == 'S':
                state = genotype_str[2:4]
            elif X == 'H':
                state = genotype_str[4:6]

            if state in ['GG', 'SS', 'HH']:
                f_i *= (p ** Decimal('2'))
            elif state in ['Gg', 'Ss', 'Hh']:
                f_i *= (Decimal('2') * p * q)
            elif state in ['gg', 'ss', 'hh']:
                f_i *= (q ** Decimal('2'))

        N_i_offspring[genotype_str] = current_state.N_Total * f_i

    return N_i_offspring


def phase_3_selection(current_state, N_i_offspring):
    C_i = {}

    for genotype_str in ALL_GENOTYPES:
        sum_alpha = Decimal('0.0')
        for X in LOCI:
            sum_alpha += current_state.alpha[X] * Decimal(HAS_DOMINANT_ALLELE(genotype_str, X))

        C_i[genotype_str] = C0 + sum_alpha

    sum_weighted_C = Decimal('0.0')
    for genotype_str, count in N_i_offspring.items():
        sum_weighted_C += count * C_i[genotype_str]

    C_bar = sum_weighted_C / current_state.N_Total
    current_state.C_bar = C_bar

    N_i_after_selection = {}
    for genotype_str, count_offspring in N_i_offspring.items():
        N_i_after_selection[genotype_str] = count_offspring * (C_i[genotype_str] / C_bar)

    return N_i_after_selection


def phase_4_dynamics(current_state, N_i_after_selection):
    N_i_combined = {}

    for genotype_str in ALL_GENOTYPES:
        N_combined = current_state.N_i[genotype_str] + N_i_after_selection[genotype_str]
        N_i_combined[genotype_str] = N_combined

    N_i_next = {}
    N_Total_next = Decimal('0.0')

    for genotype_str, count_combined in N_i_combined.items():
        N_next = LAMBDA * count_combined
        N_i_next[genotype_str] = N_next
        N_Total_next += N_next

    return N_i_next, N_Total_next


# ------------------------------------------------------------
# 4. اجرای شبیه‌سازی
# ------------------------------------------------------------

def run_simulation():
    N_i_initial = get_initial_population()

    N_Total_current = sum(N_i_initial.values())

    mu_bar_prev = INITIAL_MU0.copy()
    N_i_current = N_i_initial
    alpha_current = INITIAL_ALPHA.copy()

    # ایجاد شیء حالت برای تعیین ستون‌های CSV
    initial_state_for_header = SimulationState(0, N_Total_current)
    initial_state_for_header.mu0 = INITIAL_MU0
    initial_state_for_header.alpha = INITIAL_ALPHA

    # فراوانی‌ها را محاسبه کن تا کلیدها در to_dict تولید شوند.
    initial_state_for_header.calculate_frequencies()
    fieldnames = list(initial_state_for_header.to_dict().keys())

    with open(OUTPUT_FILENAME, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for t in tqdm(range(NUM_GENERATIONS + 1), desc="Running Simulation"):

            if N_Total_current == Decimal('0'):
                print("جمعیت به صفر رسید. شبیه‌سازی متوقف شد.")
                break

            current_state = SimulationState(t, N_Total_current)
            current_state.N_i = N_i_current
            current_state.alpha = alpha_current

            alpha_next = phase_1_analysis(current_state, mu_bar_prev)
            N_i_offspring = phase_2_reproduction(current_state)
            N_i_after_selection = phase_3_selection(current_state, N_i_offspring)
            N_i_next_float, N_Total_next = phase_4_dynamics(current_state, N_i_after_selection)

            if t % SAVE_INTERVAL == 0:
                writer.writerow(current_state.to_dict())

            mu_bar_prev = current_state.mu_bar.copy()
            alpha_current = alpha_next

            N_Total_current = N_Total_next
            N_i_current = N_i_next_float

    print(f"شبیه‌سازی با موفقیت پایان یافت. نتایج در فایل '{OUTPUT_FILENAME}' ذخیره شد.")


run_simulation()
