import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as sps
import math

def read_from_file(filename):

    with open(filename) as f:
        lines = f.readlines()

    input = "".join(lines)
    a = input.split(",")
    sequence = [int(elem) for elem in a if len(elem) != 0]

    return sequence

def plot_distribution(nums):

    size = []
    mean_array = []
    std_array = []

    for i in range(len(nums)):
        size.append(i+1)
        mean_array.append(np.mean(nums[:i+1]))
        std_array.append(sps.tstd(nums[:i+1]))

    plt.plot(size, mean_array, label="Мат. ожидание")
    plt.plot(size, std_array, label="Среднеквадратичное отклонение")
    plt.xlabel("Размер выборки")
    plt.legend()
    plt.show()

def pogr(seq):

    GOAL_MEAN = 0.5
    GOAL_STD = 0.2887

    cur_value_mean = np.mean(seq)
    cur_value_std = sps.tstd(seq)

    print("Относительная погрешность мат. ожидания: ", abs(GOAL_MEAN - cur_value_mean))
    print("Относительная погрешность среднекв. отклонения: ", abs(GOAL_STD - cur_value_std))

def transform_lst_nums(num_lst):

    max_elem = max(num_lst) + 1
    return [num / max_elem for num in num_lst]

def chi_2(seq, alpha=0.05, lst_=None, exp=None, param=None):

    if param is None: 
        param = len(np.unique(seq))
    if lst_ is None: 
        _, lst_ = np.unique(seq, return_counts=True)
    if exp is None: 
        exp = np.array([len(seq) / param] * param)

    chi, stat = np.sum((lst_ - exp) ** 2 / exp), sps.chi2.ppf(1 - alpha, param - 1)

    if chi > stat: 
        return "-"
    else: 
        return "+"

def series(seq):
    d = 16
    alpha = 0.05
    param = d ** 2
    res = np.zeros(param, dtype=int)

    for j in range(len(seq) // 2):
        res[int(seq[2 * j] * d) * d + int(seq[2 * j + 1] * d)] += 1

    return chi_2(seq, alpha, res, np.full(param, len(seq) / (2 * param)), param)

def intervals(seq):

    d = 16
    j, s, emp = -1, 0, 8 * [0]
    t = 7
    n = len(seq)
    interval_amount = n / 10
    half = 0.5
    theor = [interval_amount * half * (1.0 - half) ** r for r in range(t)] + [interval_amount * (1.0 - half) ** t]

    while s != interval_amount and j != n:
        j += 1
        r = 0
        while j != n and seq[j] < d / 2:
            j += 1
            r += 1
        emp[min(r, t)] += 1
        s += 1

    if j == n:
        return "-"
    
    return chi_2(seq, 0.05 ,theor, emp, t + 1)

def partitions(seq):
    alpha = 0.05
    n = 100
    param = int(10000 / n)
    r = np.array([0] * (param + 1))

    for i in range(n):
        r[len(np.unique(seq[param * i : param * (i + 1)]))] += 1

    p = []
    s = 1

    for i in range(param + 1):
        d = 100
        p_i = d
        for j in range(1, i):
            p_i *= d - j
        p.append(p_i / pow(d, param) * s)

    dk_lst = np.array([math.comb(param + i - 1, i) / pow(d, param) for i in range(param + 1)])
    return chi_2(seq, alpha, dk_lst[1:], p[1:], param)

def permutations(seq):

    alpha = 0.05
    t = 10
    n = len(seq)
    dict = {}
    param = math.factorial(t)

    for i in range(0, n, t):
        group = tuple(sorted(seq[i:i + t]))
        dict[group] = dict.get(group, 0) + 1

    lst_obs = sorted(list(dict.values()), reverse=True)

    exp = np.array([n / param] * len(lst_obs))

    return chi_2(seq, alpha, lst_obs, exp, param)

def monotony(seq):

    alpha = 0.05
    A = [
        [4529.4, 9044.9, 13568, 22615,  22615,  27892 ],
        [9044.9, 18097,  27139, 36187,  452344, 55789 ],
        [13568,  27139,  40721, 54281,  67582,  83685 ],
        [18091,  36187,  54281, 72414,  90470,  111580],
        [22615,  45234,  67852, 90470,  113262, 139476],
        [27892,  55789,  83685, 111580, 139476, 172860]
    ]
    b = [1 / 6, 5 / 24, 11 / 120, 19 / 720, 29 / 5040, 1 / 840]
    n = len(seq)
    lst = []

    i = 0
    while i < n:
        s = 1
        while i + s < n and seq[i + s - 1] <= seq[i + s]:
            s += 1
        lst.append(s)
        i += s

    counts = {}
    for l in lst:
        counts[l] = counts.get(l, 0) + 1

    res = []
    temp = 0
    for c in lst:
        m = 1 / 6
        min_val = min(c, 6)
        for i in range(min_val):
            for j in range(min_val):
                m += (seq[i + temp] - n * b[i]) * (seq[j + temp] - n * b[j]) * A[i][j]
        temp += c
        res.append(m)

    return chi_2(res, alpha)

def conflicts(srq):
    m = 1024
    l = len(srq)
    sr_ = l / m
    p0 = 1 - l / m + math.factorial(l) / (2 * math.factorial(l - 2) * m*2)

    conf = l / m - 1 + p0
    return "-" if abs(conf - sr_) > 10 else "+"

if __name__ == "__main__":

    path = input()
    p = read_from_file(path)
    trans_p = transform_lst_nums(p)

    mean = np.mean(trans_p)
    print(f"Мат. ожидание последовательности: {mean}")

    std = sps.tstd(trans_p)
    print(f"Среднеквадратичное отклонение последовательности: {std}")

    pogr(trans_p)

    print("Критерий хи-квадрат:")
    print(chi_2(trans_p))

    print("Критерий серий:")
    print(series(trans_p))

    print("Критерий интервалов:")
    print(intervals(trans_p))

    print("Критерий разбиений:")
    print(partitions(trans_p))

    print("Критерий перестановок:")
    print(permutations(trans_p))

    print("Критерий монотонности:")
    print(monotony(trans_p))

    print("Критерий конфликтов:")
    print(conflicts(trans_p))

    plot_distribution(trans_p)