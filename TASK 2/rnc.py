import argparse
import numpy as np

def U(x, m):
    return x / m

def st(a, b, l):
    y = []
    m = max(l) + 1
    for x in l:
        y.append(a + U(x ,m) * b)
    return y

def tr(a, b, l):
    y = []
    m = max(l) + 1
    for i in range(0, len(l)-1, 2):
        y.append(a + b * (U(l[i], m) + U(l[i + 1], m) - 1))
    return y

def ex(a, b, l):
    y = []
    m = max(l) + 1
    for x in l:
        y.append(a - b * np.log(U(x, m)))
    return y

def nr(a, b, l):
    y = []
    m = max(l) + 1
    for i in range(0, len(l)-1, 2):
        y.append(a + b * np.sqrt(-2 * np.log(1 - U(l[i], m))) * np.cos(2 * np.pi * U(l[i+1], m)))
        y.append(a + b * np.sqrt(-2 * np.log(1 - U(l[i], m))) * np.sin(2 * np.pi * U(l[i+1], m)))
    return y

def gm(a, b, c, l):
    y = []
    m = max(l) + 1
    u = []
    uk = []
    for x in l:
        u.append(U(x, m))
    for i in range(0, len(u), c):
        uk.append((u[i : i + c]))

    if len(uk[-1]) != c:
        uk.pop()

    for mass in uk:
        x = 1
        for el in mass:
            x *= 1 - el
        y.append(a - b * np.log(x))
    return y

def ln(a, b, l):
    y = []
    m = max(l) + 1
    l = nr(0, 1, l)
    for x in l:
        y.append(a + np.exp(b - x))
    return y

def ls(a, b, l):
    y = []
    m = max(l) + 1
    u = []
    for x in l:
        u.append(U(x, m)) 
    for x in u:
        y.append(a + b * np.log(x / (1 - x)))
    return y

def factor(x):
    y = 1
    for i in range(x):
        y *= (i + 1)
    return y

def bi(a, b, l):
    y = []
    m = max(l) + 1
    u = []
    for x in l:
        u.append(U(x, m))
    for i in u:
        s = 0
        k = 0
        while(True):
            s += (factor(b) / (factor(k) * factor(b - k)) * (a ** k) * ((1 - a) ** (b - k)))
            if s > i:
                y.append(k)
                break
            if k < b - 1:
                k += 1
                continue
            y.append(b)
    return y

def write_to_file(data, f):
    with open(f, "w", encoding="UTF-8") as f:
        f.write(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="rnc.py")

    helpd = """Код распределения для преобразования последовательности:\n 
                st – стандартное равномерное с заданным интервалом,\n
                tr – треугольное распределение,\n
                ex – общее экспоненциальное распределение\n
                nr – нормальное распределение,\n
	            gm – гамма распределение,\n
	            ln – логнормальное распределение,\n 
	            ls – логистическое распределение,\n
	            bi – биномиальное распределение"""
    
    helpf = "Имя файла с входной последовательностью."
    helpp1 = "1-й параметр, необходимый, для генерации ПСЧ заданного распределения"
    helpp2 = "2-й параметр, необходимый, для генерации ПСЧ заданного распределения"
    helpp3 = "3-й параметр, необходимый, для генерации ПСЧ гамма-распределением."

    parser.add_argument("-f", nargs=1, default=["rnd.dat"], help=helpf)
    parser.add_argument("-d", help=helpd, required=True, choices=["st", "tr", "ex", "nr", "gm", "ln", "ls", "bi"], nargs=1)
    parser.add_argument("-p1", nargs=1, type=float, required=True, help=helpp1)
    parser.add_argument("-p2", nargs=1, type=int, required=True, help=helpp2)
    parser.add_argument("-p3", nargs=1, type=int, help=helpp3, default=[None])

    args = parser.parse_args()

    def check(n, args):
        if len(args) != n:
                raise Exception("Передано неверное количество аргументов")
        for arg in args:
            if not arg.isdigit():
                raise Exception("Переданы нечисловые параметры")
        return True

    d_name = args.d[0]
    f_name = args.f[0]
    p1 = args.p1[0]
    p2 = args.p2[0]
    p3 = args.p3[0]

    try:

        with open(f_name, "r") as f:
            line = f.readline()
            l  = list(map(int, line.split(",")))

        match d_name:
            
            case 'st':
                    y = st(p1, p2, l)
                    res_str = ",".join(map(str, y))
                    write_to_file(res_str, "distr-st.dat")

            case 'tr': 
                    y = tr(p1, p2, l)
                    res_str = ",".join(map(str, y))
                    write_to_file(res_str, "distr-tr.dat")

            case 'ex':
                    y = ex(p1, p2, l)
                    res_str = ",".join(map(str, y))
                    write_to_file(res_str, "distr-ex.dat")

            case 'nr':
                    y = nr(p1, p2, l)
                    res_str = ",".join(map(str, y))
                    write_to_file(res_str, "distr-nr.dat")

            case 'gm':
                    y = gm(p1, p2, p3, l)
                    res_str = ",".join(map(str, y))
                    write_to_file(res_str, "distr-gm.dat")

            case 'ln':
                    y = ln(p1, p2, l)
                    res_str = ",".join(map(str, y))
                    write_to_file(res_str, "distr-ln.dat")

            case 'ls':
                    y = ls(p1, p2, l)
                    res_str = ",".join(map(str, y))
                    write_to_file(res_str, "distr-ls.dat")

            case 'bi':
                    y = bi(p1, p2, l)
                    res_str = ",".join(map(str, y))
                    write_to_file(res_str, "distr-bi.dat")
           
    except Exception as err:
        print("В процессе генерации произошла ошибка! " + str(err))