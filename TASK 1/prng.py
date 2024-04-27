import random
import sys
import math
import argparse

def lc(m, a, c, x0, n):

    if m <= 0 or a > m or a < 0 or c > m or c < 0 or x0 > m or x0 < 0:
        print("Ошибка!")
        return
    
    print('Генерация выполнена на 0%')

    x = [0] * n
    x[0] = x0
    for i in range(1, n):
            x[i] = (a * x[i - 1] + c) % m
            if (i / n * 100 == 25):
                print('Генерация выполнена на 25%')
            if (i / n * 100 == 50):
                print('Генерация выполнена на 50%')
            if (i / n * 100 == 75):
                print('Генерация выполнена на 75%')
    
    print('Генерация выполнена на 100%')

    return x

def add(m, k, j, starts, n):

    if m <= 0 or k >= j or k < 1 or j + 3 > len(starts):
        print("Ошибка!")
        return
    
    print('Генерация выполнена на 0%')

    x = starts
    for i in range(n):
        x.append((x[i - k] + x[i - j]) % m)
        if (i / n * 100 == 25):
            print('Генерация выполнена на 25%')
        if (i / n * 100 == 50):
            print('Генерация выполнена на 50%')
        if (i / n * 100 == 75):
            print('Генерация выполнена на 75%')
    
    print('Генерация выполнена на 100%')

    return x

def get_bit(num, num_bit):
    return (num & ( 1 << num_bit )) >> num_bit

def set_bit(num, num_bit, bit):
    mask = 1 << num_bit
    num &= ~mask
    if bit:
        return num | mask
    else:
        return num
    
def shift(num, s):
    new_num = 0
    bit = 0
    for i in range(s):
        new_num = set_bit(new_num, i, bit)
        bit = get_bit(num, i)
    new_num = set_bit(new_num, 0, bit)
    return new_num

def lfsr(x0, reg, n):

    x = []
    print('Генерация выполнена на 0%')

    lenreg = len(reg)
    reg = int(reg, 2)
    x0 = int(x0, 2)

    for i in range(n):
        cur_bit = 0

        for j in range(lenreg):
            cur_bit ^= get_bit(x0, j) * get_bit(reg, j)
        
        reg = shift(reg, lenreg)
        reg = set_bit(reg, 0, cur_bit)
        x.append(reg)

        if (i / n * 100 == 25):
            print('Генерация выполнена на 25%')
        if (i / n * 100 == 50):
            print('Генерация выполнена на 50%')
        if (i / n * 100 == 75):
            print('Генерация выполнена на 75%')
    
    print('Генерация выполнена на 100%')

    return(x)

def p5(p,q1,q2,q3,w,x0,n):
    
    if q1 >= p or q2 >= p or q3 >= p:
        print("Ошибка!")
        return
    
    x = []
    x.append(x0)

    mask_w = 0

    for i in range(w):
        mask_w = set_bit(mask_w, i, 1)

    mask_p = 0

    for i in range(p):
        mask_p = set_bit(mask_p, i, 1)

    print('Генерация выполнена на 0%')
    for i in range(n):
        cur_bit = 0

        cur_bit ^= get_bit(x0, q1)
        cur_bit ^= get_bit(x0, q2)
        cur_bit ^= get_bit(x0, q3)
        cur_bit ^= get_bit(x0, 0)

        x0 = shift(x0, p)
        x0 = set_bit(x0, 0, cur_bit)
        x0 = x0 & mask_p
        x.append(x0 & mask_w)

        if (i / n * 100 == 25):
            print('Генерация выполнена на 25%')
        if (i / n * 100 == 50):
            print('Генерация выполнена на 50%')
        if (i / n * 100 == 75):
            print('Генерация выполнена на 75%')
    
    print('Генерация выполнена на 100%')

    return x

def lfsr_help(x0, reg, n):

    x = []

    lenreg = len(reg)
    reg = int(reg, 2)

    for i in range(n):
        cur_bit = 0

        for j in range(lenreg):
            cur_bit ^= get_bit(x0, j) * get_bit(reg, j)
        
        reg = shift(reg, lenreg)
        reg = set_bit(reg, 0, cur_bit)
        x.append(reg)

    return(x)

# print(p5(87, 20, 40, 69, 9, 712, 100))

def nfsr(r1, r2, r3, w, x1, x2, x3, n):

    len_R1 = len(r1)
    len_R2 = len(r2)
    len_R3 = len(r3)

    R1 = lfsr_help(x1, r1, n)
    R2 = lfsr_help(x2, r2, n)
    R3 = lfsr_help(x3, r3, n)

    w1 = 0
    x = []
    print('Генерация выполнена на 0%')

    for i in range(int(w)):
        w1 = set_bit(w1, i, 1)

    for i in range(n):
        x.append(((R1[i] ^ R2[i]) + (R2[i] ^ R3[i]) + R3[i]) & w1)
        if (i / n * 100 == 25):
            print('Генерация выполнена на 25%')
        if (i / n * 100 == 50):
            print('Генерация выполнена на 50%')
        if (i / n * 100 == 75):
            print('Генерация выполнена на 75%')

    print('Генерация выполнена на 100%')
    return x

#print(nfsr("100000001001001", "0011000000", "101011001001001", 9, 25, 60, 45, 100))

def mt(mod, x0, n):

    p, w, r, q, a, u, s, t, l, b, c = 624, 32, 31, 397, 2567483615, 11, 7, 15, 18, 2636928640, 4022730752
    lower_mask = (1 << r) - 1

    w1 = 0
    for i in range(w):
        w1 = set_bit(w1, i, 1)
    upper_mask = (~lower_mask * -1) & w1

    res = []
    print('Генерация выполнена на 0%')
  
    MT = []
    MT.append(x0)

    for i in range(1, p):
        MT.append((MT[i - 1] ^ (MT[i - 1] >> 30)) + i)
    
    ind = p
    for j in range(n):

        if (ind >= p):
            for i in range(p):
                x = (MT[i] & upper_mask) + (MT[(i + 1) % p] & lower_mask)
                xA = x >> 1
                if (x & 1):
                    xA ^= a
                MT[i] = MT[(i + q) % p] ^ xA
            ind = 0

        y = MT[ind]
        ind += 1
        y ^= (y >> u)
        y ^= (y << s) & b
        y ^= (y << t) & c
        y ^= (y >> l)

        res.append(y % mod)

        if (j / n * 100 == 25):
            print('Генерация выполнена на 25%')
        if (j / n * 100 == 50):
            print('Генерация выполнена на 50%')
        if (j / n * 100 == 75):
            print('Генерация выполнена на 75%')

    print('Генерация выполнена на 100%')
    return res

#print(mt(1000, 1234, 100))

def rc4(k, n):
    s = [i for i in range(256)]

    j = 0
    for i in range(256):
        j = (j + s[i] + k[i]) % 256
        s[i], s[j] = s[j], s[i]
    
    i, j = 0, 0

    x = []
    print('Генерация выполнена на 0%')
    for k in range(n):
        num = 0
        i = (i + 1) % 256
        j = (j + s[i]) % 256
        s[i], s[j] = s[j], s[i]
        num = s[(s[i] + s[j]) % 256]

        x.append(num)
        if (k / n * 100 == 25):
            print('Генерация выполнена на 25%')
        if (k / n * 100 == 50):
            print('Генерация выполнена на 50%')
        if (k / n * 100 == 75):
            print('Генерация выполнена на 75%')

    print('Генерация выполнена на 100%')
    return x


#print(rc4([213,968,838,64,355,214,212,36,695,139,897,518,656,956,810,510,985,105,670,8,907,951,685,989,222,931,169,286,289,556,731,902,688,701,771,533,990,630,708,884,255,683,25,214,792,348,34,758,9,781,946,580,615,955,585,5,886,563,81,38,809,444,619,222,544,53,635,621,630,251,497,257,2,467,897,790,728,676,722,838,465,781,10,828,903,235,857,841,146,719,681,678,961,652,491,38,256,909,251,21,110,811,273,25,642,286,489,478,184,812,770,846,241,141,266,500,375,827,633,761,154,663,461,206,529,212,667,342,360,165,523,749,582,803,553,345,786,990,361,702,256,380,234,238,73,965,266,300,847,755,969,681,146,843,125,306,845,752,879,458,788,833,727,817,122,239,765,877,827,327,733,658,644,880,150,474,493,689,670,368,611,263,113,417,834,103,725,754,117,824,623,338,540,337,879,521,183,370,808,120,571,871,301,210,796,744,398,106,845,745,842,876,399,27,105,601,802,831,53,266,157,352,175,303,505,484,994,425,292,729,654,584,860,420,412,49,281,417,703,400,48,404,772,389,733,152,271,585,404,333,381,696,928,609,659,180,9], 100))

def rsa(N, e, w, x0, n):
    x = []
    x.append(x0)
    print('Генерация выполнена на 0%')

    for i in  range(n):
        z = 0
        for j in range(w):
            x0 = pow(x0, e, N) 
            z = set_bit(z, w - j - 1, x0 & 1)
        x.append(z)
        if (i / n * 100 == 25):
            print('Генерация выполнена на 25%')
        if (i / n * 100 == 50):
            print('Генерация выполнена на 50%')
        if (i / n * 100 == 75):
            print('Генерация выполнена на 75%')

    print('Генерация выполнена на 100%')
    return x

#print(rsa(12709189, 53, 10, 245, 100))

def bbs(x0, w, N):
    p = 127
    q = 131
    n = p * q

    x = []

    print('Генерация выполнена на 0%')
    for i in range(N):
        z = 0
        for j in range(w):
            x0 = x0 * x0 % n
            z = set_bit(z, w - j - 1, x0 & 1)
        x.append(z)
        if (i / n * 100 == 25):
            print('Генерация выполнена на 25%')
        if (i / n * 100 == 50):
            print('Генерация выполнена на 50%')
        if (i / n * 100 == 75):
            print('Генерация выполнена на 75%')
    
    print('Генерация выполнена на 100%')    
    return x


#print(bbs(15621, 10, 100))

def write_to_file(data, filepath="rnd.dat"):
    with open(filepath, "w", encoding="UTF-8") as f:
        f.write(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="prng.py")

    helpg = """
            -g <код_метода> -- параметр указывает на метод генерации ПСЧ, при этом код_метода может быть
            одним из следующих:\n                                 
            1) lc – линейный конгруэнтный метод;\n                
            2) add – аддитивный метод;\n 
            3) 5p – пятипараметрический метод;\n 
            4) lfsr – регистр сдвига с обратной связью (РСЛОС);\n 
            5) nfsr – нелинейная комбинация РСЛОС;\n 
            6) mt – вихрь Мерсенна;\n 
            7) rc4 – RC4;\n 
            8) rsa – ГПСЧ на основе RSA;\n 
            9) bbs – алгоритм Блюма-Блюма-Шуба.\n 
"""
    helpi = """
            -i <парметры> -- параметр указывает на парметры генерации ПСЧ, при этом для -g могут быть
            одними из следующих:\n                                 
            1) lc /i: модуль, множитель, приращение, начальное значение;\n                
            2) add /i: модуль, младший индекс, старший индекс, последовательность начальных значений;\n 
            3) 5p /i: p, q_1, q_2, q_3, w, x0;\n 
            4) lfsr /i: двоичное представление вектора коэффициентов, начальное значение регистра;\n 
            5) nfsr /i: двоичные представления векторов коэффициентов для R1, R2, R3, скомбинированных функцией R1^R2 + R2^R3 + R3;\n 
            6) mt /i: модуль, начальное значение x;\n 
            7) rc4 /i: 256 начальных значений;\n 
            8) rsa /i: модуль n, число e, начальное значение x. e удовлетворяет: 1 < e < (p - 1) * (q - 1), НОД(e, (p - 1) * (q - 1)) = 1, p * q = n, x из [1, n];\n 
            9) bbs /i: начальное значение x (взаимно простое с n). При генерации используются параметры: p=127, q=131, n=p*q=16637.\n 
"""
    helpn = '-n <длина> -- количество генерируемых чисел. Если параметр не указан, -- генерируется 10000 чисел.\n'

    helpf = '-f <полное_имя_файла> -- полное имя файла, в который будут выводиться данные. Если параметр не указан, данные будут записаны в файл с именем rnd.dat.\n'
    
    helph = '-h информация о допустимых параметрах командной строки программы.'

    parser.add_argument("-g", help=helpg, required=True, choices=["lc", "add", "5p", "lfsr", "nfsr", "mt", "rc4", "rsa", "bbs"], nargs=1)
    parser.add_argument("-i", type=str, help=helpi)
    parser.add_argument("-n", nargs=1, type=int, default=[10000], help=helpn)
    parser.add_argument("-f", nargs=1, default=["rnd.dat"], help=helpf)

    args = parser.parse_args()

    def check(n, args):
        if len(args) != n:
                raise Exception("Передано неверное количество аргументов")
        for arg in args:
            if not arg.isdigit():
                raise Exception("Переданы нечисловые параметры")
        return True

    g_name = args.g[0]
    i_args = args.i.split(sep=",")
    f_path = args.f[0]
    n_ = args.n[0]

    try:
        match g_name:

            case 'lc':

                if check(4, i_args):
                    m, a, c = int(i_args[0]), int(i_args[1]), int(i_args[2])
                    x0 = int(i_args[3])
                    x = lc(m, a, c, x0, n_)
                res_str = ",".join(map(str, x))
                write_to_file(res_str, filepath=f_path)

            case 'add':

                if check(4, i_args):
                    m = int(i_args[0])
                    k, j = int(i_args[1]), int(i_args[2])
                    starts = list(map(int, i_args[3:]))
                    x = add(m, k, j, starts, n_)
                res_str = ",".join(map(str, x))
                write_to_file(res_str, filepath=f_path)

            case '5p':

                if check(6, i_args):
                    p, q1, q2, q3, w = int(i_args[0]), int(i_args[1]), int(i_args[2]), int(i_args[3]), int(i_args[4])
                    x0 = int(i_args[5])
                    x = p5(p, q1, q2, q3, w, x0, n_)
                    res_str = ",".join(map(str, x))
                    write_to_file(res_str, filepath=f_path)

            case 'lfsr':

                if check(2, i_args):
                    x0, reg = i_args[0], i_args[1]
                    x = lfsr(x0, reg, n_)
                    res_str = ",".join(map(str, x))
                    write_to_file(res_str, filepath=f_path)

            case 'nfsr':

                if check(7, i_args):
                    r1, r2, r3 = i_args[0], i_args[1], i_args[2]
                    w, x1, x2, x3 = int(i_args[3]), int(i_args[4]), int(i_args[5]), int(i_args[6])
                    x = nfsr(r1, r2, r3, w, x1, x2, x3, n_)
                    res_str = ",".join(map(str, x))
                    write_to_file(res_str, filepath=f_path)

            case 'mt':

                if check(2, i_args):
                    mod = int(i_args[1])
                    x0 = int(i_args[0])
                    x = mt(mod, x0, n_)
                    res_str = ",".join(map(str, x))
                    write_to_file(res_str, filepath=f_path)

            case 'rc4':

                if check(256, i_args):
                    k = list(map(int, i_args))
                    x = rc4(k, n_)
                    res_str = ",".join(map(str, x))
                    write_to_file(res_str, filepath=f_path)

            case 'rsa':

                if check(4, i_args):
                    N, e, w, x0 = int(i_args[0]), int(i_args[1]), int(i_args[2]), int(float(i_args[3]))
                    x = rsa(N, e, w, x0, n_)
                    res_str = ",".join(map(str, x))
                    write_to_file(res_str, filepath=f_path)

            case 'bbs':

                if check(2, i_args):
                    x0, w = int(i_args[0]), int(i_args[1])
                    x = bbs(x0, w, n_)
                    res_str = ",".join(map(str, x))
                    write_to_file(res_str, filepath=f_path)
                    
    except Exception as err:
        print("В процессе генерации произошла ошибка! ", str(err))