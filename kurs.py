import numpy as np
import matplotlib.pyplot as plt
import math


x_s = np.arange(0, 281, 2, dtype=np.float64)
y_s = np.array([172, 170, 169, 166, 163, 160, 156, 152, 147, 143, 140, 136, 131, 127, 122, 117, 114, 108, 103, 98, 93, 86, 80, 74, 69, 64, 59, 54, 49, 42, 37, 31, 27, 25, 23, 22, 21, 20, 19.98, 19.8, 19.97, 20, 20, 20.05, 20, 19.98, 19.5, 19, 18, 17, 15, 13, 10, 8, 7, 6, 5.7, 5.5, 5.47, 5.5, 5.7, 6, 7, 8, 9, 10, 11, 12.5, 14, 15, 15.5, 15.3, 14.8, 13, 12, 10, 9, 8, 7, 6.5, 6.1, 6, 5.9, 6, 6.1, 6.7, 7, 8.5, 10, 12, 14, 17, 20, 23, 26, 32, 37, 45, 49, 51, 54, 56, 57, 58, 59, 60, 60.5, 61, 62, 62.5, 63, 64, 65, 67, 69, 72, 76, 82, 100, 111, 115, 118, 120, 121, 122, 123.2, 125, 125.5, 127, 128, 129, 131, 134, 139, 146, 155, 162, 165, 168, 170, 172], dtype=np.float64)

x_cubic = np.array([0, 22, 32, 44, 56, 66, 72, 84, 100, 110, 132, 138, 154, 172, 178, 184, 198, 210, 226, 242, 264, 280])
y_cubic = np.array([172, 136, 114, 80, 49, 25, 21, 20, 15, 6, 11, 15, 8, 7, 12, 20, 51, 60, 72, 118, 134, 172])

def digitization(x, y):
    plt.title("Оцифровка ямы")
    plt.xlabel("x[mm]")
    plt.ylabel("y[mm]")
    plt.plot(x, y)
    plt.grid()
    plt.show()

def newton(x_values, y_values):
    x_1 = np.array([0, 22, 44, 66, 110, 132, 154, 198, 242, 264, 280], dtype=np.float64)
    y_1 = np.array([172, 136, 80, 25, 6, 11, 8, 51, 118, 134, 172], dtype=np.float64)

    def divided_differences(x_values, y_values, k):
        result = 0
        for j in range(k + 1):
            mul = 1
            for i in range(k + 1):
                if i != j:
                    mul *= (x_values[j] - x_values[i])
            result += y_values[j]/mul
        return result


    def create_Newton_polynomial(x_values, y_values):
        div_diff = []
        for i in range(1, len(x_values)):
            div_diff.append(divided_differences(x_values, y_values, i))
        def newton_polynomial(x):
            result = y_values[0]
            for k in range(1, len(y_values)):
                mul = 1
                for j in range(k):
                    mul *= (x-x_values[j])
                result += div_diff[k-1]*mul
            return result
        return newton_polynomial

    plt.title("Полином Ньютона")
    plt.xlabel("x[mm]")
    plt.ylabel("y[mm]")
    new_pol = create_Newton_polynomial(x_1, y_1)
    plt.plot(x_s, y_s, x_s, new_pol(x_s))
    plt.grid()
    plt.show()

def lagrange(x_values, y_values):
    x_1 = np.array([0, 22, 44, 66, 110, 132, 154, 198, 242, 264, 280], dtype=np.float64)
    y_1 = np.array([172, 136, 80, 25, 6, 11, 8, 51, 118, 134, 172], dtype=np.float64)

    def create_basic_pol(x_values, i):
        def basic_pol(x):
            divider = 1
            res = 1
            for j in range(len(x_values)):
                if j != i:
                    res *= (x - x_values[j])
                    divider *= (x_values[i] - x_values[j])
            return res/divider
        return basic_pol


    def create_Lagr_pol(x_values, y_values):
        basic_pol = []
        for i in range(len(x_values)):
            basic_pol.append(create_basic_pol(x_values, i))
        def lagr_pol(x):
            res = 0
            for i in range(len(y_values)):
                res += y_values[i] * basic_pol[i](x)
            return res
        return lagr_pol

    plt.title("Полином Лагранжа")
    plt.xlabel("x[mm]")
    plt.ylabel("y[mm]")
    new_pol = create_Lagr_pol(x_1, y_1)
    plt.plot(x_s, y_s, x_s, new_pol(x_s))
    plt.grid()
    plt.show()
    return new_pol

def cubic_interp1d(x_values, y_values):
    x_cubic = np.array([0, 22, 32, 44, 56, 66, 72, 84, 100, 110, 132, 138, 154, 172, 178, 184, 198, 210, 226, 242, 264, 280])
    y_cubic = np.array([172, 136, 114, 80, 49, 25, 21, 20, 15, 6, 11, 15, 8, 7, 12, 20, 51, 60, 72, 118, 134, 172])

    def spline_create(x0, x, y):
        x = np.asfarray(x)
        y = np.asfarray(y)

        if np.any(np.diff(x) < 0):
            indexes = np.argsort(x)
            x = x[indexes]
            y = y[indexes]

        size = len(x)

        xdiff = np.diff(x)
        ydiff = np.diff(y)

        Li = np.empty(size)
        Li_1 = np.empty(size - 1)
        z = np.empty(size)

        Li[0] = math.sqrt(2 * xdiff[0])
        Li_1[0] = 0.0
        B0 = 0.0
        z[0] = B0 / Li[0]

        for i in range(1, size - 1, 1):
            Li_1[i] = xdiff[i - 1] / Li[i - 1]
            Li[i] = math.sqrt(2 * (xdiff[i - 1] + xdiff[i]) - Li_1[i - 1] * Li_1[i - 1])
            Bi = 6 * (ydiff[i] / xdiff[i] - ydiff[i - 1] / xdiff[i - 1])
            z[i] = (Bi - Li_1[i - 1] * z[i - 1]) / Li[i]

        i = size - 1
        Li_1[i - 1] = xdiff[-1] / Li[i - 1]
        Li[i] = math.sqrt(2 * xdiff[-1] - Li_1[i - 1] * Li_1[i - 1])
        Bi = 0.0
        z[i] = (Bi - Li_1[i - 1] * z[i - 1]) / Li[i]

        i = size - 1
        z[i] = z[i] / Li[i]
        for i in range(size - 2, -1, -1):
            z[i] = (z[i] - Li_1[i - 1] * z[i + 1]) / Li[i]

        index = x.searchsorted(x0)
        np.clip(index, 1, size - 1, index)

        xi1, xi0 = x[index], x[index - 1]
        yi1, yi0 = y[index], y[index - 1]
        zi1, zi0 = z[index], z[index - 1]
        hi1 = xi1 - xi0

        f0 = zi0 / (6 * hi1) * (xi1 - x0) ** 3 + zi1 / (6 * hi1) * (x0 - xi0) ** 3 + (yi1 / hi1 - zi1 * hi1 / 6) * (
                    x0 - xi0) + (yi0 / hi1 - zi0 * hi1 / 6) * (xi1 - x0)
        return f0

    plt.scatter(x_cubic, y_cubic)
    plt.plot(x_s, y_s, x_s, spline_create(x_s, x_cubic, y_cubic))
    plt.title("Сплайн интерполяция")
    plt.xlabel("x[mm]")
    plt.ylabel("y[mm]")
    plt.grid()
    plt.show()


def dif(x_s, y_s):
    x_sdif = np.arange(0, 280, 2)
    dif_1 = []
    dif_2 = []
    dif_3 = []
    dif_3_2 = []
    for i in range(len(x_s) - 1):
        dif_1.append(y_s[i] - y_s[i + 1])

    for i in range(1, len(x_s)):
        dif_2.append(y_s[i - 1] - y_s[i])

    for i in range(len(x_s) - 1):
        dif_3.append((y_s[i - 1] - y_s[i + 1]) / 2)

    for i in range(len(x_s) - 1):
        dif_3_2.append((y_s[i - 1] - 2 * y_s[i] + y_s[i + 1]) / 2)

    plt.title("Численное дифференцирование")
    plt.xlabel("x[mm]")
    plt.ylabel("y[mm]")
    plt.plot(x_sdif, dif_1, 'k')
    plt.plot(x_sdif, dif_2, 'y')
    plt.plot(x_sdif, dif_3, 'b')
    plt.plot(x_sdif, dif_3_2, 'c')
    plt.grid()
    plt.show()

def integr():
    x_s_1 = np.arange(0, 278, 2, dtype=np.float64)

    def f(x):
        y_s = [172, 170, 169, 166, 163, 160, 156, 152, 147, 143, 140, 136, 131, 127, 122, 117, 114, 108, 103, 98, 93,
               86, 80,
               74, 69, 64, 59, 54, 49, 42, 37, 31, 27, 25, 23, 22, 21, 20, 19.98, 19.8, 19.97, 20, 20, 20.05, 20, 19.98,
               19.5,
               19, 18, 17, 15, 13, 10, 8, 7, 6, 5.7, 5.5, 5.47, 5.5, 5.7, 6, 7, 8, 9, 10, 11, 12.5, 14, 15, 15.5, 15.3,
               14.8,
               13, 12, 10, 9, 8, 7, 6.5, 6.1, 6, 5.9, 6, 6.1, 6.7, 7, 8.5, 10, 12, 14, 17, 20, 23, 26, 32, 37, 45, 49,
               51, 54,
               56, 57, 58, 59, 60, 60.5, 61, 62, 62.5, 63, 64, 65, 67, 69, 72, 76, 82, 100, 111, 115, 118, 120, 121,
               122,
               123.2, 125, 125.5, 127, 128, 129, 131, 134, 139, 146, 155, 162, 165, 168, 170, 172]
        y = y_s[x]
        return y

    def left_rect():
        rsl = []
        n = 140
        a = 0
        b = 280
        h = (b - a) // n
        sum = 0
        for i in range(n - 1):
            sum += h * f(i)
            rsl.append(sum)
        return rsl

    def right_rect():
        rsl = []
        n = 140
        a = 0
        b = 280
        h = (b - a) // n
        sum = 0
        for i in range(1, n):
            sum += h * f(i)
            rsl.append(sum)
        return rsl

    def middle_rect():
        rsl = []
        n = 140
        a = 0
        b = 280
        h = (b - a) // n
        sum = 0
        for i in range(1, n):
            sum += h * f(i - 1 + h // 2)
            rsl.append(sum)
        return rsl

    def trapeze():
        rsl = []
        n = 140
        a = 0
        b = 280
        h = (b - a) // n
        sum = f(a) + f(b // 2)
        for i in range(n - 1):
            sum += h * f(i)
            rsl.append(sum)
        return rsl

    def simpson():
        rsl = []
        n = 70
        a = 0
        b = 280
        h = (b - a) // (2 * n)
        sum = f(a) + f(b // 2)
        for i in range(n - 1):
            sum += (h * 2 * f(2 * i)) // 3
            rsl.append(sum)
        for i in range(n):
            sum += (h * 4 * f(2 * i - 1)) // 3
            rsl.append(sum)
        return rsl

    res_5 = simpson()
    res_4 = trapeze()
    res_3 = middle_rect()
    res_2 = right_rect()
    res = left_rect()

    plt.title("Численное интегрирование")
    plt.xlabel("x[mm]")
    plt.ylabel("y[mm]")
    plt.plot(x_s_1, res, 'b')
    plt.plot(x_s_1, res_2, 'r')
    plt.plot(x_s_1, res_3, 'm')
    plt.plot(x_s_1, res_4, 'k')
    plt.plot(x_s_1, res_5, 'c')
    plt.show()

def math_model():
    class LagrangeInterpolation:
        def __init__(self, x_build, y_build):
            self.x_build = x_build
            self.y_build = y_build
            self.polynomial_degree = len(x_build) - 1

        def process(self, x):
            result = 0

            for j in range(self.polynomial_degree + 1):
                result += self.y_build[j] * self._base_polynomial(x, j)

            return result

        def _base_polynomial(self, x, j):
            result = 1

            for m in range(self.polynomial_degree + 1):
                if m != j:
                    result *= (x - self.x_build[m]) / (self.x_build[j] - self.x_build[m])

            return result

    def x_speed(c, y, tga):
        if c - 9.8 * y > 0:
            return (2 * ((c - 9.8 * y) / (1 + tga ** 2))) ** (1 / 2)
        return -(-2 * (c - 9.8 * y) / (1 + tga ** 2)) ** (1 / 2)

    def ball_launcher(x, lagrange, X, Y):
        c = 9.8 * lagrange.process(x)
        eps = 0.01
        dx = 0.005
        dy = lagrange.process(x) - lagrange.process(x + dx)

        right = dy > 0
        if right:
            x += dx
        else:
            x -= dx

        ax = plt.subplot()
        x_arr = []
        speed_arr = []
        plt.pause(1)

        while True:
            y = lagrange.process(x)
            dy = y - lagrange.process(x + dx)
            tga = dy / dx

            speed = x_speed(c, y, tga)
            speed_arr.append(speed)
            x_arr.append(x)

            x += speed * (1 if right else -1)

            if speed < eps:
                #break
                right = not right

            ax.clear()
            ax.set_xlabel("x[mm]")
            ax.set_ylabel("L(x)[mm]")
            ax.set_title("Математическая модель")
            ax.scatter(x, lagrange.process(x), color="red")
            ax.plot(X, Y, color="black")
            plt.pause(0.01 * abs(speed))

        return x_arr, speed_arr

    x_s = range(0, 281, 2)
    y_1 = [172, 136, 80, 25, 6, 11, 8, 51, 118, 134, 172]
    x_1 = [0, 22, 44, 66, 110, 132, 154, 198, 242, 264, 280]
    s = LagrangeInterpolation(x_1, y_1)
    y_lagrange = [s.process(x) for x in x_s]
    x_start = int(input())
    x_mas, speed_mas = ball_launcher(x_start, s, list(x_s), y_lagrange)
    plt.title("График скорости")
    plt.xlabel("x[mm]")
    plt.ylabel("y[mm]")
    plt.plot(x_mas, speed_mas)
    plt.grid()
    plt.show()


digitization(x_s, y_s)  # оцифровка ямы
newton(x_s, y_s)  # полином ньютона
lag_pol = lagrange(x_s, y_s)  # полином лагранжа
cubic_interp1d(x_s, y_s)  # сплайн интерполяция
integr()
dif(x_s, y_s)  # численное дифференцирование
math_model()  # математическая модель















