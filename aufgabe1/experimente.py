# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Pythonskript zur Reproduktion der Experimente zur Approximation von Ableitungen
im Programm derivative_approximation.py.

Datum: 08.11.2019
"""

import math
import numpy as np
import derivative_approximation as dv


def g1(x):
    """ Funktion g_1 """
    return math.sin(x)/x

def d_g1(x):
    """ Erste Ableitung von Funktion g_1 """
    return (x*math.cos(x)-math.sin(x))/(x*x)

def dd_g1(x):
    """ Zweite Ableitung von Funktion g_1 """
    return (-(x**2)*math.sin(x) - 2*x*math.cos(x) + 2*math.sin(x)) / (x**3)



def g01(x):
    """ Funktion g_0.1 """
    return math.sin(0.1*x)/x

def d_g01(x):
    """ Erste Ableitung von Funktion g_0.1 """
    return (math.cos(0.1*x))/(10*x) - (math.sin(0.1*x))/x**2

def dd_g01(x):
    """ Zweite Ableitung von Funktion g_0.1 """
    return -(math.sin(0.1*x))/(100*x)-(math.cos(0.1*x))/(5*x**2)+(2*math.sin(0.1*x))/(x**3)




def g001(x):
    """ Funktion g_0.01 """
    return (math.sin(0.01*x))/x

def d_g001(x):
    """ Erste Ableitung von Funktion g_0.01 """
    return (x*math.cos(0.01*x)-100*math.sin(0.01*x))/(100*x**2)

def dd_g001(x):
    """ Zweite Ableitung von Funktion g_0.01 """
    return ((-x*(math.sin(0.01*x)*x) +200*math.cos(0.01*x)) + 20000*math.sin(0.01*x)) / (10000*x**3)



def g10(x):
    """ Funktion g_10 """
    return math.sin(10*x)/x

def d_g10(x):
    """ Erste Ableitung von Funktion g_10 """
    return (10*math.cos(10*x))/(x) - (math.sin(10*x))/x**2

def dd_g10(x):
    """ Zweite Ableitung von Funktion g_10 """
    return -(100*math.sin(10*x))/(x) + (2*math.sin(10*x))/(x**3) - (20*math.cos(10*x))/(x**2)



def g100(x):
    """ Funktion g_100 """
    return math.sin(100*x)/x

def d_g100(x):
    """ Zweite Ableitung von Funktion g_100 """
    return (100*math.cos(100*x))/x - (math.sin(100*x))/x**2

def dd_g100(x):
    """ Zweite Ableitung von Funktion g_100 """
    return -(10000*math.sin(100*x))/x + (2*math.sin(100*x))/x**3 - (200*math.cos(100*x))/x**2



def main():
    """
    main-Funktion von experimente.py, wird bei Programmausführung als
    Hauptprogramm automatisch ausgeführt. Es werden verschiedene Plots im Rahmen
    der Experimente berechnet und angezeigt.

    Parameter: keine
    Ausgaben: keine
    """


    #Aufgabe 1.2a

    #h=pi/3:
    das = dv.FiniteDifference(np.pi/3, g1, d_g1, dd_g1)
    das.plot_functions(np.pi, np.pi*3, 100)

    #h=pi/4:
    das2 = dv.FiniteDifference(np.pi/4, g1, d_g1, dd_g1)
    das2.plot_functions(np.pi, np.pi*3, 100)

    #h=pi/5:
    das3 = dv.FiniteDifference(np.pi/5, g1, d_g1, dd_g1)
    das3.plot_functions(np.pi, np.pi*3, 100)

    #h=pi/10:
    das4 = dv.FiniteDifference(np.pi/10, g1, d_g1, dd_g1)
    das4.plot_functions(np.pi, np.pi*3, 100)


    #Aufgabe 1.2b

    #minimale Schrittweite h = 10^-4:
    x0, xe = math.log10(0.0001), math.log10(1)
    h_list = [10**(x0 + (xe-x0)*i/1000) for i in range(1, 1001)]
    das4.plot_errors(h_list, np.pi, np.pi*3, 1000)

    #minimale Schrittweite h = 10^-7:
    x0, xe = math.log10(0.0000001), math.log10(1)
    h_list = [10**(x0 + (xe-x0)*i/1000) for i in range(1, 1001)]
    das4.plot_errors(h_list, np.pi, np.pi*3, 1000)

    #minimale Schrittweite h = 10^-9:
    x0, xe = math.log10(0.000000001), math.log10(1)
    h_list = [10**(x0 + (xe-x0)*i/1000) for i in range(1, 1001)]
    das4.plot_errors(h_list, np.pi, np.pi*3, 1000)

    #minimale Schrittweite h = 10^-13
    x0, xe = math.log10(0.0000000000001), math.log10(1)
    h_list = [10**(x0 + (xe-x0)*i/1000) for i in range(1, 1001)]
    das4.plot_errors(h_list, np.pi, np.pi*3, 1000)



    #weiterführende Experiente (Aufgabe 1.3)

    obj1 = dv.FiniteDifference(1, g1, d_g1, dd_g1)
    obj2 = dv.FiniteDifference(1, g01, d_g01, dd_g01)
    obj3 = dv.FiniteDifference(1, g001, d_g001, dd_g001)
    obj4 = dv.FiniteDifference(1, g10, d_g10, dd_g10)
    obj5 = dv.FiniteDifference(1, g1, d_g100, dd_g100)

    #minimale Schrittweite h = 10^-9:
    x0, xe = math.log10(0.000000001), math.log10(1)
    h_list = [10**(x0 + (xe-x0)*i/1000) for i in range(1, 1001)]

    obj1.plot_errors(h_list, np.pi, np.pi*3, 1000)
    obj2.plot_errors(h_list, np.pi, np.pi*3, 1000)
    obj3.plot_errors(h_list, np.pi, np.pi*3, 1000)
    obj4.plot_errors(h_list, np.pi, np.pi*3, 1000)
    obj5.plot_errors(h_list, np.pi, np.pi*3, 1000)

    #minimale Schrittweite h = 10^-13:
    x0, xe = math.log10(0.0000000000001), math.log10(1)
    h_list = [10**(x0 + (xe-x0)*i/1000) for i in range(1, 1001)]

    obj1.plot_errors(h_list, np.pi, np.pi*3, 1000)
    obj2.plot_errors(h_list, np.pi, np.pi*3, 1000)
    obj3.plot_errors(h_list, np.pi, np.pi*3, 1000)
    obj4.plot_errors(h_list, np.pi, np.pi*3, 1000)
    obj5.plot_errors(h_list, np.pi, np.pi*3, 1000)


if __name__ == "__main__":
    main()
