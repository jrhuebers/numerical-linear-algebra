"""
    Date: 21.10.2019
"""

# -*- coding: utf-8 -*-
#!/usr/bin/env python

import math

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt





class FiniteDifference:
    """ Diese Klasse repräsentiert die Annäherung der Differenzenquotienten zur ersten und zweiten
        Ableitung einer Funktion.
        Außerdem wird eine exakte Berechnung der Abweichung zu den realten Ableitungen ermöglicht.

    Parameter
    ----------
    h : float
        Schrittweite der Approximation
    f : callable
        Betrachtete Funktion
    d_f : callable, optional
        Die analytische Approximation der ersten Ableitung von `f`.
    dd_f : callable, optional
        Die analytische Approximation der zweiten Ableitung von`f`.

    Attribute
    __________

       h : float
           Schrittweite der Approximation
    """

    def __init__(self, h, f, d_f=None, dd_f=None):
        """Die Methode __init__ wird aufgerufen, sobald ein Objekt der Klasse instanziiert wird."""
        self.h = h
        self.f = f
        self.d_f = d_f
        self.dd_f = dd_f

    def compute_errors(self, a, b, p):
        """ Berechnet eine Annährung an die Fehler zwischen der Approximation und der exakten
            Ableitung erster und zweiter Ordnung mit der Maximumsnorm.

        Parameter
        __________

        a, b : float
               Intervallgrenzen
        p : int
            Anzahl der Punkte, die das Intervall [a, b] zerlegen zur Approximation des Fehler
            mit der Maximumnorm.

        Ausgabe
        -------
        float
            Maximaler Fehler der Approximation der 1. Ableitung
        float
            Maximaler Fehler der Approximation der 2. Ableitung

        Raises
        ------
        ValueError
            Falls keine Ableitung der Funktion durch den Nutzer bereitgestellt wurde.
        """

        if self.d_f is None or self.dd_f is None:
            print("Keine analytischen Ableitungen gegeben.")
            raise ValueError

        list_d = [abs(self.d_f(a + i*abs(b-a)/p) - self.compute_df(a + i*abs(b-a)/p))
                  for i in range(p+1)]

        list_dd = [abs(self.dd_f(a + i*abs(b-a)/p) - self.compute_ddf(a + i*abs(b-a)/p))
                   for i in range(p+1)]

        return max(list_d), max(list_dd)

    def compute_df(self, x):
        """ Berechnet eine Annäherung an die erste Ableitung der Funktion f an der Stelle x.

        Parameter
        __________

        x : float
            Betrachtete Stelle x

        Ausgabe
        ________
        float
            Approximation der Ableitung von der Funktion f an der Stelle x
        """
        return (self.f(x + self.h) - self.f(x)) / self.h

    def compute_ddf(self, x):
        """ Berechnet eine Annäherung an die zweite Ableitung der Funktion f an der Stelle x.

        Parameter
        __________
        x : float
            Betrachtete Stelle

        Ausgabe
        ________
        float
            Approximation der zweiten Ableitung von der Funktion f an der Stelle x
        """
        return (self.f(x + self.h) - 2 * self.f(x) + self.f(x - self.h)) / self.h ** 2

    def plot_functions(self, a, b, p):
        """ Zeichnet die Funktionen f, die Approximationen der 1. und 2. Ableitung
            auf dem Intervall [a, b] (und gegebenenfalls f' und f'' vergleiche Aufgabenstellung)

        Parameter
        __________

        a, b : float
               Intervallgrenzen
           p : int
               Zerlegungsfeinheit des Intervalls [a, b, bestimmt den Abstand der x-Werte.

        Ausgabe
        _______

            keine

        """


        x_values = [a+i*(b-a)/p for i in range(p+1)]
        y_values = [self.f(x) for x in x_values]
        graph1, = plt.plot(x_values, y_values, label="f")

        y_values = [self.compute_df(x) for x in x_values]
        graph4, = plt.plot(x_values, y_values, "r", label="f' approximiert")

        y_values = [self.compute_ddf(x) for x in x_values]
        graph5, = plt.plot(x_values, y_values, "g", label="f'' approximiert")

        handle_list = [graph1, graph4, graph5]

        if self.d_f is None or self.dd_f is None:
            print("Analytische Ableitungen fehlen.")
        else:
            y_values = [self.d_f(x) for x in x_values]
            graph2, = plt.plot(x_values, y_values, "g--", label="f' real")

            y_values = [self.dd_f(x) for x in x_values]
            graph3, = plt.plot(x_values, y_values, "r--", label="f'' real")

            handle_list.insert(1, graph3)
            handle_list.insert(1, graph2)



        plt.legend(handles=handle_list)

        plt.title("Funktion, Ableitungen und Approximationen")
        plt.xlabel("x")

        plt.show()


    def plot_errors(self, h_list, a, b, p):
        """ Zeichnet die Annäherungen an die Approximationsfehler für die 1. und 2. Ordnung.

        Parameter
        _________

        h_list : float
                 Minimale Schrittweite
          a, b : float
                 Intervallgrenzen
             p : int Zerlegungsfeinheit des Intervalls [a, b, bestimmt den Abstand der x-Werte.

        Ausgabe
        ________

            keine
        """

        error_list = []
        for h in h_list:
            self.h = h
            error_list.append(self.compute_errors(a, b, p))

        graph1, graph2 = plt.plot(h_list, error_list)
        graph3, = plt.plot(h_list, h_list, ":")
        graph4, = plt.plot(h_list, list(map(lambda h: h**2, h_list)), ":")
        graph5, = plt.plot(h_list, list(map(lambda h: h**3, h_list)), ":")

        plt.legend([graph1, graph2, graph3, graph4, graph5], ["Fehler f\'", "Fehler f\'\'",
                                                              "h^1", "h^2", "h^3"])

        plt.title("Maximaler Fehler und Approximationsschrittweite")

        plt.xscale("log")
        plt.yscale("log")

        plt.xlabel("Schrittweite h")
        plt.ylabel("Maximaler Fehler")

        plt.show()





def g1(x):
    """ Diese Funktion beinhaltet die Beispielfunktion sin(x)/x und gibt den Wert der Funktion an
        Stelle x aus.

    Parameter
    _________
         x : float
             Betrachtete Stelle

    Ausgabe
    ______
    float
        mit der Standardlibrary math berechneter Funktionswert der Funkion an Stelle x
    """

    return math.sin(x)/x

def d_g1(x):
    """ Diese Funktion beinhaltet die 1. Ableitung der Beispielfunktion,
        g1'(x) = cos(x)-ain(x)/x**2, und gibt ihren Wert an Stelle x aus.

    Parameter
    _________
         x : float
             Betrachtete Stelle

    Ausgabe
    ______
    float
        mit der Standardlibrary math berechneter Wert der Ableitung von f
    """

    return (x*math.cos(x)-math.sin(x))/(x*x)

def dd_g1(x):
    """ Diese Funktion gibt den Wert der zweiten Ableitung der Beispielfunktion,
        g1''(x) = (-x(**2)*sin(x) - 2x*cos(x) + 2*sin(x)) / x**3, an Stelle x aus.
    _________
         x : float
             Betrachtete Stelle

    Ausgabe
    ______
    float
        mit der Standardlibrary math berechneter Funktionswert der zweiten Ableitung an Stelle x
    """

    return (-(x**2)*math.sin(x) - 2*x*math.cos(x) + 2*math.sin(x)) / (x**3)


def secure_input(typ, message):
    """ Diese Funktion ermöglicht die Eingabe von Zahlen vom Typ int oder float durch den Nutzer.
        Ungültige Eingaben werden abgefangen und der Nutzer zur erneuten Eingabe aufgefordert.

    Parameter
    __________
        typ : int oder float
              Gewünschter Datentyp
    message : string
              vom User eingegebener Input

    Ausgabe
    ________
    int oder float
        Zulässige Usereingabe
    """

    while True:
        try:
            inp = typ(input(message))
            if inp != 0:
                return inp
        except ValueError:
            print("Eingabe muss vom Typ {} sein.".format(str(typ)))






def main():
    """ Main-Funktion von derivative_approximation.py, bietet Testdurchläufe anhand der Funktion
        g1(x) = sin(x)/x. Sie wird abgerufen, wenn das Programm als Hauptprogramm gestartet wird.
        Im Standarddurchlauf wird das Intervall [pi, 3*pi] für die Rechnungen und Plots verwendet.
        die Approximationsschrittweite h und ein Berechnungspunkt x können vom Nutzer eingegeben
        werden.
        Im Nicht-Standarddurchlauf kann der Nutzer auch Intervallgrenzen und Plotauflösung angeben.
        Die einzelnen Programmpunkte können hier auch mit neuen Werten wiederholt werden.

    Parameter: keine
    Ausgabe: keine
    """

    print()
    print("Testdurchlauf von derivate_approximation.py anhand der Funktion g1(x) = sin(x)/x")
    print("================================================================================")
    print()

    while True:
        standard = input("Standard-Durchlauf starten? (j/n): ")[0]
        if standard in "jnJN":
            break


    if standard in 'nN':

        print("Approximationen der Ableitungen:")
        print("--------------------------------")
        print()

        while True:

            while True:
                h = float(input("Schrittweite h = "))
                if h != 0:
                    break
                print('Ungültige Eingabe, Division durch Null unzulässig.')

            obj = FiniteDifference(h, g1, d_g1, dd_g1)


            while True:
                x = float(input("Approximationsstelle x = "))
                if x != 0:
                    break
                print("Eingabe ungültig, Division durch 0 nicht erlaubt.")


            print("Approximationen:")
            print("f'(x)  ≈ {}".format(obj.compute_df(x)))
            print("f''(x) ≈ {}".format(obj.compute_ddf(x)))
            print()
            print("reale Werte:")
            print("f'(x)  = {}".format(obj.d_f(x)))
            print("f''(x) = {}".format(obj.dd_f(x)))
            print()

            while True:
                repeat = input("Mit neuen Werten wiederholen? (j/n): ")[0]
                if repeat in "jn":
                    break
            print()
            if repeat == 'n':
                print()
                break

        print("Plot der Funktion, Ableitungen, Approximationen:")
        print("------------------------------------------------")
        print()

        while True:

            while True:
                a, b = float(input("Intervallgrenze a = ")), float(input("Intervallgrenze b = "))
                if (a < 0 and b < 0) or (a > 0 and b > 0):
                    break
                print("Ungültige Eingabe, Intervall darf die Null nicht enthalten, "
                      + "g1 in 0 nicht definiert.")

            n_p = secure_input(int, "Anzahl zu plottender Punkte, n_p = ")

            obj.plot_functions(a, b, n_p)

            print()

            while True:
                repeat = input("Erneut plotten? (j/n): ")[0]
                if repeat in "jn":
                    break

            print()
            if repeat == 'n':
                print()
                break


        print("Approximationsfehler berechnen:")
        print("-------------------------------")
        print()

        while True:

            while True:
                carry = input("Zuletzt eingegebene Werte übernehmen? "
                              + "a = {}, b = {}? (j/n): ".format(a, b)).lower()[0]
                if carry in "jn":
                    break

            if carry == 'n':
                while True:
                    a = float(input("Intervallgrenze a = "))
                    b = float(input("Intervallgrenze b = "))
                    if (a < 0 and b < 0) or (a > 0 and b > 0):
                        break
                    print("Ungültige Eingabe, Intervall darf die Null nicht enthalten, "
                          + "g1 in 0 nicht definiert.")


            p = secure_input(int, "Anzahl zu prüfender Stellen, p = ")

            print()
            print("Maximaler Fehler: {}".format(obj.compute_errors(a, b, p)))

            while True:
                repeat = input("Wiederholen? (j/n): ")
                if repeat in "jn":
                    break

            print()
            if repeat == 'n':
                print()
                break

        print("Fehlerplot:")
        print("-----------")
        print()

        while True:

            while True:
                carry = input("Zuletzt eingegebene Werte übernehmen? "
                              + "a = {}, b = {}, p = {}? (j/n): ".format(a, b, p))[0]
                if carry in "jn":
                    break

            if carry == 'n':
                while True:
                    a = float(input("Intervallgrenze a = "))
                    b = float(input("Intervallgrenze b = "))
                    if (a < 0 and b < 0) or (a > 0 and b > 0):
                        break
                    print("Ungültige Eingabe, Intervall darf die Null nicht enthalten, "
                          + "g1 in 0 nicht definiert.")

                p = secure_input(int, "Anzahl zu prüfender Stellen, p = ")

            h_min = secure_input(float, "Minimale Schrittweite h_min = ")
            h_max = secure_input(float, "Maximale Schrittweite h_max = ")

            plot_resolution = secure_input(int, "Anzahl der Plotpunkte: ")

            print()

            x0, xe = math.log10(h_min), math.log10(h_max)
            h_list = [10**(x0 + (xe-x0)*i/plot_resolution) for i in range(1, plot_resolution+1)]
            obj.plot_errors(h_list, a, b, p)

            while True:
                repeat = input("Plot Wiederholen? (j/n): ")[0]
                if repeat in "jn":
                    break

            print()
            if repeat == 'n':
                print()
                break




    if standard == 'j':

        print()
        a = math.pi
        b = 3*math.pi
        print('Das vorgegebene Intervall ist: [{}, {}]'.format(a, b))

        # Hier wird der User zur Angabe der Schrittweite aufgefordert:
        h = secure_input(float, "Approximationsschrittweite h = ")
        # Initialisierung der Beispielfunktion:
        obj = FiniteDifference(h, g1, d_g1, dd_g1)
        # Hier wird der User zur Angabe der Berechnungsstelle aufgefordert:
        x = secure_input(float, "Berechnungstelle x = ")

        print()
        print("Approximationen:")
        print("f'(x)  ≈ {}".format(obj.compute_df(x)))
        print("f''(x) ≈ {}".format(obj.compute_ddf(x)))
        print()
        print("Exakte Werte:")
        print("f'(x)  = {}".format(obj.d_f(x)))
        print("f''(x) = {}".format(obj.dd_f(x)))
        print()

        print("Plot der Funktion, Ableitungen, Ableitungsapproximationen:")
        p = secure_input(int, "Anzahl zu plottender Punkte: ")
        obj.plot_functions(a, b, p)
        print()

        print("Fehlerapproximation:")
        p = secure_input(int, "Unterteilungsfeinheit vom Intervall, p = ")
        tup = obj.compute_errors(a, b, p)
        print("Maximaler Fehler g1':  {}\nMaximaler Fehler g1'': {}".format(tup[0], tup[1]))
        print()

        print("Fehlerplot:")
        print("Empfohlene Werte: p = 1000, h_min = 0.000000001 h_max = 1, Plotauflösung: 500")
        p = secure_input(int, "Unterteilungsfeinheit vom Intervall, p = ")
        h_min = secure_input(float, "Minimale Schrittweite h_min = ")
        h_max = secure_input(float, "Maximale Schrittweite h_max = ")
        plot_resolution = secure_input(int, "Anzahl zu plottender Punkte: ")

        x0, xe = math.log10(h_min), math.log10(h_max)
        h_list = [10**(x0 + (xe-x0)*i/plot_resolution) for i in range(1, plot_resolution+1)]
        obj.plot_errors(h_list, a, b, p)

        print()
        print("Testdurchlauf beendet.")



if __name__ == "__main__":
    main()
