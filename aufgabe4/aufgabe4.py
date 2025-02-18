"""
    Datum: 17.01.2020

    Modul aufgabe4.py, enthält Methoden zum Lösen von überbestimmten linearen Gleichungssystemen
    und zum Approximieren einer hängenden Kette mithilfe von Exponentialfunktionen.
    Zur Visualisierung der Experimente dienen Funktionen, die die Kettenapproximation, die
    Konditionen relevanter Matrizen und das Residuum, das heißt den Fehler der Approximation
    zeichnen.
"""

import sys

import numpy as np
import scipy.linalg as linalg

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt



def solve_least_squares(a, b):
    """ Approximiert mit der Methode der kleinsten Quadrate ein überbestimmtes lineares
        Gleichungssystem a*x = b für eine mxn-Matrix vollen Spaltenrangs a (m >= n) und
        einen Vektor b.

        Parameter
        ---------
        a : numpy.ndarray
            mxn-Matrix vollen Spaltenrangs (m >= n)
        b : numpy.ndarray
            1D-Vektor mit m Einträgen

        Returns
        -------
        numpy.ndarray
            1D-Vektor mit n Einträgen, Lösung x von a*x = b
        float
            Residuum ||ax - b||

        Raises
        ------
        ValueError
            Falls Matrix a nicht vollen Spaltenrangs ist. In diesem Fall terminiert das Programm
    """

    try:
        q, r = linalg.qr(a)

        if (r[0, 0] == 0 or r[1, 1] == 0 or r[2, 2] == 0):
            print("Fehler: Matrix A nicht vollen Spaltenrangs")
            raise ValueError

        z = (q.T).dot(b)
        x = solve_r(r, z)

        residuum = linalg.norm(a.dot(x)-b, 2)

        return x, residuum

    except ValueError:
        print("Programm bricht ab")
        sys.exit(-1)

def get_conditions(a):
    """ Gibt für eine nicht notwendigerweise quadratische mxn-Matrix (m >= n) a vollen Spaltenrangs
        die Konditionen von a und des Produkts der transponierten a^T und a aus.

        Parameter
        ---------
        a : numpy.ndarray
            mxn-Matrix (m >= n) vollen Spaltenrangs

        Returns
        -------
        float
            Kondition von a
        float
            Kondition von a^T * a
    """

    try:
        a_plus = linalg.pinv(a)
        ata_inv = linalg.inv((a.T).dot(a))

        cond_a = linalg.norm(a, 2)*linalg.norm(a_plus, 2)
        cond_ata = linalg.norm((a.T).dot(a), 2)*linalg.norm(ata_inv, 2)

        return cond_a, cond_ata

    except linalg.LinAlgError:
        print("Fehler: Matrix A^T A nicht invertierbar")
        print("Programm bricht ab")
        sys.exit(-1)

def solve_r(r, z):
    """ Löst für eine obere mxn-Dreiecksmatrix r (m >= n) vollen Spaltenrangs und einen Vektor z das
        lineare Gleichungssystem r*x = z und gibt die Lösung x zurück.

        Parameter
        ---------
        r : numpy.ndarray
            Obere mxn-Dreiecksmatrix (m >= n) vollen Spaltenrangs
        z : numpy.ndarray
            Vektor der Länge m

        Returns
        -------
        numpy.ndarray
            Lösungsvektor x von r*x = z der Länge n
    """

    n = r.shape[1]
    x = np.zeros(n)

    for l in range(n-1, -1, -1):
        summ = 0
        for k in range(l, n):
            summ += r[l, k]*x[k]
        x[l] = (z[l] - summ) / r[l, l]

    return x



def read_data_set(path, selection):
    """ Liest für gegebenen Dateipfad eine Auswahl von Datenpunkten (x,y) aus einem Datensatz ein
        und gibt sie als Liste aus.

        Parameter
        ---------
        path : string
            Dateipfad des Datensatz
        selection : iterable (ints)
            Gibt eine Auswahl von Datenpunkten (Zeilen), die in eine Liste eingelesen werden
            sollen, an

        Returns
        -------
        iterable (tuples of ints)
            Liste von eingelesenen Datenpunkten
    """

    try:
        f = open(path, "r")
    except FileNotFoundError:
        print("\"" + path + "\"", "nicht gefunden")
        print("Programm bricht ab")
        sys.exit(-1)

    chain = []

    for line in f:
        entries = line.split(',')
        chain.append((float(entries[0]), float(entries[1])))

    for i in range(len(chain)-1, -1, -1):
        if i not in selection:
            del chain[i]

    chain = sorted(chain, key=lambda entry: entry[0])

    return chain



def approx_chain(chain, d):
    """ Liefert für gegebenen Datensatz (Punkte einer Kette) und Lösungsparameter d die anderen
        Lösungsparameter a, b, c, mit denen die Kette beschrieben werden kann und das Residuum
        der kleinste-Quadrate-Lösung, das heißt den Approximationsfehler.

        Parameter
        ---------
        chain : iterable
            Iterable von Datenpunkten (x, y) (Tupel) einer Kette

        Returns
        -------
        numpy.ndarray
            1D-Vektor mit 3 Einträgen den Lösungsparametern a, b, c
        float
            Residuum der kleinste-Quadrate-Lösung
    """

    n = len(chain)

    a = np.array([[np.e**(d*chain[j][0]), np.e**(-d*chain[j][0]), 1] for j in range(n)])
    b = np.array([chain[j][1] for j in range(n)])
    return solve_least_squares(a, b)

def approx_d(chain):
    """ Liefert Approximation des Lösungsparameters d im Intervall [0.1, 0.5] für
        gegebenen Datensatz.

        Parameter
        ---------
        chain : iterable
            Sammlung von Datenpunkten (x, y) (tuples) einer Kette

        Returns
        -------
        float
            Approximation des Lösungsparameters d
    """

    best_d = 0.1
    best_error = approx_chain(chain, best_d)[1]

    for d in np.linspace(0.1, 0.5, 100)[1:]:
        error = approx_chain(chain, d)[1]
        if error < best_error:
            best_error = error
            best_d = d

    return best_d


def plot_residuum_for_d(chain_list, labels=0):
    """ Berechnet und plottet die Residuumsnorm ||Ax - b|| für Lösungsapproximationen gegebener
        Datensätze über Werte von d zwischen 0.1 und 0.5.

        Parameter
        ---------
        chain_list : iterable
            Sammlung von Datensätzen (Listen von Datenpunkten (x, y) (tuples)), für die die Lösung
            approximiert und das Residuum gezeichnet werden soll
        ( labels : iterable (strings)
              Sammlung von Legendenbeschreibungen (strings) für die gegebenen Datensätze.
              Bei Standard 0 wird keine Legende erstellt )
    """

    plot_interval = np.linspace(0.1, 0.5, 100)

    for i in range(len(chain_list)):
        residuum_list = []

        chain = chain_list[i]
        for d in plot_interval:
            residuum_list.append(approx_chain(chain, d)[1])

        if labels != 0:
            plt.plot(plot_interval, residuum_list, label=labels[i])
        else:
            plt.plot(plot_interval, residuum_list)

    if labels != []:
        plt.legend()

    plt.title("Residuumsnorm ||Ax - b||")
    plt.xlabel("d")
    plt.ylabel("||Ax - b||")
    plt.show()

def plot_conditions_for_d(chain_list, labels=0):
    """ Berechnet und plottet die Konditionen der Matrizen A und A^T * A für gegebene Datensätze
        über Werte von d zwischen 0.1 und 0.5.

        Parameter
        ---------
        chain_list : iterable (tuples of floats)
            Sammlung von Datensätzen (Listen von Datenpunkten (x, y) (tuples)), für die die
            Konditionen der zum Problem korrespondierenden Matrizen berechnet werden sollen
        ( labels : iterable (strings)
              Sammlung von strings, enthält die Legendenbeschreibungen der Plots zu den Datensätzen
              aus chain_list. Beim Standard 0 wird keine Legende erstellt )
    """
    plot_interval = np.linspace(0.1, 0.5, 100)
    colors = ["r", "g", "b"]

    for i in range(len(chain_list)):
        a_cond_list = []
        ata_cond_list = []

        chain = chain_list[i]
        for d in plot_interval:
            a = np.array([[np.e**(d*chain[j][0]), np.e**(-d*chain[j][0]), 1]
                          for j in range(len(chain))])
            conditions = get_conditions(a)
            a_cond_list.append(conditions[0])
            ata_cond_list.append(conditions[1])

        c = colors[i%3]
        if labels != 0:
            plt.plot(plot_interval, a_cond_list, ""+c, label=labels[i][0])
            plt.plot(plot_interval, ata_cond_list, "--"+c, label=labels[i][1])
        else:
            plt.plot(plot_interval, a_cond_list, ""+c)
            plt.plot(plot_interval, ata_cond_list, "--"+c)

    if labels != 0:
        plt.legend()

    plt.yscale("log")
    plt.title("Kondition von A und A^T*A")
    plt.xlabel("d")
    plt.ylabel("Kondition")
    plt.show()

def plot_multiple_approximations(chain_list, d_list, labels=0, title="Kettenapproximationen"):
    """ Berechnet die Kettenapproximationen für verschiedene Datensätze und Parameter d und
        stellt sie und die erste in chain_list gegebene Datenreihe in einem Schaubild dar.

        Parameter
        ---------
        chain_list : iterable
            Sammlung von chains, Listen von Datenpunkten (tuples (x, y)), für die jeweils das
            Problem gelöst werden soll. Die erste Datenreihe wird im Schaubild eingezeichnet
        d_list : iterable
            Sammlung von Listen, die für Index-korrespondierende chains aus chain_list die zu
            betrachtenden Werte für d enthalten
        ( labels : iterable (strings)
              Sammlung der Beschreibungen (strings) der einzelnen Plots für jeden Datensatz und
              jedes d, Beschreibungen erscheinen in Legende.
              Bei Standard 0 wird keine Legende erstellt
          title : string
              Titel des Plots, erscheint über dem Schaubild, Standard ist "Kettenapproximationen" )
    """

    y = lambda a, b, c, d, t: a*np.e**(d*t) + b*np.e**(-d*t) + c

    for i in range(len(chain_list)):
        chain = chain_list[i]
        plot_interval = np.linspace(chain[0][0], chain[-1][0])

        for d in d_list[i]:
            a, b, c = approx_chain(chain, d)[0]
            value_list = [y(a, b, c, d, t) for t in plot_interval]
            if labels != 0:
                plt.plot(plot_interval, value_list, label=labels[i]+", d = {:.3f}".format(d))
            else: plt.plot(plot_interval, value_list, label="d = {:.3f}".format(d))

    plt.plot([chain_list[0][i][0] for i in range(len(chain_list[0]))],
             [chain_list[0][i][1] for i in range(len(chain_list[0]))],
             color="#000000", marker="x", linestyle="None", label="Datensatz")

    plt.legend()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def plot_datasets(ds_list, labels=0):
    """ Plottet Datenreihen.

        Parameter
        ---------
        ds_list : iterable
            Enthält die zu plottenden Listen von Tupeln (x,y)
        ( labels : iterable (strings)
              Sammlung von Legendenbeschreibungen (strings) für die gegenbenen Datensätze.
              Bei Standard 0 wird keine Legende erstellt )
    """

    for i in range(len(ds_list)):
        if labels != 0:
            plt.plot([point[0] for point in ds_list[i]], [point[1] for point in ds_list[i]],
                     label=labels[i], linestyle=":", marker='x')
        else:
            plt.plot([point[0] for point in ds_list[i]], [point[1] for point in ds_list[i]])

    if labels != 0:
        plt.legend()
    plt.title("Datensätze")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def main():
    """ main-Funktion von aufgabe4.py, wird bei Ausführung des Moduls als Hauptprogramm ausgeführt.
        Demonstriert Methoden aus dem Modul. Hierbei werden Plots der Kettenapproximationen für
        verschiedene Lösungsparameter d und Datensätze chain, chain2, chain3, und Plots des
        Residuums und der Konditionen von A und A^T * A über d für die Datensätze produziert.
    """

    chain = read_data_set("data/originaldaten", range(10))
    chain2 = read_data_set("data/daten2", range(10))
    chain3 = read_data_set("data/daten3", range(10))

    d = approx_d(chain)
    d2 = approx_d(chain2)
    d3 = approx_d(chain3)

    plot_datasets([chain, chain2, chain3], labels=["Originaldaten", "Datenreihe 2", "Datenreihe 3"])

    plot_multiple_approximations([chain], [[0.1, d, 0.366, 0.5]],
                                 title="Kettenapproximationen für Originaldaten")
    plot_multiple_approximations([chain2], [[0.1, d2, 0.366, 0.5]],
                                 title="Kettenapproximationen für Datensatz 2")
    plot_multiple_approximations([chain3], [[0.1, d3, 0.366, 0.5]],
                                 title="Kettenapproximationen für Datensatz 3")

    plot_multiple_approximations([chain, chain2, chain3], [[d], [d2], [d3]],
                                 labels=["Originaldaten", "Datensatz 2", "Datensatz 3"],
                                 title="Kettenapproximationen für optimale d")

    plot_multiple_approximations([chain], [[d]], title="Kettenapproximation Originaldaten")
    plot_multiple_approximations([chain2], [[d2]], title="Kettenapproximation Datensatz 2")
    plot_multiple_approximations([chain3], [[d3]], title="Kettenapproximation Datensatz 3")

    plot_residuum_for_d([chain, chain2, chain3],
                        labels=["Originaldaten", "Datensatz 2", "Datensatz 3"])

    labels = [("Originaldaten, cond(A)", "Originaldaten, cond(A^T A)"),
              ("Datensatz 2, cond(A)", "Datensatz 2, cond(A^T A)"),
              ("Datensatz 3, cond(A)", "Datensatz 3, cond(A^T A)")]
    plot_conditions_for_d([chain, chain2, chain3], labels)


if __name__ == "__main__":
    main()
