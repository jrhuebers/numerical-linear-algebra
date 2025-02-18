"""
    Modul experimente.py
    Datum: 07.02.2020

    Enthält Methoden zur Untersuchung und zum Vergleich der Eigenschaften der LU-Zerlegung-
    Gesamtschritt- und Einzelschrittverfahren zum Lösen von linearen Gleichungssystemen
    Ax = b und konkreter zum Lösen eines Poisson-Problems.
    Für den zugehörigen Bericht werden Plots produziert.
"""


import numpy as np
import scipy.linalg

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import rhs
import block_matrix
import linear_solvers



def u(x, k=1):
    """ In Aufgabe 3.2a gegebene Funktion u: Omega -> R.

    Parameters
    ----------
    x : numpy.nd_array
        Punkt in Omega.
    (k : int
        Parameter k wie in Funktion f. Damit u das durch f gegebene Poisson-Problem loest,
        muessen die Parameter k in beiden Funktionen stets uebereinstimmen. Standard k=1. )

    Returns
    -------
    float
        Funktionswert von u in x.
    """

    product = 1
    for x_l in x:
        product *= x_l*np.sin(k*np.pi*x_l)

    return product

def f(x, k=1):
    """ Funktion f: Omega -> R, sodass u die Loesung des Poisson-Problems zu f ist.

    Parameters
    ----------
    x : numpy.nd_array
        Punkt in Omega.
    (k : int
        Parameter k wie bei u, muss bei allen Rechnungen mit dem Parameter in u
        uebereinstimmen, damit u das Problem zu f loest. Standardmaessig k=1.)

    Returns
    -------
    float
        Funktionswert von f in x.
    """

    summ = 0
    for x_r in x:
        summ += 1/np.tan(k*np.pi*x_r)/x_r

    return -u(x)*(2*k*np.pi*summ - x.shape[0]*k**2*np.pi**2)



def solve_gs(d, N, params=dict(eps=1e-8, max_iter=1000, min_red=0)):
    """ Funktion zum Approximieren der Lösung des Poisson-Problems, mithilfe des Gesamtschritt-
    Verfahrens.

    Parameter
    ---------
    d : int
        Dimension des Poisson-Problems
    N : int
        Diskretisierungsfeinheit zum Aufstellen der Matrix
    params : dict, optional
        dictionary mit den Terminierungsbedingungen für die iterativen Verfahren

        eps : float
            Toleranz der unendlich-Norm der Residuen. Wenn kleiner oder gleich 0,
            gibt es keine solche Einschränkung.
        max_iter : int
            maximale Anzahl von Iterationen, die die Iterationsverfahren durchführen.
            Wenn kleiner oder gleich 0, gibt es keine solche Beschränkung.
        min_red : float
            minimale Abnahme des Residuums bezüglich der unendlich-Norm in jedem
            Iterationsschritt für die GS- und ES-Verfahren. Wenn kleiner oder gleich 0,
            gibt es keine solche Einschränkung.

    Returns
    -------
    str
        Abbruchgrund des Iterationsverfahrens
    list
        die Iterierten des Algorithmus
    list
        Residuen der Iterierten
    """

    A = block_matrix.BlockMatrix(d, N).get_sparse()
    return linear_solvers.solve_gs(A, rhs.rhs(d, N, f), np.zeros((N-1)**d),
                                   params=params)

def solve_es(d, N, params=dict(eps=1e-8, max_iter=1000, min_red=0)):
    """ Funktion zum Approximieren der Lösung des Poisson-Problems, mithilfe des
    Einzelschrittverfahrens.

    Parameter
    ---------
    d : int
        Dimension des Poisson-Problems
    N : int
        Diskretisierungsfeinheit zum Aufstellen der Matrix
    params : dict, optional
        dictionary mit den Terminierungsbedingungen für die iterativen Verfahren

        eps : float
            Toleranz der unendlich-Norm der Residuen. Wenn kleiner oder gleich 0,
            gibt es keine solche Einschränkung.
        max_iter : int
            maximale Anzahl von Iterationen, die die Iterationsverfahren durchführen.
            Wenn kleiner oder gleich 0, gibt es keine solche Beschränkung.
        min_red : float
            minimale Abnahme des Residuums bezüglich der unendlich-Norm in jedem
            Iterationsschritt für die GS- und ES-Verfahren. Wenn kleiner oder gleich 0,
            gibt es keine solche Einschränkung.

    Returns
    -------
    str
        Abbruchgrund des Iterationsverfahrens
    list
        die Iterierten des Algorithmus
    list
        Residuen der Iterierten
    """

    A = block_matrix.BlockMatrix(d, N).get_sparse()
    return linear_solvers.solve_es(A, rhs.rhs(d, N, f), np.zeros((N-1)**d),
                                   params=params)



def plot_max_abs_error(d, n, params=dict(eps=0, max_iter=1000, min_red=0)):
    """ Produziert Plot des maximalen absoluten Fehlers der Lösung des Poisson-Problems
    für Einzelschritt- und Gleichschrittverfahren und LU-Zerlegung. Für die
    Iterationsverfahren können Abbruchbedingungen eps, max_iter, min_red angegeben werden.

    Parameter
    ---------
    d : int
        Dimension des Poisson-Problems
    N : int
        Diskretisierungsfeinheit zum Aufstellen Matrix
    params : dict, optional
        dictionary mit den Terminierungsbedingungen für die iterativen Verfahren

        eps : float
            Toleranz der unendlich-Norm der Residuen. Wenn kleiner oder gleich 0,
            gibt es keine solche Einschränkung.
        max_iter : int
            maximale Anzahl von Iterationen, die die Iterationsverfahren durchführen.
            Wenn kleiner oder gleich 0, gibt es keine solche Beschränkung.
        min_red : float
            minimale Abnahme des Residuums bezüglich der unendlich-Norm in jedem
            Iterationsschritt für die GS- und ES-Verfahren. Wenn kleiner oder gleich 0,
            gibt es keine solche Einschränkung.
    """

    errors_gs = [rhs.compute_error(d, n, hat_u, u) for hat_u in solve_gs(d, n, params)[1]]
    errors_es = [rhs.compute_error(d, n, hat_u, u) for hat_u in solve_es(d, n, params)[1]]

    plt.plot(range(len(errors_gs)), errors_gs, label="GS")
    plt.plot(range(len(errors_es)), errors_es, label="ES")

    conditions_string = ""
    if params["eps"] > 0:
        conditions_string += "eps = {}, ".format(params["eps"])
    if params["max_iter"] >= 1:
        conditions_string += "max_iter = {}, ".format(params["max_iter"])
    if params["min_red"] > 0:
        conditions_string += "min_red = {}, ".format(params["min_red"])

    plt.title("max. abs. Fehler (" + conditions_string[0:-2] + ")")
    plt.xlabel("n (Iteration)")
    plt.ylabel("Fehler")

    plt.yscale("log")

    plt.legend()
    plt.show()

def plot_norm_residuum(d, n, params=dict(eps=0, max_iter=1000, min_red=0)):
    """ Produziert Plot der Residuennorm der Lösung des Poisson-Problems
    für Einzelschritt- und Gleichschrittverfahren und LU-Zerlegung. Für die iterativen
    Verfahren können Abbruchbedingungen eps und min_red gegeben werden.

    Parameter
    ---------
    d : int
        Dimension des Poisson-Problems
    N : int
        Diskretisierungsfeinheit zum Aufstellen Matrix
    params : dict, optional
        dictionary mit den Terminierungsbedingungen für die iterativen Verfahren

        eps : float
            Toleranz der unendlich-Norm der Residuen. Wenn kleiner oder gleich 0,
            gibt es keine solche Einschränkung.
        max_iter : int
            maximale Anzahl von Iterationen, die die Iterationsverfahren durchführen.
            Wenn kleiner oder gleich 0, gibt es keine solche Beschränkung.
        min_red : float
            minimale Abnahme des Residuums bezüglich der unendlich-Norm in jedem
            Iterationsschritt für die GS- und ES-Verfahren. Wenn kleiner oder gleich 0,
    """

    res_gs = [scipy.linalg.norm(res, np.inf) for res in solve_gs(d, n, params)[2]]
    res_es = [scipy.linalg.norm(res, np.inf) for res in solve_es(d, n, params)[2]]

    plt.plot(range(len(res_gs)), res_gs, label="GS")
    plt.plot(range(len(res_es)), res_es, label="ES")


    conditions_string = ""
    if params["eps"] > 0:
        conditions_string += "eps = {}, ".format(params["eps"])
    if params["max_iter"] >= 1:
        conditions_string += "max_iter = {}, ".format(params["max_iter"])
    if params["min_red"] > 0:
        conditions_string += "min_red = {}, ".format(params["min_red"])

    plt.title("Residuumsnorm (" + conditions_string[0:-2] + ")")
    plt.xlabel("n (Iteration)")
    plt.ylabel("Residuumsnorm")
    plt.yscale("log")

    plt.legend()
    plt.show()



def plot_comparison_residuals(d, N_max, params=dict(eps=1e-8, max_iter=0, min_red=0)):
    """ Produziert Plot der unendlich-Normen der Residuen der Lösungen des Poisson-Problems
    für Gesamtschritt-, Einzelschritt- und LU-Zerlegungs-Verfahren über die
    Diskretisierungsfeinheit. Für die Iterationsverfahren können Abbruchbedingungen
    eps, max_iter, min_red angegeben werden.

    Parameter
    ---------
    d : int
        Dimension des Poisson-Problems
    N_max : int
        maximale zu testende Diskretisierungsfeinheit, für alle Feinheiten zwischen 2 und
        N_max wird das Problem mit den verschiedenen Verfahren gelöst und die Residuen
        geplottet.
    params : dict, optional
        dictionary mit den Terminierungsbedingungen für die iterativen Verfahren

        eps : float
            Toleranz der unendlich-Norm der Residuen. Wenn kleiner oder gleich 0,
            gibt es keine solche Einschränkung.
        max_iter : int
            maximale Anzahl von Iterationen, die die Iterationsverfahren durchführen.
            Wenn kleiner oder gleich 0, gibt es keine solche Beschränkung.
        min_red : float
            minimale Abnahme des Residuums bezüglich der unendlich-Norm in jedem
            Iterationsschritt für die GS- und ES-Verfahren. Wenn kleiner oder gleich 0,
            gibt es keine solche Einschränkung.
    """

    gs_residuum_list, es_residuum_list, lu_residuum_list = [], [], []
    for N in range(2, N_max+1):
        norm = lambda x: scipy.linalg.norm(x, np.inf)
        gs_residuum_list.append(norm(solve_gs(d, N, params)[2][-1]))
        es_residuum_list.append(norm(solve_es(d, N, params)[2][-1]))

        matrix = block_matrix.BlockMatrix(d, N)
        pr, lower, upper, pc = matrix.get_lu()
        hat_u = linear_solvers.solve_lu(pr, lower, upper, pc, rhs.rhs(d, N, f))

        lu_residuum_list.append(norm(matrix.get_sparse() @ hat_u - rhs.rhs(d, N, f)))

    plt.plot(range(2, N_max+1), gs_residuum_list, label="GS")
    plt.plot(range(2, N_max+1), es_residuum_list, label="ES")
    plt.plot(range(2, N_max+1), lu_residuum_list, label="LU-Zerlegung")

    conditions_string = ""
    if params["eps"] > 0:
        conditions_string += "eps = {}, ".format(params["eps"])
    if params["max_iter"] > 0:
        conditions_string += "max_iter = {}, ".format(params["max_iter"])
    if params["min_red"] > 0:
        conditions_string += "min_red = {}, ".format(params["min_red"])

    plt.title("Normen der Residuen (" + conditions_string[0:-2] + ")")
    plt.xlabel("N (Diskretisierungsfeinheit)")
    plt.ylabel("Norm")

    plt.xscale("log")
    plt.yscale("log")

    plt.legend()
    plt.show()

def plot_comparison_max_abs_error(d, N_max, params=dict(eps=1e-8, max_iter=0, min_red=0)):
    """ Produziert Plot der maximalen absoluten Fehler der Lösungen des Poisson-Problems
    für Gesamtschritt-, Einzelschritt- und LU-Zerlegungs-Verfahren über die
    Diskretisierungsfeinheit. Für die Iterationsverfahren können Abbruchbedingungen
    eps, max_iter, min_red angegeben werden.

    Parameter
    ---------
    d : int
        Dimension des Poisson-Problems
    N_max : int
        maximale zu testende Diskretisierungsfeinheit, für alle Feinheiten zwischen 2 und
        N_max wird das Problem mit den verschiedenen Verfahren gelöst und der maximale
        absolute Fehler geplottet.
    params : dict, optional
        dictionary mit den Terminierungsbedingungen für die iterativen Verfahren

        eps : float
            Toleranz der unendlich-Norm der Residuen. Wenn kleiner oder gleich 0,
            gibt es keine solche Einschränkung.
        max_iter : int
            maximale Anzahl von Iterationen, die die Iterationsverfahren durchführen.
            Wenn kleiner oder gleich 0, gibt es keine solche Beschränkung.
        min_red : float
            minimale Abnahme des Residuums bezüglich der unendlich-Norm in jedem
            Iterationsschritt für die GS- und ES-Verfahren. Wenn kleiner oder gleich 0,
            gibt es keine solche Einschränkung.
    """

    gs_error_list, es_error_list, lu_error_list = [], [], []
    for N in range(2, N_max+1):
        error = lambda x: rhs.compute_error(d, N, x, u)
        gs_error_list.append(error(solve_gs(d, N, params)[1][-1]))
        es_error_list.append(error(solve_es(d, N, params)[1][-1]))

        matrix = block_matrix.BlockMatrix(d, N)
        pr, lower, upper, pc = matrix.get_lu()
        hat_u = linear_solvers.solve_lu(pr, lower, upper, pc, rhs.rhs(d, N, f))

        lu_error_list.append(error(hat_u))

    plt.plot(range(2, N_max+1), gs_error_list, linestyle="--", alpha=1, linewidth=3, label="GS")
    plt.plot(range(2, N_max+1), es_error_list, linestyle="-.", alpha=0.7, linewidth=3, label="ES")
    plt.plot(range(2, N_max+1), lu_error_list, label="LU-Zerlegung")

    conditions_string = ""
    if params["eps"] > 0:
        conditions_string += "eps = {}, ".format(params["eps"])
    if params["max_iter"] >= 1:
        conditions_string += "max_iter = {}, ".format(params["max_iter"])
    if params["min_red"] > 0:
        conditions_string += "min_red = {}, ".format(params["min_red"])

    plt.title("max. abs. Fehler (" + conditions_string[0:-2] + ")")
    plt.xlabel("N (Diskretisierungsfeinheit)")
    plt.ylabel("Fehler")

    plt.xscale("log")
    plt.yscale("log")

    plt.legend()
    plt.show()



def plot_convergence_max_abs_error_for_k(d, N_max):
    """ Produziert Plot der maximalen absoluten Fehler der Lösungen des Poisson-Problems
    mit Gesamtschritt- und Einzelschrittverfahren über die Diskretisierungsfeinheit.

    Parameter
    ---------
    d : int
        Dimension des Poisson-Problems
    N_max : int
        maximale zu testende Diskretisierungsfeinheit, für alle Feinheiten zwischen 2 und
        N_max wird das Problem mit den verschiedenen Verfahren gelöst und der maximale
        absolute Fehler geplottet.
    """

    for k in [-2, 0, 2, 4, 6]:
        gs_list, es_list = [], []
        for N in range(2, N_max+1):
            params = dict(eps=1/N**k, max_iter=0, min_red=0)
            error = lambda x: rhs.compute_error(d, N, x, u)

            iteration = solve_gs(d, N, params)[1]
            gs_list.append(error(iteration[-1]))

            iteration = solve_es(d, N, params)[1]
            es_list.append(error(iteration[-1]))

        plt.plot(range(2, N_max+1), gs_list, "-", linewidth=2, label="GS, eps = h^{}".format(k))
        plt.plot(range(2, N_max+1), es_list, "--", linewidth=2, label="ES, eps = h^{}".format(k))

    plt.plot(np.linspace(2, N_max+1), np.linspace(2, N_max+1)**(-2), "-.", color="0", label="h^2")
    plt.title("max. abs. Fehler für verschiedene eps (max_iter = 0, min_red = 0)")
    plt.xlabel("N (Diskretisierungsfeinheit)")
    plt.ylabel("Fehler")
    plt.xscale("log")
    plt.yscale("log")

    plt.legend()
    plt.show()



def main():
    """ main-Funktion von experimente.py, produziert die Plots aus dem Bericht. """

    d, N = 1, 50

    plot_max_abs_error(d, N, params=dict(eps=0, max_iter=0, min_red=1e-8))
    plot_max_abs_error(d, N, params=dict(eps=0, max_iter=0, min_red=1e-6))

    plot_norm_residuum(d, N, params=dict(eps=0, max_iter=0, min_red=1e-8))
    plot_norm_residuum(d, N, params=dict(eps=0, max_iter=0, min_red=1e-6))

    plot_comparison_residuals(d, N, params=dict(eps=1e-8, max_iter=0, min_red=0))
    plot_comparison_residuals(d, N, params=dict(eps=1e-4, max_iter=0, min_red=0))
    plot_comparison_residuals(d, N, params=dict(eps=1e-8, max_iter=1000, min_red=1e-10))

    plot_comparison_max_abs_error(d, N, params=dict(eps=1e-8, max_iter=0, min_red=0))
    plot_comparison_max_abs_error(d, N, params=dict(eps=1e-4, max_iter=0, min_red=0))
    plot_comparison_max_abs_error(d, N, params=dict(eps=1e-8, max_iter=1000, min_red=1e-10))

    plot_convergence_max_abs_error_for_k(d, N)

if __name__ == "__main__":
    main()
