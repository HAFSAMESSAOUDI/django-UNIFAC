from django.shortcuts import render
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objs as go
from django.http import HttpResponseBadRequest
import time

# Backend code
D_AB_exp = 1.33e-5
T = 313.13
Xa = 0.35
Xb = 0.65
lambda_a = 1.4311 ** (1 / 3)
lambda_b = 0.92 ** (1 / 3)
q_a = 1.432
q_b = 1.4
D_AB0 = 2.10e-5
D_BA0 = 2.67e-5

def calculate_D_AB(params, Xa):
    a_AB, a_BA = params
    Xb = 1 - Xa

    D = Xa * D_BA0 + Xb * np.log(D_AB0) + 2 * (
            Xa * np.log(Xa + (Xb * lambda_b) / lambda_a) +
            Xb * np.log(Xb + (Xa * lambda_a) / lambda_b)
    ) + 2 * Xa * Xb * (
                    (lambda_a / (Xa * lambda_a + Xb * lambda_b)) * (1 - (lambda_a / lambda_b)) +
                    (lambda_b / (Xa * lambda_a + Xb * lambda_b)) * (1 - (lambda_b / lambda_a))
            ) + Xb * q_a * (
                    (1 - ((Xb * q_b * np.exp(-a_BA / T)) / (Xa * q_a + Xb * q_b * np.exp(-a_BA / T))) ** 2) * (
                                -a_BA / T) +
                    (1 - ((Xb * q_b) / (Xb * q_b + Xa * q_a * np.exp(-a_AB / T))) ** 2) * np.exp(-a_AB / T) * (
                                -a_AB / T)
            ) + Xa * q_b * (
                    (1 - ((Xa * q_a * np.exp(-a_AB / T)) / (
                                Xa * q_a * np.exp(-a_AB / T) + Xb * q_b)) ** 2) * (-a_AB / T) +
                    (1 - ((Xa * q_a) / (Xa * q_a + Xb * q_b * np.exp(-a_BA / T)))) ** 2 * np.exp(-a_BA / T) * (
                                -a_BA / T)
            )

    D_AB = np.exp(D)
    return D_AB

def objective(params):
    D_AB_calculated = calculate_D_AB(params, Xa)
    return (D_AB_calculated - D_AB_exp) ** 2

def index(request):
    return render(request, 'index.html')

def results(request):
    if request.method == 'POST':
        # Récupérer les données du formulaire
        Xa = request.POST.get('Xa')
        D_AB_exp = request.POST.get('D_AB_exp')
        T = request.POST.get('T')
        lambda_a = request.POST.get('lambda_a')
        lambda_b = request.POST.get('lambda_b')
        q_a = request.POST.get('q_a')
        q_b = request.POST.get('q_b')
        D_AB0 = request.POST.get('D_AB0')
        D_BA0 = request.POST.get('D_BA0')

        # Vérifier si tous les champs sont remplis
        if None in [Xa, D_AB_exp, T, lambda_a, lambda_b, q_a, q_b, D_AB0, D_BA0]:
            return HttpResponseBadRequest("Tous les champs doivent être remplis.")

        # Convertir les valeurs en float
        try:
            Xa = float(Xa)
            D_AB_exp = float(D_AB_exp)
            T = float(T)
            lambda_a = float(lambda_a)
            lambda_b = float(lambda_b)
            q_a = float(q_a)
            q_b = float(q_b)
            D_AB0 = float(D_AB0)
            D_BA0 = float(D_BA0)
        except ValueError:
            return HttpResponseBadRequest("Veuillez saisir des valeurs numériques valides.")

        # Continuer avec le reste de votre logique d'optimisation
        a_AB_init = 900
        a_BA_init = 900
        params_initial = [a_AB_init, a_BA_init]
        tolerance = 1e-5
        max_iterations = 1000

        # Début du chronomètre
        start_time = time.time()

        for iteration in range(max_iterations):
            result = minimize(objective, params_initial, method='Powell')
            a_AB_opt, a_BA_opt = result.x
            D_AB_opt = calculate_D_AB([a_AB_opt, a_BA_opt], Xa)
            error = abs(D_AB_opt - D_AB_exp)

            params_initial = [a_AB_opt, a_BA_opt]

            if error <= tolerance:
                break

        # Fin du chronomètre
        end_time = time.time()

        # Calcul de la durée de l'itération
        iteration_duration = end_time - start_time
        print("Duration of iteration:", iteration_duration, "")
        

        # Generating plot
        Xa_values = np.linspace(0.1, 0.7, 100)
        D_AB_values = [calculate_D_AB([a_AB_opt, a_BA_opt], Xa) for Xa in Xa_values]

        # Creating plotly trace
        trace = go.Scatter(x=Xa_values, y=D_AB_values, mode='lines', name='D_AB vs Xa')

        # Creating layout
        layout = go.Layout(title='Variation de D_AB avec la fraction molaire Xa',
                           xaxis=dict(title='Fraction molaire Xa'),
                           yaxis=dict(title='Coefficient de diffusion D_AB (cm^2/s)'),
                           showlegend=True)

        # Creating figure
        fig = go.Figure(data=[trace], layout=layout)

        # Converting plotly figure to JSON
        plot_json = fig.to_json()

        return render(request, 'results.html', {
            'a_AB_opt': a_AB_opt,
            'a_BA_opt': a_BA_opt,
            'D_AB_opt': D_AB_opt,
            'error': abs(D_AB_opt - D_AB_exp),
            'plot_json': plot_json,
            'iteration_duration': iteration_duration
        })
    else:
        return HttpResponseBadRequest("Cette page ne peut être accédée que via une soumission de formulaire.")
