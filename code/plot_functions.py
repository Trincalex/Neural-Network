"""

    plot_functions.py
    - Alessandro Trincone
    - Mario Gabriele Carofano

    Questo file contiene alcune funzionalita' aggiuntive per il disegno dei grafici tramite la libreria 'matplotlib', tra cui il disegno dei grafici di errore e accuracy in fase di addestramento / validazione della rete neurale e il disegno delle cifre del dataset MNIST.

"""

# ########################################################################### #
# LIBRERIE

import constants
import dataset_functions as df

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
import matplotlib.pyplot as plot
from matplotlib.ticker import FormatStrFormatter
import os

# ########################################################################### #
# FUNZIONI PER LA FASE DI ADDESTRAMENTO

def plot_learning_curves(
        lc_plot : Axes,
        y_label : str,
        history_training : list[float],
        history_validation : list[float]
) -> None:
    
    """
        Disegna un grafico che confronta le misure calcolate in fase di training e in fase di validazione.

        Parameters:
        -   lc_plot : il subplot in cui disegnare la 'learning curve' richiesta.
        -   y_label : l'etichetta per l'asse delle ordinate.
        -   history_training : la lista di misure calcolate sui dati di addestramento.
        -   history_validation : la lista di misure calcolate sui dati di validazione.

        Returns:
        -   None.
    """

    x_min = 0
    x_max = len(history_training)

    lc_plot.set_xlim(x_min, x_max + x_max * 0.1)

    lc_plot.plot(range(x_min, x_max), history_training, 'b', label=f'Training {y_label}')
    lc_plot.plot(range(x_min, x_max), history_validation, 'r', label=f'Validation {y_label}')

# end

def plot_search_report(
        out_directory : str,
        title : str,
        k_fold_report : list[dict],
        search_report : pd.Series
) -> None:
    
    """
        Disegna una singola immagine contenente:
        -   un primo barchart per mostrare tutti i valori di errore medio ottenuti dall'addestramento dei modelli sulle diverse fold;
        -   un secondo barchart per mostrare tutti i valori di accuracy media ottenuti dall'addestramento dei modelli sulle diverse fold.
        -   infine, tutti i valori contenuti nel 'search_report'.

        Parameters:
        -   out_directory : la directory di output dove salvare il grafico.
        -   title : il titolo del grafico, che indica il tipo di tecnica di ricerca utilizzata. 
        -   k_fold_report : una lista di metriche di valutazione relative ai valori di errore e di accuracy di validazione ottenuti durante le fasi di addestramento dei modelli sulle diverse fold.
        -   search_report : un dizionario contenente i valori degli iper-parametri della combinazione valutata, la media e la deviazione standard dei punteggi di errore e accuracy ottenuti dall'esecuzione della k-fold cross validation.

        Returns:
        -   None.
    """

    # Identificativo del grafico.
    idx = str(search_report.name)

    # Dati da visualizzare.
    em          = search_report["Eta minus"]
    ep          = search_report["Eta plus"]
    hl          = search_report["Hidden layer"]
    errs_mean   = search_report["Mean error"]
    errs_std    = search_report["Std error"]
    accs_mean   = search_report["Mean accuracy"]
    accs_std    = search_report["Std accuracy"]

    train_errs      = [r['ET_history'] for r in k_fold_report]
    val_errs        = [r['EV_history'] for r in k_fold_report]
    train_accs      = [r['AT_history'] for r in k_fold_report]
    val_accs        = [r['AV_history'] for r in k_fold_report]
    folds_idx       = [f"Fold {r['Fold']}" for r in k_fold_report]
    folds_e_value   = [r['E_value'] for r in k_fold_report]
    folds_a_value   = [r['A_value'] for r in k_fold_report]

    # Creazione dell'immagine.
    fig = plot.figure(layout='constrained', figsize=constants.PLOT_SEARCH_FIGSIZE)
    subfigs = fig.subfigures(3, 1, wspace=0.07)

    # Titolo del grafico.
    plot_title = (
        fr"$\bf{{{title}}}$" + "\n" + 
        fr"$\it{{Eta\;minus\;=\;{em:.5f},\ Eta\;plus\;=\;{ep:.5f},\ Hidden\;layer\;=\;{hl:.0f}}}$"
    )
    fig.suptitle(plot_title, fontsize=20)

    # Creazione dell'immagine superiore contenente i barchart.
    axes_top = subfigs[0].subplots(1, 2)

    # Creazione del bar chart per l'errore di validazione.
    errs_bar_chart = axes_top[0]
    errs_bars = errs_bar_chart.bar(folds_idx, folds_e_value)

    errs_bar_chart.set_ylim(0, np.max(folds_e_value)+np.max(folds_e_value)*0.1)
    errs_bar_chart.set_ylabel(fr"$\bf{{Error}}$", fontsize=14)
    errs_bar_chart.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    errs_xlabel = (
        fr"$\bf{{Folds}}$" + "\n" + 
        fr"$\it{{Media\;=\;{errs_mean:.2f},\ Deviazione\;standard\;=\;{errs_std:.2f}}}$"
    )
    errs_bar_chart.set_xlabel(errs_xlabel, fontsize=14)
    errs_bar_chart.tick_params('x', labelrotation=20.0)

    for b in errs_bars:
        h = b.get_height()
        errs_bar_chart.text(b.get_x() + b.get_width() / 2.0, h, f'{h:.2f}', ha='center', va='bottom', fontsize=7)

    # Creazione del bar chart per l'accuracy di validazione.
    accs_bar_chart = axes_top[1]
    accs_bars = accs_bar_chart.bar(folds_idx, np.multiply(folds_a_value, 100))

    accs_bar_chart.set_ylim(0, 110)
    accs_bar_chart.set_ylabel(fr"$\bf{{Accuracy}}$", fontsize=14)
    accs_bar_chart.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    accs_xlabel = (
        fr"$\bf{{Folds}}$" + "\n" +
        fr"$\it{{Media\;=\;{accs_mean*100:.2f}\%,\ Deviazione\;standard\;=\;{accs_std*100:.2f}\%}}$"
    )
    accs_bar_chart.set_xlabel(accs_xlabel, fontsize=14)
    accs_bar_chart.tick_params('x', labelrotation=20.0)

    for b in accs_bars:
        h = b.get_height()
        accs_bar_chart.text(b.get_x() + b.get_width() / 2.0, h, f'{h:.2f} %', ha='center', va='bottom', fontsize=7)
    
    # Creazione dell'immagine inferiore contenente le learning curves dell'errore.
    axes_error = subfigs[1].subplots(1, len(k_fold_report), sharey=True)

    # plot_learning_curves(
    #     axes_down[0][0],
    #     "Error",
    #     train_errs[0],
    #     val_errs[0]
    # )

    for n, ax in enumerate(axes_error.ravel()):

        if n == 0:
            y_max = max(max(map(max, train_errs)), max(map(max, val_errs)))
            ax.set_ylim(0, y_max + y_max * 0.1)
            ax.set_ylabel(fr"$\bf{{Error}}$")

        ax.set_title(fr"$\bf{{{folds_idx[n % len(k_fold_report)]}}}$")
        plot_learning_curves(
            ax,
            "Error",
            train_errs[n],
            val_errs[n]
        )

    # end for n, ax

    # Creazione dell'immagine inferiore contenente le learning curves dell'accuracy.
    axes_accs = subfigs[2].subplots(1, len(k_fold_report), sharey=True)

    for n, ax in enumerate(axes_accs.ravel()):

        if n == 0:
            ax.set_ylim(0, 1.1)
            ax.set_ylabel(fr"$\bf{{Accuracy}}$")

        ax.set_xlabel(fr"$\bf{{Epochs}}$")
        plot_learning_curves(
            ax,
            "Accuracy",
            train_accs[n],
            val_accs[n]
        )

    # end for n, ax

    # Salvataggio del grafico.
    os.makedirs(out_directory, exist_ok=True)
    plot.savefig(out_directory + "report_" + idx + ".pdf", bbox_inches='tight')
    plot.close()

# end

# ########################################################################### #
# FUNZIONI PER LA FASE DI TESTING

def plot_testing(
        idTest : list[int],
        Xtest : np.ndarray,
        Ytest : np.ndarray,
        probabilities : np.ndarray,
        out_directory : str
) -> None:
    
    """
        Disegna un'immagine divisa in 2 colonne:
        -   la prima colonna contiene la rappresentazione in scala di grigi dell'immagine 28x28 delle cifre del MNIST test set e, come titolo, la relativa etichetta.
        -   la seconda colonna contiene la rappresentazione della predizione della rete neurale sull'esempio di testing corrispondente, tramite un bar chart.

        Parameters:
        -   idTest : l'array contenente gli identificativi degli esempi di testing.
        -   Xtest : la matrice contenente gli esempi di testing da elaborare. Ogni riga e' la rappresentazione dell'immagine del singolo esempio di training.
        -   Ytest : la matrice contenente le etichette corrispondenti per gli esempi di testing. Ogni riga rappresenta l'etichetta per il rispettivo esempio di testing.
        -   probabilities : la matrice contenente la distribuzione di probabilita' delle predizioni della rete neurale.
        -   out_directory : la directory di output dove salvare l'immagine.

        Returns:
        -   None.
    """

    os.makedirs(out_directory, exist_ok=True)

    for i in idTest:
        x = Xtest[i]
        y = Ytest[i]
        prob = probabilities[i]

        fig, axes = plot.subplots(1, 2)
        fig.suptitle(f"testing-report_{i}", fontsize=20)
        fig.set_size_inches(constants.PLOT_TESTING_FIGSIZE)
        fig.tight_layout()

        img         = x.reshape((constants.DIMENSIONE_IMMAGINE, constants.DIMENSIONE_IMMAGINE))
        test_label  = df.convert_to_label(y)
        pred_label  = df.convert_to_label(prob)

        image_plot = axes[constants.PLOT_TESTING_IMAGE_PLOT_INDEX]
        image_plot.imshow(img, cmap='gray')
        image_plot.set_title(f"Etichetta: {test_label}")
        image_plot.axis('off')

        bar_chart = axes[constants.PLOT_TESTING_BAR_CHART_INDEX]
        bars = bar_chart.bar(constants.ETICHETTE_CLASSI, prob*100)
        bar_chart.set_title(f"Predizione: {pred_label}")
        bar_chart.set_ylim(0, 110)
        bar_chart.tick_params('x', labelrotation=35.0)

        for b in bars:
            h = b.get_height()
            bar_chart.text(b.get_x() + b.get_width() / 2.0, h, f'{h:.2f} %', ha='center', va='bottom', fontsize=8)

        plot.savefig(out_directory + f"testing-report_{i}.pdf", bbox_inches='tight')
        plot.close()

    # end for i

# end

def plot_predictions(
        idTest : np.ndarray,
        Xtest : np.ndarray,
        Ytest : np.ndarray,
        probabilities : np.ndarray,
        out_directory : str,
        plot_mode : constants.PlotTestingMode = constants.PlotTestingMode.REPORT
) -> None:
    
    """
        Disegna un barchart che mostra a quale categoria appartengono le predizioni restituite in output dalla rete neurale.

        Parameters:
        -   idTest : l'array contenente gli identificativi degli esempi di testing.
        -   Xtest : la matrice contenente gli esempi di testing da elaborare. Ogni riga e' la rappresentazione dell'immagine del singolo esempio di training.
        -   Ytest : la matrice contenente le etichette corrispondenti per gli esempi di testing. Ogni riga rappresenta l'etichetta per il rispettivo esempio di testing.
        -   probabilities : la matrice contenente la distribuzione di probabilita' delle predizioni della rete neurale.
        -   out_directory : la directory di output dove salvare i grafici richiesti.
        -   plot_mode : serve a distinguere per quali esempi in input e quali predizioni in output si devono disegnare i grafici (vedi documentazione di PlotTestingMode).

        Returns:
        -   None.
    """

    high_confidence_corrects = []
    low_confidence_corrects = []
    almost_corrects = []
    wrongs = []

    for i, (x, y) in enumerate(zip(probabilities, Ytest)):
        prob_index = np.argmax(x)
        target_index = np.argmax(y)

        # print("prob:", prob_index, "target:", target_index)
        if prob_index == target_index:
            if x[prob_index] >= constants.PLOT_TESTING_CONFIDENCE_THRESHOLD:
                high_confidence_corrects.append(i)
            else:
                low_confidence_corrects.append(i)
        else:
            if x[target_index] >= (1-constants.PLOT_TESTING_CONFIDENCE_THRESHOLD):
                almost_corrects.append(i)
            else:
                wrongs.append(i)

    highs_acc       = len(high_confidence_corrects) / len(Xtest)
    lows_acc        = len(low_confidence_corrects) / len(Xtest)
    almosts_acc     = len(almost_corrects) / len(Xtest)
    wrongs_acc      = len(wrongs) / len(Xtest)

    highs_label     = f"{len(high_confidence_corrects)} ({highs_acc:.2%})"
    lows_label      = f"{len(low_confidence_corrects)} ({lows_acc:.2%})"
    almosts_label   = f"{len(almost_corrects)} ({almosts_acc:.2%})"
    wrongs_label    = f"{len(wrongs)} ({wrongs_acc:.2%})"

    # Se "plot_mode" e' constants.PlotTestingMode.NONE, stampa i risultati del testing.
    if plot_mode == constants.PlotTestingMode.NONE:
        print(f"Esempi di testing: {len(Xtest)}")
        print(f"Predizioni ad alta confidenza: {highs_label}")
        print(f"Predizioni a bassa confidenza: {lows_label}")
        print(f"Predizioni quasi corrette: {almosts_label}")
        print(f"Predizioni errate: {wrongs_label}")
        return
    
    # Altrimenti, salva i risultati del testing in un barchart.
    plot.title("testing-report", fontsize=20)
    plot.tight_layout()
    bar_chart = plot.bar(
        [
            "Alta confidenza",
            "Bassa confidenza",
            "Quasi corrette",
            "Errate"
        ],
        [
            len(high_confidence_corrects),
            len(low_confidence_corrects),
            len(almost_corrects),
            len(wrongs)
        ]
    )

    bar_colors = ['#40916C', '#95D5B2', 'orange', '#BC4749']
    bar_labels = [highs_label, lows_label, almosts_label, wrongs_label]

    for i, b in enumerate(bar_chart):
        b.set_color(bar_colors[i])
        h = b.get_height()
        plot.text(b.get_x() + b.get_width() / 2.0, h, f'{bar_labels[i]}', ha='center', va='bottom', fontsize=8)

    plot.xlabel("Predizioni", fontsize=15)
    plot.ylabel("Numero esempi", fontsize=15)
    plot.ylim(0, len(Xtest)+len(Xtest)*0.1)

    os.makedirs(out_directory, exist_ok=True)
    plot.savefig(out_directory + "testing-report.pdf", bbox_inches='tight')

    plot.close()

    # ##### #

    if plot_mode == constants.PlotTestingMode.ALL or plot_mode == constants.PlotTestingMode.HIGH_CONFIDENCE_CORRECT:
        highs_out = out_directory+"high_confidence_corrects/"
        plot_testing(high_confidence_corrects, Xtest, Ytest, probabilities, highs_out)
        print(f"Salvataggio dei 'testing-report' in '{highs_out}' completato.")
    
    if plot_mode == constants.PlotTestingMode.ALL or plot_mode == constants.PlotTestingMode.LOW_CONFIDENCE_CORRECT:
        lows_out = out_directory+"low_confidence_corrects/"
        plot_testing(low_confidence_corrects, Xtest, Ytest, probabilities, lows_out)
        print(f"Salvataggio dei 'testing-report' in '{lows_out}' completato.")

    if plot_mode == constants.PlotTestingMode.ALL or plot_mode == constants.PlotTestingMode.ALMOST_CORRECT:
        almosts_out = out_directory+"almost_corrects/"
        plot_testing(almost_corrects, Xtest, Ytest, probabilities, almosts_out)
        print(f"Salvataggio dei 'testing-report' in '{almosts_out}' completato.")
    
    if plot_mode == constants.PlotTestingMode.ALL or plot_mode == constants.PlotTestingMode.WRONG:
        wrongs_out = out_directory+"wrongs/"
        plot_testing(wrongs, Xtest, Ytest, probabilities, wrongs_out)
        print(f"Salvataggio dei 'testing-report' in '{wrongs_out}' completato.")

# end

# ########################################################################### #
# ALTRE FUNZIONI

def get_random_color() -> str:
    """
        Genera un numero intero di 24-bit corrispondenti alle 3 componenti di colore RGB, rispettivamente di 8 bit, e ne restituisce la rappresentazione in stringa del codice esadecimale.

        Returns:
        -   hex_color : il codice esadecimale di un colore RGB.
    """

    import random

    # Si ottiene un numero intero di 24-bit. 
    color = random.randrange(0, 2**24)

    # Si scrive il valore intero in una stringa esadecimale di 6 caratteri.
    hex_color = f"#{color:06x}"

    return hex_color

# end

# ########################################################################### #
# RIFERIMENTI

# https://github.com/MrDataScience/tutorials/blob/master/Data/MNIST/How%20To%20Plot%20MNIST%20Digits%20Using%20Matplotlib.ipynb
# https://stackoverflow.com/questions/18717877/prevent-plot-from-showing-in-jupyter-notebook
# https://www.geeksforgeeks.org/create-random-hex-color-code-using-python/
# https://stackoverflow.com/questions/12638408/decorating-hex-function-to-pad-zeros
# https://machinelearningmastery.com/how-to-configure-k-fold-cross-validation/
# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subfigures.html
# https://engineeringfordatascience.com/posts/matplotlib_subplots/
# https://www.researchgate.net/profile/Isaac-Nti-3/publication/356914937_Performance_of_Machine_Learning_Algorithms_with_Different_K_Values_in_K-fold_Cross-Validation/links/61b3101c19083169cb7f2c17/Performance-of-Machine-Learning-Algorithms-with-Different-K-Values-in-K-fold-Cross-Validation.pdf