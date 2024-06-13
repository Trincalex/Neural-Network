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
import os
import matplotlib.pyplot as plot
from datetime import datetime

# ########################################################################### #
# FUNZIONI PER LA FASE DI ADDESTRAMENTO

def plot_training_epochs(
        out_directory : str,
        title : str,
        y_label : str,
        history_training : list[float],
        history_validation : list[float],
        x_max : float = None,
        y_max : float = None
) -> None:
    
    """
        Disegna un grafico che confronta le misure calcolate in fase di training e in fase di validazione.

        Parameters:
        -   out_directory : la directory di output dove salvare il grafico richiesto.
        -   title : il titolo del grafico.
        -   y_label : l'etichetta per l'asse delle ordinate.
        -   history_training : la lista di misure calcolate sui dati di addestramento.
        -   history_validation : la lista di misure calcolate sui dati di validazione.

        Returns:
        -   None.
    """

    if history_validation is not None:
        # print(len(history_training), len(history_validation))
        if not len(history_training) == len(history_validation):
            raise IndexError("Le misure calcolate in fase training e validation non coincidono.")

    h_val = [] if history_validation is None else history_validation 

    x_min = 0
    y_min = 0

    min_value   = min(history_training + h_val)
    max_value   = max(history_training + h_val)
    mean_value  = sum(history_training + h_val) / len(history_training + h_val)

    if x_max is None:
        x_max = len(history_training)
    
    if y_max is None:
        y_max = max_value + max_value * 0.1

    plot.title(title)

    plot.xlim(x_min, x_max+1)
    plot.xlabel('Epochs')
    plot.xticks(range(x_min, x_max+1, x_max//10))

    plot.ylim(y_min, y_max)
    plot.ylabel(y_label)

    y_ticks = [0, min_value, max_value, mean_value]
    if y_label == constants.ReportTitle.Accuracy.name:
        y_ticks = y_ticks + [100]
    plot.yticks(y_ticks)

    plot.plot(range(x_min, x_max), history_training, 'b', label=f'Training {y_label}')

    if history_validation is not None:
        plot.plot(range(x_min, x_max), h_val, 'r', label=f'Validation {y_label}')

    plot.legend()

    os.makedirs(out_directory, exist_ok=True)
    plot.savefig(out_directory + title + '.pdf', bbox_inches='tight')
    print(f"Salvataggio di '{title}.pdf' in '{out_directory}' completato.")

    plot.close()

# end

def plot_error(
        out_directory : str,
        title : str,
        history_training_costs : list[float],
        history_validation_costs : list[float] = None
) -> None:
    
    """
        Disegna il grafico delle curve di errore, per mostrare se il modello e' in una condizione di overfitting sui dati di addestramento.

        Parameters:
        -   out_directory : la directory di output dove salvare il grafico delle curve di errore.
        -   title : il titolo del grafico.
        -   history_training_costs : la lista di misure di errore calcolate sui dati di addestramento.
        -   history_validation_costs : la lista di misure di errore calcolate sui dati di validazione.

        Returns:
        -   None.
    """

    plot_training_epochs(
        out_directory,
        f"{title}_{constants.ReportTitle.Error.value}",
        constants.ReportTitle.Error.name,
        history_training_costs,
        history_validation_costs
    )

# end

def plot_accuracy(
        out_directory : str,
        title : str,
        history_training_accuracy : list[float],
        history_validation_accuracy : list[float] = None
) -> None:
    
    """
        Disegna il grafico delle curve di accuratezza, per valutare la capacita' del modello di generalizzare sui dati di validazione.

        Parameters:
        -   out_directory : la directory di output dove salvare il grafico delle curve di accuracy.
        -   title : il titolo del grafico.
        -   history_training_accuracy : la lista di misure di accuracy calcolate sui dati di addestramento.
        -   history_validation_accuracy : la lista di misure di accuracy calcolate sui dati di validazione.

        Returns:
        -   None.
    """

    plot_training_epochs(
        out_directory,
        f"{title}_{constants.ReportTitle.Accuracy.value}",
        constants.ReportTitle.Accuracy.name,
        history_training_accuracy,
        history_validation_accuracy,
        y_max=100
    )

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
        ...

        Parameters:
        -   ... : ...
        -   out_directory : la directory di output dove salvare i grafici richiesti.

        Returns:
        -   None.
    """

    os.makedirs(out_directory, exist_ok=True)

    for i in idTest:
        x = Xtest[i]
        y = Ytest[i]
        prob = probabilities[i]

        fig, axes = plot.subplots(1, constants.PLOT_TESTING_COLUMNS)
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
        plot_mode : constants.PlotTestingMode = constants.PlotTestingMode.REPORT,
) -> None:
    
    """
        Disegna una tabella composta di 2 colonne e un numero di righe pari al numero di esempi di testing in input. E' strutturata come segue:
        -   la prima colonna contiene la rappresentazione in scala di grigi dell'immagine 28x28 delle cifre del dataset MNIST e la relativa etichetta prese dal testing set.
        -   la seconda colonna contiene la rappresentazione della predizione della rete neurale sull'esempio di testing corrispondente, tramite un bar chart.

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

    highs_acc       = len(high_confidence_corrects) / len(Xtest) * 100
    lows_acc        = len(low_confidence_corrects) / len(Xtest) * 100
    almosts_acc     = len(almost_corrects) / len(Xtest) * 100
    wrongs_acc      = len(wrongs) / len(Xtest) * 100

    highs_label     = f"{len(high_confidence_corrects)} ({highs_acc:.2f} %)"
    lows_label      = f"{len(low_confidence_corrects)} ({lows_acc:.2f} %)"
    almosts_label   = f"{len(almost_corrects)} ({almosts_acc:.2f} %)"
    wrongs_label    = f"{len(wrongs)} ({wrongs_acc:.2f} %)"

    # Se "plot_mode" e' constants.PlotTestingMode.NONE, stampa i risultati del testing.
    if plot_mode == constants.PlotTestingMode.NONE:
        print(f"Esempi di testing: {len(Xtest)}")
        print(f"Predizioni ad alta confidenza: {highs_label}")
        print(f"Predizioni a bassa confidenza: {lows_label}")
        print(f"Predizioni quasi corrette: {almosts_label}")
        print(f"Predizioni errate: {wrongs_label}")
        return
    
    # Altrimenti, salva i risultati del testing in un istogramma.
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
    
    """
        Infine, se e' stata selezionata un'altra modalita', stampa anche i singoli risultati delle predizioni.
        -   constants.PlotTestingMode.ALL : per visulizzare tutte le predizioni.
        -   constants.PlotTestingMode.HIGH_CONFIDENCE_CORRECT : per visulizzare solo le predizioni corrette ad alta confidenza.
        -   constants.PlotTestingMode.LOW_CONFIDENCE_CORRECT : per visulizzare tutte le altre predizioni corrette (a bassa confidenza).
        -   constants.PlotTestingMode.ALMOST_CORRECT : per visulizzare solo le predizioni errate che superano la soglia di confidenza sull'etichetta esatta.
        -   constants.PlotTestingMode.WRONG : per visulizzare tutte le altre predizioni errate.
    """

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
# RIFERIMENTI

# https://github.com/MrDataScience/tutorials/blob/master/Data/MNIST/How%20To%20Plot%20MNIST%20Digits%20Using%20Matplotlib.ipynb
# https://stackoverflow.com/questions/18717877/prevent-plot-from-showing-in-jupyter-notebook