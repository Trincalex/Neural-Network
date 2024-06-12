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
        title : constants.ReportTitle,
        history_training : list[float],
        history_validation : list[float],
        x_max : float = None,
        y_max : float = None,
) -> None:
    
    """
        Disegna un grafico che confronta le misure calcolate in fase di training e in fase di validazione.

        Parameters:
        -   title : il titolo del grafico.
        -   history_training : la lista di misure calcolate sui dati di addestramento.
        -   history_validation : la lista di misure calcolate sui dati di validazione.

        Returns:
        -   None.
    """

    if history_validation is not None:
        # print(len(history_training), len(history_validation))
        if not len(history_training) == len(history_validation):
            raise IndexError("Le misure calcolate in fase training e validation non coincidono.")
    
    if not isinstance(title, constants.ReportTitle):
        raise TypeError("Il titolo deve essere uno dei valori dell'enumerazione ReportTitle.")

    h_val = [] if history_validation is None else history_validation 

    max_value = max(history_training + h_val)

    x_min = 0
    y_min = 0

    if x_max is None:
        x_max = len(history_training)
    
    if y_max is None:
        y_max = max_value + max_value * 0.1

    plot.figure(num=int(title))
    plot.title(str(title))

    plot.xlim(x_min, x_max)
    plot.xlabel('Epochs')
    plot.xticks(range(x_min, x_max))

    plot.ylim(y_min, y_max)
    plot.ylabel(title.name)

    plot.plot(range(x_min, x_max), history_training, 'b', label='Training ' + title.name)

    if history_validation is not None:
        plot.plot(range(x_min, x_max), h_val, 'r', label='Validation ' + title.name)

    plot.legend()

    out_directory = constants.OUTPUT_DIRECTORY + datetime.now().strftime("%Y-%m-%d_%H-%M") + "/"
    os.makedirs(out_directory, exist_ok=True)
    plot.savefig(out_directory + str(title) + '.pdf', bbox_inches='tight')
    print(f"Salvataggio di '{str(title)}.pdf' in '{out_directory}' completato.")

    plot.close()

# end

def plot_error(
        history_training_costs : list[float],
        history_validation_costs : list[float] = None
) -> None:
    
    """
        Disegna il grafico delle curve di errore, per mostrare se il modello e' in una condizione di overfitting sui dati di addestramento.

        Parameters:
        -   history_training_costs : la lista di misure di errore calcolate sui dati di addestramento.
        -   history_validation_costs : la lista di misure di errore calcolate sui dati di validazione.

        Returns:
        -   None.
    """

    plot_training_epochs(
        constants.ReportTitle.Error,
        history_training_costs,
        history_validation_costs
    )

# end

def plot_accuracy(
        history_training_accuracy : list[float],
        history_validation_accuracy : list[float] = None
) -> None:
    
    """
        Disegna il grafico delle curve di accuratezza, per valutare la capacita' del modello di generalizzare sui dati di validazione.

        Parameters:
        -   history_training_accuracy : la lista di misure di accuracy calcolate sui dati di addestramento.
        -   history_validation_accuracy : la lista di misure di accuracy calcolate sui dati di validazione.

        Returns:
        -   None.
    """

    plot_training_epochs(
        constants.ReportTitle.Accuracy,
        history_training_accuracy,
        history_validation_accuracy,
        y_max=100
    )

# end

# ########################################################################### #
# FUNZIONI PER LA FASE DI TESTING

def plot_testing(
        id : int,
        x : np.ndarray,
        y : np.ndarray,
        prob : np.ndarray,
        out_directory : str
) -> None:
    
    """
        ...

        Parameters:
        -   ... : ...

        Returns:
        -   None.
    """

    os.makedirs(out_directory, exist_ok=True)

    fig, axes = plot.subplots(1, constants.PLOT_TESTING_COLUMNS)
    fig.suptitle(f"testing-report_{id}", fontsize=20)
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
    bar_chart.bar(constants.ETICHETTE_CLASSI, prob*100)
    bar_chart.set_title(f"Predizione: {pred_label}")
    bar_chart.set_ylim(0, 105)
    bar_chart.tick_params('x', labelrotation=35.0)

    plot.savefig(out_directory + f"testing-report_{id}.pdf", bbox_inches='tight')
    plot.close()

# end

def plot_predictions(
        idTest : np.ndarray,
        Xtest : np.ndarray,
        Ytest : np.ndarray,
        probabilities : np.ndarray,
        plot_mode : constants.PlotTestingMode = constants.PlotTestingMode.NONE
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
        -   plot_mode : serve a distinguere per quali esempi in input e quali predizioni in output si devono disegnare i grafici (vedi documentazione di PlotTestingMode).

        Returns:
        -   None.
    """

    out_directory = constants.OUTPUT_DIRECTORY + datetime.now().strftime("%Y-%m-%d_%H-%M") + "/"

    high_confidence_corrects = []
    low_confidence_corrects = []
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
            wrongs.append(i)
    
    acc = (len(Xtest)-len(wrongs)) / len(Xtest) * 100

    print(f"Esempi di testing: {len(Xtest)}")
    print(f"Predizioni ad alta confidenza: {len(high_confidence_corrects)}")
    print(f"Predizioni a bassa confidenza: {len(low_confidence_corrects)}")
    print(f"Predizioni errate: {len(wrongs)}")
    print(f"Accuracy: {acc:.2f} %")

    if plot_mode == constants.PlotTestingMode.NONE:
        return
    
    if plot_mode == constants.PlotTestingMode.ALL or plot_mode == constants.PlotTestingMode.HIGH_CONFIDENCE_CORRECT:
        out_directory = out_directory+"high_confidence_corrects/" if plot_mode == constants.PlotTestingMode.ALL else out_directory
        for i in high_confidence_corrects:
            plot_testing(idTest[i], Xtest[i], Ytest[i], probabilities[i], out_directory)
    
    if plot_mode == constants.PlotTestingMode.ALL or plot_mode == constants.PlotTestingMode.LOW_CONFIDENCE_CORRECT:
        out_directory = out_directory+"low_confidence_corrects/" if plot_mode == constants.PlotTestingMode.ALL else out_directory
        for i in low_confidence_corrects:
            plot_testing(idTest[i], Xtest[i], Ytest[i], probabilities[i], out_directory)
    
    if plot_mode == constants.PlotTestingMode.ALL or plot_mode == constants.PlotTestingMode.WRONG:
        out_directory = out_directory+"wrongs/" if plot_mode == constants.PlotTestingMode.ALL else out_directory
        for i in wrongs:
            plot_testing(idTest[i], Xtest[i], Ytest[i], probabilities[i], out_directory)
    
    print(f"Salvataggio dei 'testing-report' in '{out_directory}' completato.")

# end

# ########################################################################### #
# RIFERIMENTI

# https://github.com/MrDataScience/tutorials/blob/master/Data/MNIST/How%20To%20Plot%20MNIST%20Digits%20Using%20Matplotlib.ipynb
# https://stackoverflow.com/questions/18717877/prevent-plot-from-showing-in-jupyter-notebook