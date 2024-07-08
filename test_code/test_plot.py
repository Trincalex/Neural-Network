import matplotlib.pyplot as plot
import os

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
    if y_label == "Accuracy":
        y_ticks = y_ticks + [0.1]
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
        f"{title}_error-report",
        "Error",
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
        f"{title}_accuracy-report",
        "Accuracy",
        history_training_accuracy,
        history_validation_accuracy,
        y_max=0.1
    )

# end