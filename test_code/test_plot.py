import numpy as np
import matplotlib.pyplot as plot
import os
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import PercentFormatter

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

def plot_k_fold_error_scores(
        out_directory : str,
        fold_reports : list[dict]
) -> None:
    
    """
        Disegna un line plot per visualizzare i risultati della k-fold cross validation rispetto agli errori di validazione di ogni singola fold.

        Parameters:
        -   out_directory : la directory di output dove salvare i grafici richiesti.
        -   fold_reports : e' una lista di dizionari contenenti tutte le metriche di valutazione della fase di addestramento della rete neurale su tutte le fold della cross validation.

        Returns:
        -   err_mean : la media degli errori di validazione su tutte le fold.
        -   err_std : la deviazione standard degli errori di validazione su tutte le fold.
    """

    errs = [r['Report'].validation_error for r in fold_reports]
    err_mean = np.mean(errs)
    err_std = np.std(errs)

    # Titolo del grafico.
    plot.suptitle("Error scores", fontsize=20)
    plot.title(f"Media: {err_mean:.2f}, Deviazione standard: {err_std:.2f}", fontsize=10)
    plot.tight_layout()

    # Etichette sull'asse x.
    x_ticks = [f"Fold {r['Fold']}" for r in fold_reports]
    plot.xlabel("Folds", fontsize=15)

    # Si calcola il massimo errore di validazione di tutte le fold.
    y_max = max([r['E_max'] for r in fold_reports])
    # y_min = max([r['E_min'] for r in fold_reports])
    plot.ylim(0, y_max)
    plot.margins()

    # Etichette sull'asse y.
    plot.ylabel("Error", fontsize=15)
    plot.locator_params(axis='y', nbins=5)
    plot.yticks([y for y in plot.yticks()[0]] + [err_mean])
    plot.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Disegna un line plot per confrontare gli errori di validazione.
    # Ad ogni fold e' anche associato una barra di errore che mostra il massimo ed il minimo errore di validazione ottenuto in fase di addestramento.
    plot.errorbar(
        x_ticks,
        [r['E_mean'] for r in fold_reports],
        yerr=[abs(r['E_max']-r['E_min']) for r in fold_reports],
        fmt='o'
    )

    # Disegna una linea retta rossa che indica la media degli errori di validazione su tutte le fold.
    plot.plot(
        x_ticks,
        [err_mean for _ in fold_reports],
        color='red', linestyle='dashed', linewidth=1
    )

    # Salvataggio del grafico.
    os.makedirs(out_directory, exist_ok=True)
    plot.savefig(out_directory + "error_scores.pdf", bbox_inches='tight')
    plot.close()

    return err_mean, err_std

# end

def plot_k_fold_accuracy_scores(
        out_directory : str,
        fold_reports : list[dict]
) -> None:
    
    """
        Disegna un line plot per visualizzare i risultati della k-fold cross validation rispetto alle accuracy di validazione di ogni singola fold.

        Parameters:
        -   out_directory : la directory di output dove salvare i grafici richiesti.
        -   fold_reports : e' una lista di dizionari contenenti tutte le metriche di valutazione della fase di addestramento della rete neurale su tutte le fold della cross validation.

        Returns:
        -   acc_mean : la media delle accuracy di validazione su tutte le fold.
        -   acc_std : la deviazione standard delle accuracy di validazione su tutte le fold.
    """

    accs = [r['Report'].validation_accuracy for r in fold_reports]
    acc_mean = np.mean(accs)
    acc_std = np.std(accs)

    # Titolo del grafico.
    plot.suptitle("Accuracy scores", fontsize=20)
    plot.title(f"Media: {acc_mean:.2%}, Deviazione standard: {acc_std:.2%}", fontsize=10)
    plot.tight_layout()

    # Etichette sull'asse x.
    x_ticks = [f"Fold {r['Fold']}" for r in fold_reports]
    plot.xlabel("Folds", fontsize=15)

    # Si calcola la massima accuracy di validazione di tutte le fold.
    y_max = max([r['A_max'] for r in fold_reports])
    y_min = min([r['A_min'] for r in fold_reports])
    plot.ylim(y_min, y_max)
    plot.margins()

    # Etichette sull'asse y.
    plot.ylabel("Accuracy", fontsize=15)
    plot.locator_params(axis='y', nbins=5)
    plot.yticks([y for y in plot.yticks()[0]] + [acc_mean])
    plot.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=2))

    # Disegna un line plot per confrontare le accuracy di validazione.
    # Ad ogni fold e' anche associato una barra di errore che mostra la massima e la minima accuracy di validazione ottenuta in fase di addestramento.
    plot.errorbar(
        x_ticks,
        [r['A_mean'] for r in fold_reports],
        yerr=[abs(r['A_max']-r['A_min']) for r in fold_reports],
        fmt='o'
    )

    # Disegna una linea retta rossa che indica la media delle accuracy di validazione.
    plot.plot(
        x_ticks,
        [acc_mean for _ in fold_reports],
        color='red', linestyle='dashed', linewidth=1
    )

    # Salvataggio del grafico.
    os.makedirs(out_directory, exist_ok=True)
    plot.savefig(out_directory + "accuracy_scores.pdf", bbox_inches='tight')
    plot.close()

    return acc_mean, acc_std

# end

# ########################################################################### #

# # Disegno di un line plot con le percentuali di accuracy, media e deviazione standard su tutte le fold.
# acc_mean, acc_std = pf.plot_k_fold_accuracy_scores(out_directory, k_fold_report)

# # Disegno di un line plot con i valori di errore, media e deviazione standard su tutte le fold.
# err_mean, err_std = pf.plot_k_fold_error_scores(out_directory, k_fold_report)

# # return err_mean, err_std, acc_mean, acc_std
# # -   err_mean : la media dei valori di errore di validazione su tutti i modelli addestrati.
# # -   err_std : la deviazione standard dei valori di errore di validazione su tutti i modelli addestrati.
# # -   acc_mean : la media delle percentuali di accuracy di validazione su tutti i modelli addestrati.
# # -   acc_std : la deviazione standard delle percentuali di accuracy di validazione su tutti i modelli addestrati.