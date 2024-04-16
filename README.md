# Neural-Network
Questa repository contiene codice e documentazione per l'elaborato finale del corso "Machine Learning (mod. Neural Networks and Deep Learning)".

### 🎓 Università
**Università degli studi di Napoli "Federico II"**

### 👤 Professore
**[*Roberto Prevete*](https://www.docenti.unina.it/roberto.prevete)**

### 🗓️ Anno Accademico
**2023-2024**

### 👨‍💻 Gruppo
| # | First name | Last name | Email |
| --- | --- | --- | --- |
| 1 | Alessandro | Trincone | [*al.trincone@studenti.unina.it*](mailto:al.trincone@studenti.unina.it) |
| 2 | Mario Gabriele | Carofano | [*m.carofano@studenti.unina.it*](mailto:m.carofano@studenti.unina.it) |

### 📄 Traccia
**Parte A**

Progettazione ed implementazione di una libreria di funzioni per:
- Simulare la propagazione in avanti di una rete neurale multi-strato full-connected.
Con tale libreria deve essere possibile implementare reti con più di uno strato di nodi interni e con qualsiasi funzione di attivazione per ciascun strato.
- La realizzazione della back-propagation per reti neurali multi-strato, per qualunque scelta della funzione di attivazione dei nodi della rete e la possibilità di usare almeno la somma dei quadrati o la cross-entropy con e senza soft-max come funzione di errore.

**Parte B**

Si consideri come input le immagini raw del dataset “mnist” di immagini di cifre scritte a mano (http://yann.lecun.com/exdb/mnist/).
Si ha, allora, un problema di classificazione a C classi, con C=10.
Si estragga opportunamente un dataset globale di N coppie (almeno N=10000).
Si fissi la resilient backpropagation (RProp) come algoritmo di aggiornamento dei pesi, ed una rete neurale con un unico strato di nodi interni. Si scelgano gli iper-parametri del modello, cioè i parametri della Rprop (eta-positivo ed eta-negativo) ed il numero di nodi interni, sulla base di un approccio di cross-validation k-fold (ad esempio k=10).
Per la ricerca degli iper-parametri confrontare l’approccio classico “a griglia” con quello “random”  (J Bergstra, Y Bengio, Randomsearch for hyper-parameter optimization, 2012).
Scegliere e mantenere invariati tutti gli altri “parametri” come, ad esempio, le funzioni di attivazione.
Se è necessario, per questioni di tempi computazionali e spazio in memoria, si possono ridurre le dimensioni delle immagini raw del dataset mnist (ad esempio utilizzando in matlab la funzione imresize).

### ⚠️ Requisiti
-
