'''

    auxfunc.py
    - Alessandro Trincone
    - Mario Gabriele Carofano

    Questo file contiene alcune funzionalità aggiuntive per la creazione
    della rete neurale e l'esecuzione del programma.

'''

def stampa_matrice(matrice):
    for riga in matrice:
        for elemento in riga:
            print(int(elemento), end=" ")
        print()