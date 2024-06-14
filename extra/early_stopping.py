# v_costs = np.array([r.validation_error for r in history_report[-es_patience:]])
# diffs   = np.diff(v_costs)

# # Si moltiplica per -1 perche' si vuole ottenere un valore positivo quando si hanno miglioramenti nell'errore di validazione (cioe', sta diminuendo).
# value   = np.sum(diffs) * (-1)

# if constants.DEBUG_MODE:
#     with np.printoptions(threshold=np.inf):
#         print("--- NETWORK TRAINING (v_costs) ---\n")
#         pprint.pprint(v_costs)
#         print("\n-----\n")
#         print("--- NETWORK TRAINING (diffs) ---\n")
#         pprint.pprint(diffs)
#         print("\n-----\n")
#         print("--- NETWORK TRAINING (value) ---\n")
#         print(value)
#         print("\n-----\n\n")

# del v_costs, diffs
# gc.collect()