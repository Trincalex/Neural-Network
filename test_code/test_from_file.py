"""

    Se si vuole utilizzare una rete già addestrata con i seguenti parametri, iper-parametri e metriche di valutazione:

    NeuralNetwork(
        depth = 2,
        input_size = 784,
        network_layers = [
        Layer(
            size = 64,
            act_fun = <function leaky_relu at 0x10791c4c0>,
            inputs_size = (12500, 784)
            weights_shape = (64, 784),
            biases_shape = (64, 1)
        ),
        Layer(
            size = 10,
            act_fun = <function identity at 0x107dda680>,
            inputs_size = (12500, 64)
            weights_shape = (10, 64),
            biases_shape = (10, 1)
        )],
        err_fun = <function cross_entropy_softmax at 0x107dda8c0>,
        training_report = TrainingReport(
            num_epochs = 100,
            elapsed_time = 184.053 secondi,
            training_error = 1.83840,
            training_accuracy = 79.22 %,
            validation_error = 0.00000
            validation_accuracy = 0.00 %
        )
    )

"""

net = NeuralNetwork.load_network_from_file("../output/2024-06-13_17-31/params.pkl")
net = NeuralNetwork(
    784, [64, 10],
    l_act_funs=[auxfunc.leaky_relu, auxfunc.identity],
    e_fun=auxfunc.cross_entropy_softmax
)

history_report = net.train(Xtrain, Ytrain, examples=len(Xtrain), rprop=True)
pf.plot_error(out_directory, "rprop", [r.training_error for r in history_report])
pf.plot_accuracy(out_directory, "rprop", [r.training_accuracy for r in history_report])

print(repr(net))