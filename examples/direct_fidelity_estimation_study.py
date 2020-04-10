"""Studies the impact of exhaustive vs Monte Carlo DFE.

Studies when to set --n_clifford_trials to 0 (exhaustive) and when to set it
to sampling value.
"""

from typing import cast
from typing import List
from typing import Tuple
import cirq
import examples.direct_fidelity_estimation as dfe
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
import statistics

def build_circuit(n_qubits) -> Tuple[cirq.Circuit, List[cirq.Qid]]:
    assert n_qubits >= 1

    # Builds a Clifford circuit to be studied.
    qubits: List[cirq.Qid] = cast(List[cirq.Qid],
                                  cirq.LineQubit.range(n_qubits))
    circuit: cirq.Circuit = cirq.Circuit()

    # circuit.append(cirq.Z(qubits[0]))
    # for i in range(1, n_qubits):
    #     circuit.append(cirq.CNOT(qubits[0], qubits[i]))

    for i in range(0, n_qubits):
        circuit.append(cirq.H(qubits[i]))
    for i in range(1, n_qubits):
        circuit.append(cirq.CZ(qubits[i - 1], qubits[i]))

    return circuit, qubits


def main():
    plt.switch_backend('agg')
    noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(0.1))
    n_qubits_range = range(1, 9)

    clean_simulator = cirq.DensityMatrixSimulator()
    noisy_simulator = cirq.DensityMatrixSimulator(noise=noise)

    true_fidelities = []
    for n_qubits in n_qubits_range:
        circuit, qubits = build_circuit(n_qubits=n_qubits)


        clean_density_matrix = cast(
            cirq.DensityMatrixTrialResult,
            clean_simulator.simulate(circuit)).final_density_matrix
        noisy_density_matrix = cast(
            cirq.DensityMatrixTrialResult,
            noisy_simulator.simulate(circuit)).final_density_matrix

        true_fidelity = cirq.fidelity(clean_density_matrix, noisy_density_matrix)
        true_fidelities.append(true_fidelity)


    n_clifford_trials_range = [1, 10, 100, None]

    delta_fidelities_means = np.zeros([len(n_clifford_trials_range), len(n_qubits_range)])
    delta_fidelities_stddevs = np.zeros([len(n_clifford_trials_range), len(n_qubits_range)])

    legends = []
    for i in range(len(n_clifford_trials_range)):
        n_clifford_trials = n_clifford_trials_range[i]
        if n_clifford_trials is not None:  # DO NOT SUBMIT
            print('TONYBOOM n_clifford_trials=%d' % (n_clifford_trials))  # DO NOT SUBMIT
        else:  # DO NOT SUBMIT
            print('TONYBOOM n_clifford_trials=Exhaustive')  # DO NOT SUBMIT
        delta_fidelities_stddev = []
        for j in range(len(n_qubits_range)):
            n_qubits = n_qubits_range[j]
            true_fidelity = true_fidelities[j]
            print('TONYBOOM n_qubits=%d' % (n_qubits))  # DO NOT SUBMIT
            estimated_fidelities = []
            estimated_fidelity_errors = []
            for _ in range(1000):
                circuit, qubits = build_circuit(n_qubits=n_qubits)
                estimated_fidelity = dfe.direct_fidelity_estimation(
                        circuit,
                        qubits,
                        noisy_simulator,
                        n_trials=1000,
                        n_clifford_trials=n_clifford_trials,
                        samples_per_term=0)
                estimated_fidelities.append(estimated_fidelity)
                estimated_fidelity_errors.append(estimated_fidelity - true_fidelity)
            delta_fidelities_means[i][j] = np.mean(estimated_fidelities)
            delta_fidelities_stddevs[i][j] = np.std(estimated_fidelity_errors)
        if n_clifford_trials is None:
            legends.append('Exhaustive')
        else:
            legends.append('Clifford_%d' % n_clifford_trials)

    plt.subplot(2, 1, 1)
    #plt.xlabel('#qubits')
    plt.ylabel('Fidelity')
    plt.gca().yaxis.grid(True)
    plt.xticks(n_qubits_range)

    for i in range(len(n_clifford_trials_range)):
        plt.plot(n_qubits_range, delta_fidelities_means[i, :], '.--')
    plt.plot(n_qubits_range, true_fidelities, '.-')
    plt.legend(legends + ["True"])

    plt.subplot(2, 1, 2)
    plt.xlabel('#qubits')
    plt.ylabel('std. dev. of Fidelity')
    plt.gca().yaxis.grid(True)
    plt.xticks(n_qubits_range)

    for i in range(len(n_clifford_trials_range)):
        plt.plot(n_qubits_range, delta_fidelities_stddevs[i, :], '.--')
    plt.legend(legends)


    plt.legend(legends)

    plt.savefig('examples/direct_fidelity_estimation_study.png', format='png', dpi=150)
#     n_qubits = 8
#     n_clifford_trials = 1
#     noise = cirq.ConstantQubitNoiseModel(cirq.amplitude_damp(0.01))
#
#     circuit, qubits = build_circuit(n_qubits=n_qubits)
#     clean_simulator = cirq.DensityMatrixSimulator()
#     clean_density_matrix = cast(
#         cirq.DensityMatrixTrialResult,
#         clean_simulator.simulate(circuit)).final_density_matrix
#
#     noisy_simulator = cirq.DensityMatrixSimulator(noise=noise)
#     noisy_density_matrix = cast(
#         cirq.DensityMatrixTrialResult,
#         noisy_simulator.simulate(circuit)).final_density_matrix
#
#     true_fidelity = cirq.fidelity(clean_density_matrix, noisy_density_matrix)
#
#     measurements = []
#     for iter in range(2000):
#       print('iter=%d' % (iter))
#       circuit, qubits = build_circuit(n_qubits=n_qubits)
#       estimated_fidelity = dfe.direct_fidelity_estimation(
#               circuit,
#               qubits,
#               noisy_simulator,
#               n_trials=1000,
#               n_clifford_trials=n_clifford_trials,
#               samples_per_term=0)
#       measurements.append(estimated_fidelity)
#     plt.hist(measurements, bins=[x * (1.0 / 50.0) for x in range(51)])
#
#     mu_hat = statistics.mean(measurements)
#     sigma_hat = statistics.stdev(measurements)
#
#     plt.xlabel('Estimated fidelity')
#     plt.ylabel('Count')
#     plt.title('true_fidelity=%f, estimated_fidelity=%f +/- %f' % (true_fidelity, mu_hat, sigma_hat))
#
#     plt.savefig('examples/direct_fidelity_estimation_dispersion.png', format='png')

if __name__ == '__main__':
    main()
