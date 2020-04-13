import dataclasses
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import cast
from typing import List
from typing import Tuple

import cirq
import examples.direct_fidelity_estimation as dfe
import recirq

def _build_circuit(n_qubits) -> Tuple[cirq.Circuit, List[cirq.Qid]]:
    assert n_qubits >= 1

    # Builds a Clifford circuit to be studied.
    qubits: List[cirq.Qid] = cast(List[cirq.Qid],
                                  cirq.LineQubit.range(n_qubits))
    circuit: cirq.Circuit = cirq.Circuit()

    for i in range(0, n_qubits):
        circuit.append(cirq.H(qubits[i]))
    for i in range(1, n_qubits):
        circuit.append(cirq.CZ(qubits[i - 1], qubits[i]))
    return qubits, circuit

def _get_noise(noise):
    return cirq.depolarize(noise)

@recirq.json_serializable_dataclass(namespace='recirq.readout_scan',
                                    registry=recirq.Registry,
                                    frozen=True)
class DFETask:
    """A task that runs direct fidelity estimation (DFE) studies."""
    n_repetitions: int
    n_qubits: int
    n_trials: int
    n_clifford_trials: int
    noise: float

    @property
    def fn(self):
        clifford_string = 'exhaustive' if self.n_clifford_trials is None else '%d' % (self.n_clifford_trials)
        return '%d_%d_%d_%s_%e' % (self.n_repetitions, self.n_qubits, self.n_trials, clifford_string, self.noise)

    def run(self):
        print('Computing...')
        qubits, circuit = _build_circuit(self.n_qubits)
        noisy_simulator = cirq.DensityMatrixSimulator(noise=_get_noise(self.noise))
        results = []
        for _ in range(self.n_repetitions):
            estimated_fidelity, intermediate_result = (
                dfe.direct_fidelity_estimation(
                circuit, qubits, noisy_simulator, n_trials=self.n_trials,
                n_clifford_trials=self.n_clifford_trials, samples_per_term=0))
            results.append({
                'estimated_fidelity': estimated_fidelity,
                'intermediate_result': dataclasses.asdict(intermediate_result)})
        return results

def run_one_study(n_repetitions: int, n_qubits: int, n_trials: int, n_clifford_trials: int, noise: float):
    task = DFETask(n_repetitions=n_repetitions,
                   n_qubits=n_qubits,
                   n_trials=n_trials,
                   n_clifford_trials=n_clifford_trials,
                   noise=noise)
    base_dir = os.path.expanduser(f'~/cirq_results/study/dfe')
    if recirq.exists(task, base_dir=base_dir):
        data = recirq.load(task, base_dir=base_dir)
    else:
        data = {"results": task.run()}
        recirq.save(task=task,
                    data=data,
                    base_dir=base_dir)
    return data

def get_one_study(n_repetitions: int, n_qubits: int, n_trials: int, n_clifford_trials: int, noise: float):
    data = run_one_study(n_repetitions=n_repetitions, n_qubits=n_qubits, n_trials=n_trials, n_clifford_trials=n_clifford_trials, noise=noise)
    return [x['estimated_fidelity'] for x in data['results']]

@recirq.json_serializable_dataclass(namespace='recirq.readout_scan',
                                    registry=recirq.Registry,
                                    frozen=True)
class TrueFidelityTask:
    """A task that computes the fidelity exactly."""
    n_qubits: int
    noise: float

    @property
    def fn(self):
        return '%d_%e' % (self.n_qubits, self.noise)

        circuit, qubits = build_circuit(n_qubits=n_qubits)

    def run(self):
        print('Computing...')
        qubits, circuit = _build_circuit(self.n_qubits)
        clean_simulator = cirq.DensityMatrixSimulator()
        noisy_simulator = cirq.DensityMatrixSimulator(noise=_get_noise(self.noise))

        clean_density_matrix = cast(
            cirq.DensityMatrixTrialResult,
            clean_simulator.simulate(circuit)).final_density_matrix
        noisy_density_matrix = cast(
            cirq.DensityMatrixTrialResult,
            noisy_simulator.simulate(circuit)).final_density_matrix

        return cirq.fidelity(clean_density_matrix, noisy_density_matrix)

def run_true_fidelity(n_qubits: int, noise: float):
    task = TrueFidelityTask(n_qubits=n_qubits, noise=noise)
    base_dir = os.path.expanduser(f'~/cirq_results/study/true')
    if recirq.exists(task, base_dir=base_dir):
        data = recirq.load(task, base_dir=base_dir)
    else:
        data={'true_fidelity': task.run()}
        recirq.save(task=task,
                    data=data,
                    base_dir=base_dir)
    return data

def get_true_fidelity(n_qubits: int, noise: float):
    return run_true_fidelity(n_qubits=n_qubits, noise=noise)['true_fidelity']

def main():
    noise = 0.1
    n_repetitions = 100

    n_qubits_range = range(1, 9)
    n_clifford_trials_range = [1, 10, 100, None]
    n_trials_range = [1, 10, 100, 1000]

#     plt.switch_backend('agg')
#     plt.subplot(2, 1, 1)
#     plt.ylabel('Fidelity')
#     plt.gca().yaxis.grid(True)
#     plt.xticks(n_qubits_range)
#
#     legend_str = []
#     for n_clifford_trials in n_clifford_trials_range:
#         mu = []
#         for n_qubits in n_qubits_range:
#             fidelities = get_one_study(n_repetitions=n_repetitions,
#                                        n_qubits=n_qubits,
#                                        n_trials=1,
#                                        n_clifford_trials=n_clifford_trials,
#                                        noise=noise)
#             mu.append(np.mean(fidelities))
#         plt.plot(n_qubits_range, mu, '.-')
#         legend_str.append('Exhaustive' if n_clifford_trials is None else '%d' % (n_clifford_trials))
#     true_fidelities = [get_true_fidelity(n_qubits, noise) for n_qubits in n_qubits_range]
#     plt.plot(n_qubits_range, true_fidelities, '.-')
#     legend_str.append('True fidelity')
#     plt.legend(legend_str)
#
#     plt.subplot(2, 1, 2)
#     plt.xlabel('#qubits')
#     plt.ylabel('std. dev. of Fidelity')
#     plt.gca().yaxis.grid(True)
#     plt.xticks(n_qubits_range)
#     legend_str = []
#     for n_clifford_trials in n_clifford_trials_range:
#         sigma = []
#         for n_qubits in n_qubits_range:
#             true_fidelity = get_true_fidelity(n_qubits, noise)
#             fidelities = get_one_study(n_repetitions=n_repetitions,
#                                        n_qubits=n_qubits,
#                                        n_trials=1,
#                                        n_clifford_trials=n_clifford_trials,
#                                        noise=noise)
#             sigma.append(np.std([x - true_fidelity for x in fidelities]))
#         plt.plot(n_qubits_range, sigma, '.-')
#         legend_str.append('Exhaustive' if n_clifford_trials is None else '%d' % (n_clifford_trials))
#     plt.legend(legend_str)
#
#     plt.savefig('examples/direct_fidelity_estimation_study.png', format='png', dpi=150)

    plt.switch_backend('agg')
    fig = plt.figure()

    for n_qubits in n_qubits_range:
        ax = fig.add_subplot(4, 2, n_qubits)
        legend_str = []

        true_fidelity = get_true_fidelity(n_qubits, noise)

        for n_clifford_trials in n_clifford_trials_range:
            fidelities = [np.mean(get_one_study(
                    n_repetitions=n_repetitions, n_qubits=n_qubits,
                    n_trials=n_trials, n_clifford_trials=n_clifford_trials,
                    noise=noise)) for n_trials in n_trials_range]
            ax.semilogx(n_trials_range, fidelities, '.-')
            legend_str.append('$\infty$' if n_clifford_trials is None else '%d' % (n_clifford_trials))

        ax.semilogx(n_trials_range, [true_fidelity] * len(n_trials_range), '--', color='gray')

        plt.xlabel('n_trials')
        plt.legend(legend_str)
        plt.savefig('examples/direct_fidelity_estimation_study.png', format='png', dpi=150)

if __name__ == '__main__':
    main()
