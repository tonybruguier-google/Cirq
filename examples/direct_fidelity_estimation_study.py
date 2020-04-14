import dataclasses
import math
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

    plt.switch_backend('agg')
    plt.figure(figsize=(12, 5), dpi=150)
    for i, n_trials in enumerate(n_trials_range):
        plt.subplot(2, len(n_trials_range), i + 1)
        for n_clifford_trials in n_clifford_trials_range:
            fidelities = [np.mean(get_one_study(
                n_repetitions=n_repetitions, n_qubits=n_qubits,
                n_trials=n_trials, n_clifford_trials=n_clifford_trials,
                noise=noise)) for n_qubits in n_qubits_range]
            plt.plot(n_qubits_range, fidelities)
            plt.ylim((0.0, 1.0))
        plt.plot(n_qubits_range, [get_true_fidelity(n_qubits, noise) for n_qubits in n_qubits_range])
        if i == 0:
            plt.ylabel('Fidelity')
        if i == len(n_trials_range) - 1:
            plt.legend(['%s' % ('All' if x is None else ('%d' % (x))) for x in n_clifford_trials_range] + ["True"])
        plt.grid()
        plt.title('n_trials=%d' % (n_trials))

        plt.subplot(2, len(n_trials_range), i + 1 + len(n_trials_range))

        for n_clifford_trials in n_clifford_trials_range:
            errors = []
            for n_qubits in n_qubits_range:
                true_fidelity = get_true_fidelity(n_qubits, noise)
                raw_fidelity = get_one_study(
                    n_repetitions=n_repetitions, n_qubits=n_qubits,
                    n_trials=n_trials, n_clifford_trials=n_clifford_trials,
                    noise=noise)
                errors.append(math.sqrt(np.mean([(x - true_fidelity)**2 for x in raw_fidelity])))
            plt.plot(n_qubits_range, errors)
        if i == 0:
            plt.ylabel('L2 error')
        plt.xlabel('#qubits')
        plt.grid()
        plt.ylim((0.0, 0.25))

    plt.savefig('examples/dfe.png', format='png')


if __name__ == '__main__':
    main()
