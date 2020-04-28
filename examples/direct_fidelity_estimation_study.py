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

def _build_circuit(circuit_id, n_qubits) -> Tuple[cirq.Circuit, List[cirq.Qid]]:
    assert n_qubits >= 1

    if circuit_id == 0:
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
    circuit_id: int
    n_repetitions: int
    n_qubits: int
    n_measured_operators: int
    samples_per_term: int
    noise: float

    @property
    def fn(self):
        n_measured_operators_strings = 'exhaustive' if self.n_measured_operators is None else '%d' % (self.n_measured_operators)
        return '%d_%d_%d_%s_%d_%e' % (self.circuit_id, self.n_repetitions, self.n_qubits, n_measured_operators_strings, self.samples_per_term, self.noise)

    def run(self):
        print('Computing...')
        qubits, circuit = _build_circuit(self.circuit_id, self.n_qubits)
        noisy_simulator = cirq.DensityMatrixSimulator(noise=_get_noise(self.noise))
        results = []
        for _ in range(self.n_repetitions):
            estimated_fidelity, intermediate_result = (
              dfe.direct_fidelity_estimation(
                  circuit, qubits, noisy_simulator,
                  n_measured_operators=self.n_measured_operators,
                  samples_per_term=self.samples_per_term))
            results.append({
                'estimated_fidelity': estimated_fidelity,
                'intermediate_result': dataclasses.asdict(intermediate_result)})
        return results

def run_one_study(circuit_id: int, n_repetitions: int, n_qubits: int, n_measured_operators: int, samples_per_term: int, noise: float):
    task = DFETask(circuit_id=circuit_id,
                   n_repetitions=n_repetitions,
                   n_qubits=n_qubits,
                   n_measured_operators=n_measured_operators,
                   samples_per_term=samples_per_term,
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

def get_one_study(circuit_id: int, n_repetitions: int, n_qubits: int, n_measured_operators: int, samples_per_term: int, noise: float):
    data = run_one_study(circuit_id=circuit_id, n_repetitions=n_repetitions, n_qubits=n_qubits, n_measured_operators=n_measured_operators, samples_per_term=samples_per_term, noise=noise)
    return [x['estimated_fidelity'] for x in data['results']]

@recirq.json_serializable_dataclass(namespace='recirq.readout_scan',
                                    registry=recirq.Registry,
                                    frozen=True)
class TrueFidelityTask:
    """A task that computes the fidelity exactly."""
    circuit_id: int
    n_qubits: int
    noise: float

    @property
    def fn(self):
        return '%d_%d_%e' % (self.circuit_id, self.n_qubits, self.noise)

        circuit, qubits = build_circuit(circuit_id=circuit_id, n_qubits=n_qubits)

    def run(self):
        print('Computing...')
        qubits, circuit = _build_circuit(self.circuit_id, self.n_qubits)
        clean_simulator = cirq.DensityMatrixSimulator()
        noisy_simulator = cirq.DensityMatrixSimulator(noise=_get_noise(self.noise))

        clean_density_matrix = cast(
            cirq.DensityMatrixTrialResult,
            clean_simulator.simulate(circuit)).final_density_matrix
        noisy_density_matrix = cast(
            cirq.DensityMatrixTrialResult,
            noisy_simulator.simulate(circuit)).final_density_matrix

        return cirq.fidelity(clean_density_matrix, noisy_density_matrix)

def run_true_fidelity(circuit_id: int, n_qubits: int, noise: float):
    task = TrueFidelityTask(circuit_id=circuit_id, n_qubits=n_qubits, noise=noise)
    base_dir = os.path.expanduser(f'~/cirq_results/study/true')
    if recirq.exists(task, base_dir=base_dir):
        data = recirq.load(task, base_dir=base_dir)
    else:
        data={'true_fidelity': task.run()}
        recirq.save(task=task,
                    data=data,
                    base_dir=base_dir)
    return data

def get_true_fidelity(circuit_id: int, n_qubits: int, noise: float):
    return run_true_fidelity(circuit_id=circuit_id, n_qubits=n_qubits, noise=noise)['true_fidelity']

def main():
    noise = 0.1
    n_repetitions = 100
    circuit_id = 0
    samples_per_term = 1000

    n_qubits_range = range(1, 9)
    n_measured_operators_range = [1, 10, 100, None]

    plt.switch_backend('agg')
    plt.figure(figsize=(8.5, 11), dpi=150)

    plt.subplot(3, 1, 1)
    legend_str = []
    for n_measured_operators in n_measured_operators_range:
        fidelities = [np.mean(get_one_study(
            circuit_id=circuit_id, n_repetitions=n_repetitions, n_qubits=n_qubits,
            n_measured_operators=n_measured_operators, samples_per_term=samples_per_term,
            noise=noise)) for n_qubits in n_qubits_range]
        plt.plot(n_qubits_range, fidelities)
        plt.ylim((0.0, 1.0))
        legend_str.append('%s' % ('All' if n_measured_operators is None else ('%d' % (n_measured_operators))))
    plt.plot(n_qubits_range, [get_true_fidelity(circuit_id, n_qubits, noise) for n_qubits in n_qubits_range])
    plt.ylabel('Fidelity')
    plt.legend(legend_str + ["True"])
    plt.grid()

    plt.subplot(3, 1, 2)
    for n_measured_operators in n_measured_operators_range:
        errors = []
        for n_qubits in n_qubits_range:
            true_fidelity = get_true_fidelity(circuit_id, n_qubits, noise)
            raw_fidelity = get_one_study(
                circuit_id=circuit_id, n_repetitions=n_repetitions, n_qubits=n_qubits,
                n_measured_operators=n_measured_operators, samples_per_term=samples_per_term,
                noise=noise)
            errors.append(math.sqrt(np.mean([(x - true_fidelity)**2 for x in raw_fidelity])))
        plt.plot(n_qubits_range, errors)

    plt.ylabel('L2 error')
    plt.grid()
    plt.legend(legend_str)
    #plt.ylim((0.0, 0.25))

    plt.subplot(3, 1, 3)
    fidelity_clifford_l2 = []
    for n_qubits in n_qubits_range:
      result = run_one_study(circuit_id=circuit_id,
                             n_repetitions=1,
                             n_qubits=n_qubits,
                             n_measured_operators=None,
                             samples_per_term=samples_per_term,
                             noise=noise)['results'][0]
      trial_results = result['intermediate_result']['trial_results']
      pauli_traces = result['intermediate_result']['pauli_traces']

      assert len(trial_results) == 2**n_qubits
      assert len(pauli_traces) == 2**n_qubits

      fidelity_comps = [x[0]['sigma_i'] / x[1]['rho_i'] for x in zip(trial_results, pauli_traces)]
      estimated_fidelity = result['estimated_fidelity']

      assert np.isclose(np.mean(fidelity_comps), estimated_fidelity, atol=1e-6)

      fidelity_clifford_l2.append(math.sqrt(np.mean([(x - estimated_fidelity)**2 for x in fidelity_comps])))

    plt.plot(n_qubits_range, fidelity_clifford_l2, 'k')
    #plt.ylim((0.0, 0.25))
    plt.xlabel('#qubits')
    plt.ylabel('L2 error')
    plt.grid()
    plt.legend(['Within Clifford'])

    plt.savefig('examples/dfe.png', format='png')


if __name__ == '__main__':
    main()
