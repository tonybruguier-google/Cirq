import dataclasses
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
    n_qubits: int
    n_trials: int
    n_clifford_trials: int
    noise: float

    @property
    def fn(self):
        clifford_string = 'exhaustive' if self.n_clifford_trials is None else '%d' % (self.n_clifford_trials)
        return '%d_%d_%s_%e' % (self.n_qubits, self.n_trials, clifford_string, self.noise)

    def run(self):
        qubits, circuit = _build_circuit(self.n_qubits)
        noisy_simulator = cirq.DensityMatrixSimulator(noise=_get_noise(self.noise))
        return dfe.direct_fidelity_estimation(
                                circuit,
                                qubits,
                                noisy_simulator,
                                n_trials=self.n_trials,
                                n_clifford_trials=self.n_clifford_trials,
                                samples_per_term=0)

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

def run_one_study(n_qubits: int, n_trials: int, n_clifford_trials: int, noise: float):
    task = DFETask(n_qubits=n_qubits,
                   n_trials=n_trials,
                   n_clifford_trials=n_clifford_trials,
                   noise=noise)
    estimated_fidelity, intermediate_result = task.run()
    data = {'estimated_fidelity': estimated_fidelity,
            'intermediate_result': dataclasses.asdict(intermediate_result)}
    recirq.save(task=task,
                data=data,
                base_dir=os.path.expanduser(f'~/cirq_results/study/dfe'))

def run_true_fidelity(n_qubits: int, noise: float):
    task = TrueFidelityTask(n_qubits=n_qubits, noise=noise)
    true_fidelity = task.run()
    data = {'true_fidelity': true_fidelity}
    recirq.save(task=task,
                data=data,
                base_dir=os.path.expanduser(f'~/cirq_results/study/true'))

def main():
    run_one_study(n_qubits=2, n_trials=3, n_clifford_trials=None, noise=0.1)
    run_one_study(n_qubits=2, n_trials=3, n_clifford_trials=1, noise=0.1)
    run_true_fidelity(n_qubits=2, noise=0.1)

if __name__ == '__main__':
    main()
