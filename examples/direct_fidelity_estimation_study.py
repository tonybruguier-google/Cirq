import dataclasses
import os
from typing import cast
from typing import List
from typing import Tuple

import cirq
import examples.direct_fidelity_estimation as dfe
import recirq

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

    def _build_circuit(self) -> Tuple[cirq.Circuit, List[cirq.Qid]]:
        assert self.n_qubits >= 1

        # Builds a Clifford circuit to be studied.
        qubits: List[cirq.Qid] = cast(List[cirq.Qid],
                                      cirq.LineQubit.range(self.n_qubits))
        circuit: cirq.Circuit = cirq.Circuit()

        for i in range(0, self.n_qubits):
            circuit.append(cirq.H(qubits[i]))
        for i in range(1, self.n_qubits):
            circuit.append(cirq.CZ(qubits[i - 1], qubits[i]))
        return qubits, circuit

    def run(self):
        qubits, circuit = self._build_circuit()
        noisy_simulator = cirq.DensityMatrixSimulator(noise=cirq.depolarize(self.noise))

        return dfe.direct_fidelity_estimation(
                                circuit,
                                qubits,
                                noisy_simulator,
                                n_trials=self.n_trials,
                                n_clifford_trials=self.n_clifford_trials,
                                samples_per_term=0)

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
                base_dir=os.path.expanduser(f'~/cirq_results/study'))

def main():
    run_one_study(n_qubits=2, n_trials=3, n_clifford_trials=1, noise=0.1)

if __name__ == '__main__':
    main()
