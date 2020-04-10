import argparse
import os
import sys
from typing import List

import cirq
import cirq.google as cg
from cirq.google import gate_sets
import examples.direct_fidelity_estimation as dfe
import recirq


@recirq.json_serializable_dataclass(namespace='recirq.readout_scan',
                                    registry=recirq.Registry,
                                    frozen=True)
class DFETask:
    """A task that runs direct fidelity estimation (DFE) experiments."""
    n_trials: int
    n_clifford_trials: int
    samples_per_term: int

    @property
    def fn(self):
        return '%d_%d_%d' % (self.n_trials, self.n_clifford_trials,
                             self.samples_per_term)

    def _build_circuit(self):
        qubits: List[cirq.Qid] = list(cg.Sycamore23.qubit_set())
        qubits = qubits[:4:]

        circuit: cirq.Circuit = cirq.Circuit()
        for qubit in qubits:
            circuit.append(cirq.H(qubit))
        for qubitA in qubits:
            for qubitB in qubits:
                if qubitA.is_adjacent(qubitB):
                    circuit.append(cirq.CZ(qubitA, qubitB))
        return qubits, circuit

    def run(self):
        qubits, circuit = self._build_circuit()

        noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(0.1))
        sampler = cirq.DensityMatrixSimulator(noise=noise)
        sycamore_circuit = cg.optimized_for_sycamore(circuit,
                                                     optimizer_type='sycamore')
        # sampler = cg.QuantumEngineSampler(
        #     engine=None,
        #     processor_id='tmp',
        #     gate_set=[gate_sets.SQRT_ISWAP_GATESET, gate_sets.SYC_GATESET])

        _, run_log = dfe.direct_fidelity_estimation(
            circuit,
            qubits,
            sampler,
            n_trials=self.n_trials,
            n_clifford_trials=self.n_clifford_trials,
            samples_per_term=self.samples_per_term)

        return run_log


def main(n_trials: int, n_clifford_trials: int, samples_per_term: int):
    task = DFETask(n_trials=n_trials,
                   n_clifford_trials=n_clifford_trials,
                   samples_per_term=samples_per_term)
    run_log = task.run()
    recirq.save(task=task,
                data=run_log,
                base_dir=os.path.expanduser(f'~/cirq-results/dfe'))


def parse_arguments(args):
    """Helper function that parses the given arguments."""
    parser = argparse.ArgumentParser('Direct fidelity experiment')

    parser.add_argument('--n_trials',
                        default=10,
                        type=int,
                        help='Number of trials to run.')

    parser.add_argument('--n_clifford_trials',
                        default=10,
                        type=int,
                        help='Number of trials for Clifford circuits. This is '
                        'in effect when the circuit is Clifford. In this '
                        'case, we randomly sample the Pauli traces with '
                        'non-zero probabilities. The higher the number, '
                        'the more accurate the overall fidelity '
                        'estimation, at the cost of extra computing and '
                        'measurements.')

    parser.add_argument('--samples_per_term',
                        default=10,
                        type=int,
                        help='Number of samples per trial or 0 if no sampling.')

    return vars(parser.parse_args(args))


if __name__ == '__main__':
    main(**parse_arguments(sys.argv[1:]))
