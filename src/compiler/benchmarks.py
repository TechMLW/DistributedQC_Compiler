from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.circuit.library import EfficientSU2
import random


class Benchmarks:

    def ghz(self, n):

        qc = QuantumCircuit(n)

        qc.h(0)

        for i in range(n - 1):
            qc.cx(i, i + 1)

        return qc


    def qft(self, n):

        return QFT(n)


    def random(self, n, depth=20):

        qc = QuantumCircuit(n)

        for _ in range(depth):

            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)

            while a == b:
                b = random.randint(0, n - 1)

            qc.cx(a, b)

        return qc


    def hardware_efficient(self, n):

        return EfficientSU2(n, reps=2)