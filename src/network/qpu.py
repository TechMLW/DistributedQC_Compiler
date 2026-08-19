class QPU:

    def __init__(self, id, capacity):

        self.id = id
        self.capacity = capacity
        self.logical_qubits = []

    def assign(self, qubit):

        if len(self.logical_qubits) < self.capacity:
            self.logical_qubits.append(qubit)
            return True

        return False

    def __str__(self):

        return f"QPU {self.id} | Capacity={self.capacity} | Qubits={self.logical_qubits}"