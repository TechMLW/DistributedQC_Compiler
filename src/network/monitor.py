import random

class NetworkMonitor:

    def __init__(self):
        self.links = {}

    def register_link(self, source, destination,
                      latency, fidelity,
                      bell_pairs, congestion):

        self.links[(source, destination)] = {
            "latency": latency,
            "fidelity": fidelity,
            "bell_pairs": bell_pairs,
            "congestion": congestion
        }

    def get_state(self, source, destination):

        if (source, destination) in self.links:
            return self.links[(source, destination)]

        if (destination, source) in self.links:
            return self.links[(destination, source)]

        return None

    def update(self):

        for state in self.links.values():

            state["latency"] += random.randint(-2, 4)
            state["latency"] = max(1, state["latency"])

            state["fidelity"] += random.uniform(-0.01, 0.005)
            state["fidelity"] = max(0.80,
                                    min(1.0, state["fidelity"]))

            state["bell_pairs"] += random.randint(-3, 3)
            state["bell_pairs"] = max(0, state["bell_pairs"])

            state["congestion"] += random.randint(-5, 8)
            state["congestion"] = max(
                0,
                min(100, state["congestion"])
            )

    def print_state(self):

        print("\n===== Network State =====")

        for link, state in self.links.items():

            print(
                f"{link} | "
                f"Latency={state['latency']} ms | "
                f"Fidelity={state['fidelity']:.3f} | "
                f"BellPairs={state['bell_pairs']} | "
                f"Congestion={state['congestion']}%"
            )