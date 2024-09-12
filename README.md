# Quantum-Inspired HTAAC-QSOS

A classical, quantum-inspired algorithm for polynomial optimization.

Original quantum papers: [HTAAC-QSOS](https://scholar.google.com/scholar?oi=bibs&cluster=16761718371063042923&btnI=1&hl=en), [HTAAC-QSDP](https://scholar.google.com/scholar?oi=bibs&cluster=15239233008514417786&btnI=1&hl=en)

Packages: PyTorch, [Tensorly-quantum](https://github.com/tensorly/quantum)

## Running the code for Max3SAT

This example code generates a Max3SAT instance and evaluates it using a classically-simulated version of HTAAC-QSOS. It is run in two steps:
1. Run gen_max3sat.py. This script randomly generates clauses Max3SAT instances and parses through them, saving the instance in the 'gen_max3sat' folder and W matrices in the 'problem' folder.
2. Run htaac-qsos_max3sat.py. This script uses a classically-simulated version of HTAAC-QSOS to approximate solutions to the generated max3sat instance.
