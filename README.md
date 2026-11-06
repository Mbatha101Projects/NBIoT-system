# Optimization of Energy Efficiency in NB-IoT Devices Using Power saving Mechanism

Honours mini-dissertation code, University of Zululand (2026).
Author: Mawande Mbatha. Supervisors: Dr P. Tarwireyi, Mr T. Makhalanyane (CSIR).

## Layout
- q-learning/  Python: smp_engine.py (six-state SMP model), environment.py, agent.py (Double Q-learning),
  train.py, evaluate.py, multi_seed_evaluation.py, audit_production_policy.py, monte_carlo_uncertainty.py,
  transfer_learning_bg96.py, overhead_analysis.py, literature replication scripts.
- ns-3-simulation/  nbiot-smp-benchmark.cc (computeSmp and the NS-3 LTE run), run_*.sh sweep scripts,
  results CSVs (results.csv, results_group_e.csv, results_group_f.csv) and the plotting scripts.

## Running
- Python scripts: run from the q-learning folder (smp_engine.py must sit next to them).
- NS-3: copy nbiot-smp-benchmark.cc to ~/ns-3-dev/scratch/, build with ./ns3 build,
  then run bash run_benchmark.sh (and the group E/F scripts).
- The production policy is results/q_table.npy (training seed 14, 30/30 on the audit).

The Python model (smp_engine.py) and computeSmp in the .cc are kept identical.
