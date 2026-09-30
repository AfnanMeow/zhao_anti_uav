import json, numpy as np

with open('all_test_results.json') as f:
    data = json.load(f)

rhos       = data['rhos']           # total budget per client
sigmas_drl = data['sigmas_drl']     # sigma per client per round
el         = 5                      # epoch_local

print("Budget verification:")
for i in range(3):
    rho_used = sum(2*el / (s**2) for s in sigmas_drl[i])
    print(f"  Client {i}: budget={rhos[i]:.3f}, spent={rho_used:.3f}, satisfied={rho_used <= rhos[i]}")