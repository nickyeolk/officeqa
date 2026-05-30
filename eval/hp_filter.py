import numpy as np

receipts = np.array([2161728, 2302495, 2449092, 2773979, 3020847, 3248722, 3266689, 3314893, 3328745, 3462195, 3419955, 4045980, 4896119, 4439283, 4918737], dtype=float)
outlays  = np.array([3455931, 3598086, 3538447, 3454254, 3504199, 3687623, 3854101, 3980720, 4107741, 4446583, 6551871, 6818159, 6271508, 6134433, 6751553], dtype=float)

T = len(receipts)
lam = 100.0

D = np.zeros((T-2, T))
for i in range(T-2):
    D[i, i] = 1.0
    D[i, i+1] = -2.0
    D[i, i+2] = 1.0

F = np.eye(T) + lam * (D.T @ D)

trend_r = np.linalg.solve(F, receipts)
trend_o = np.linalg.solve(F, outlays)

actual_24 = receipts[-1] - outlays[-1]
struct_24 = trend_r[-1] - trend_o[-1]
gap_24 = abs(actual_24 - struct_24)

print(f"trend_receipts_2024={trend_r[-1]}")
print(f"trend_outlays_2024={trend_o[-1]}")
print(f"actual_balance_2024={actual_24}")
print(f"structural_balance_2024={struct_24}")
print(f"gap_2024={gap_24}")
print(f"rounded: actual={round(actual_24)}, structural={round(struct_24)}, gap={round(gap_24)}")
