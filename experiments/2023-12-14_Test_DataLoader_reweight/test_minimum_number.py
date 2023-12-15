p = 1

for i in range(1000):
    if p * 0.01 == 0:
        print(f"p is zero after {i+2} iterations. p = {p} at iteration {i+1}")
        break
    else:
        p *= 0.01