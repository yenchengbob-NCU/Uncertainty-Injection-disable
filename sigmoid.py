import numpy as np
import matplotlib.pyplot as plt

# 畫圖範圍，可自行調整
x = np.linspace(-10, 10, 500)

# sigmoid 函數
y = 1 / (1 + np.exp(-x))

# 繪圖
plt.figure(figsize=(8, 5))
plt.plot(x, y)

plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('σ(x)')
plt.grid(True)
plt.show()