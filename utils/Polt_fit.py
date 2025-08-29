import numpy as np
import matplotlib.pyplot as plt
y = np.load("ankle_height.npy")
# 示例数据
x = np.linspace(0,0.5,len(y))


# 进行二次多项式拟合
coefficients = np.polyfit(x, y, 5)
print("多项式系数:", coefficients)

# 创建多项式函数
poly_func = np.poly1d(coefficients)
print("多项式方程:", poly_func)

# 生成拟合曲线
x_fit = np.linspace(0, 0.5, 100)
y_fit = poly_func(x_fit)

# 绘制结果
plt.scatter(x, y, )
plt.plot(x_fit, y_fit, 'r-')
plt.legend()
plt.show()


print()