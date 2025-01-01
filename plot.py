import matplotlib.pyplot as plt

# 读取数据
data = []
with open('part1_timeAnalysis.dat', 'r') as f:
    lines = f.readlines()

# 跳过文件的第一行（标题）
for line in lines[1:]:
    parts = line.split()
    number_thread = int(parts[0])
    time = float(parts[1])
    speedup = float(parts[2])
    efficiency = float(parts[3])
    data.append((number_thread, time, speedup, efficiency))

# 分离数据
number_threads = [d[0] for d in data]
times = [d[1] for d in data]
speedups = [d[2] for d in data]
efficiencies = [d[3] for d in data]

# 绘制图形
plt.figure(figsize=(10, 6))

# 绘制时间
plt.plot(number_threads, times, marker='o', label='Time(s)', color='blue')
# 绘制加速比
plt.plot(number_threads, speedups, marker='o', label='Speedup', color='green')
# 绘制效率
plt.plot(number_threads, efficiencies, marker='o', label='Efficiency', color='red')

# 添加图例
plt.legend()

# 设置标题和标签
plt.title('Performance Metrics')
plt.xlabel('Number of Threads')
plt.ylabel('Value')
plt.xticks(number_threads)  # 设置横坐标刻度
plt.grid()

# 显示图形
plt.show()