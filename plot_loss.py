# -*- coding: utf-8 -*-
"""读取 loss.csv，画出训练集/验证集的 loss 曲线。"""
import matplotlib
matplotlib.use('Agg')  # 无界面后端，直接存成图片
import matplotlib.pyplot as plt
import csv

steps, train, val = [], [], []
with open('loss.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # 跳过表头
    for row in reader:
        steps.append(int(row[0]))
        train.append(float(row[1]))
        val.append(float(row[2]))

plt.figure(figsize=(8, 5))
plt.plot(steps, train, label='train loss', color='#2563eb', linewidth=2)
plt.plot(steps, val, label='val loss', color='#f59e0b', linewidth=2)
plt.xlabel('training steps')
plt.ylabel('loss (cross-entropy)')
plt.title('Loss going down = the model is "learning"')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150)
print('saved loss_curve.png  (steps=%d, final train=%.4f, final val=%.4f)'
      % (steps[-1], train[-1], val[-1]))
