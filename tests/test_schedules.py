from engine.schedules import warmup_cosine_factor


def test_warmup_cosine_factor_shapes_schedule() -> None:
    # 收集 0..100 的完整乘数轨迹，一次性断言两个阶段各自的形状（预热爬升、余弦下降）。
    factors = [warmup_cosine_factor(epoch, total_epochs=100, warmup_epochs=10) for epoch in range(101)]
    # 从 1/warmup_epochs 起步而非 0：保证 epoch 0 就有非零学习率，避免首个周期空转。
    assert factors[0] == 0.1  # 线性预热从 1/warmup_epochs 开始
    # 预热终点与余弦起点同在 epoch 10，峰值 1.0 恰落在切换点 —— 衔接处无跳变。
    assert factors[10] == 1.0  # 预热结束时达到峰值
    # 预热段必须单调不降：学习率稳步逼近峰值，不允许回摆。
    assert all(factors[i] <= factors[i + 1] for i in range(10))
    # 预热后进入余弦衰减：末段乘数趋近于 0（约 0.00025），保证收敛稳定。
    assert factors[50] > factors[99]  # 余弦衰减
    assert factors[99] < 0.01
