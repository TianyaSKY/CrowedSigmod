import torch

from data.target_generator import crop_points, generate_density_target, generate_probability_target, validate_density_conservation


def test_density_conserves_points_at_boundaries_and_overlap() -> None:
    # 0.0 与 639.0 是 640 图像半开区间 [0, 640) 的两端像素；再叠一对相距仅 1 像素的
    # 相邻点 (320,320)/(321,320)，同时考验边界截断与高斯重叠两种情况下的守恒性。
    points = torch.tensor([[0.0, 0.0], [639.0, 639.0], [320.0, 320.0], [321.0, 320.0]])
    density = generate_density_target(points, output_size=160, output_stride=4, sigma=2.0)
    # 守恒断言：每张密度图积分仍必须等于点数 4；2e-4 容差吸收边界处被截断的高斯
    # 能量（核越贴近图像边缘，被裁剪掉的尾部越多，归一化会补回这部分）。
    validate_density_conservation(density, 4.0, tolerance=2e-4)
    # 守恒之外还要求非负：任何截断/归一化都不应把密度钳出负值。
    assert torch.all(density >= 0)


def test_probability_is_not_mass_normalized() -> None:
    # 两点相距 4px（= 2×sigma），核有重叠；概率图表达"该位置是头的似然"，
    # 每个峰最高为 1，与密度图"积分等于人数"的守恒语义刻意不同。
    points = torch.tensor([[100.0, 100.0], [104.0, 100.0]])
    probability = generate_probability_target(points, output_size=160, output_stride=4, sigma=2.0)
    assert probability.shape == (160, 160)
    # 存在恰为 1.0 的峰值（至少一个核中心未被边界截断），证明生成器没有整体缩放。
    assert float(probability.max()) <= 1.0
    assert float(probability.max()) == 1.0


def test_crop_uses_half_open_membership() -> None:
    # 半开区间 [origin, origin+size)：0.0 属于区间被保留，640.0 恰为上界被排除，
    # 639.0 是最后一个有效像素必须保留 —— 保证分块拼接时边界上的点只归属一个块，
    # 不会被重复计数。
    points = torch.tensor([[0.0, 0.0], [640.0, 100.0], [639.0, 639.0]])
    cropped = crop_points(points, (0, 0), 640)
    assert torch.equal(cropped, torch.tensor([[0.0, 0.0], [639.0, 639.0]]))
