from engine.schedules import warmup_cosine_factor


def test_warmup_cosine_factor_shapes_schedule() -> None:
    factors = [warmup_cosine_factor(epoch, total_epochs=100, warmup_epochs=10) for epoch in range(101)]
    assert factors[0] == 0.1  # linear warmup starts at 1/warmup_epochs
    assert factors[10] == 1.0  # peak at the end of warmup
    assert all(factors[i] <= factors[i + 1] for i in range(10))
    assert factors[50] > factors[99]  # cosine decay
    assert factors[99] < 0.01
