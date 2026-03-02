"""
Примеры использования алгоритма роя частиц (PSO).

Запуск:
    python main.py
"""

from pso import PSO


# ---------------------------------------------------------------
# Тестовые функции
# ---------------------------------------------------------------

def sphere(x: float, y: float) -> float:
    """Функция сферы: f(x, y) = x² + y².  Минимум = 0 в точке (0, 0)."""
    return x ** 2 + y ** 2


def rastrigin(x: float, y: float) -> float:
    """
    Функция Растригина (2D).
    Глобальный минимум = 0 в точке (0, 0).
    Множество локальных минимумов — хороший тест для роевого алгоритма.
    """
    import math
    A = 10
    return (
        A * 2
        + (x ** 2 - A * math.cos(2 * math.pi * x))
        + (y ** 2 - A * math.cos(2 * math.pi * y))
    )


def rosenbrock(x: float, y: float) -> float:
    """
    Функция Розенброка (2D).
    Глобальный минимум = 0 в точке (1, 1).
    """
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def ackley(x: float, y: float) -> float:
    """
    Функция Экли (2D).
    Глобальный минимум = 0 в точке (0, 0).
    """
    import math
    term1 = -20 * math.exp(-0.2 * math.sqrt(0.5 * (x ** 2 + y ** 2)))
    term2 = -math.exp(0.5 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y)))
    return term1 + term2 + math.e + 20


def custom_function(x: float) -> float:
    """
    Пример пользовательской функции одной переменной.
    f(x) = (x - 3)² + 5
    Минимум = 5 в точке x = 3.
    """
    return (x - 3) ** 2 + 5


# ---------------------------------------------------------------
# Запуск
# ---------------------------------------------------------------

def run_example(name: str, func, dimensions: int, bounds, known_min: str):
    """Запуск одного теста."""
    print(f"\n{'=' * 60}")
    print(f"  {{name}}")
    print(f"  Известный минимум: {{known_min}}")
    print(f"{'=' * 60}")

    optimizer = PSO(
        func=func,
        dimensions=dimensions,
        bounds=bounds,
        num_particles=40,
        max_iterations=300,
        seed=42,
    )

    best_pos, best_val = optimizer.optimize(verbose=True)

    pos_str = ", ".join(f"{{x:.8f}}" for x in best_pos)
    print(f"\n  Результат:")
    print(f"    Лучшая позиция : ({{pos_str}})")
    print(f"    Лучшее значение: {{best_val:.10f}}")


if __name__ == "__main__":
    print("=" * 60)
    print("  АЛГОРИТМ РОЯ ЧАСТИЦ (PSO) — поиск минимума функции")
    print("=" * 60)

    # 1. Функция сферы
    run_example(
        name="Функция сферы: f(x, y) = x² + y²",
        func=sphere,
        dimensions=2,
        bounds=[(-10, 10), (-10, 10)],
        known_min="0 в точке (0, 0)",
    )

    # 2. Функция Растригина
    run_example(
        name="Функция Растригина (2D)",
        func=rastrigin,
        dimensions=2,
        bounds=[(-5.12, 5.12), (-5.12, 5.12)],
        known_min="0 в точке (0, 0)",
    )

    # 3. Функция Розенброка
    run_example(
        name="Функция Розенброка (2D)",
        func=rosenbrock,
        dimensions=2,
        bounds=[(-5, 10), (-5, 10)],
        known_min="0 в точке (1, 1)",
    )

    # 4. Функция Экли
    run_example(
        name="Функция Экли (2D)",
        func=ackley,
        dimensions=2,
        bounds=[(-5, 5), (-5, 5)],
        known_min="0 в точке (0, 0)",
    )

    # 5. Пользовательская функция одной переменной
    run_example(
        name="Пользовательская: f(x) = (x - 3)² + 5",
        func=custom_function,
        dimensions=1,
        bounds=[(-100, 100)],
        known_min="5 в точке (3)",
    )

    print(f"\n{'=' * 60}")
    print("  Все тесты завершены!")
    print(f"{'=' * 60}")
