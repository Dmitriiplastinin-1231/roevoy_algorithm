"""
Алгоритм роя частиц (Particle Swarm Optimization, PSO).

Реализация на чистом Python без внешних библиотек.
Используется только стандартный модуль random.
"""

import random
from typing import Callable, List, Tuple, Optional


class Particle:
    """Одна частица роя."""

    def __init__(self, dimensions: int, bounds: List[Tuple[float, float]]):
        """
        Инициализация частицы.

        Args:
            dimensions: количество измерений (переменных функции).
            bounds: список кортежей (min, max) для каждого измерения.
        """
        self.position: List[float] = [
            random.uniform(lo, hi) for lo, hi in bounds
        ]
        self.velocity: List[float] = [
            random.uniform(-(hi - lo), hi - lo) for lo, hi in bounds
        ]
        self.best_position: List[float] = list(self.position)
        self.best_value: float = float("inf")

    def __repr__(self) -> str:
        pos = ", ".join(f"{x:.6f}" for x in self.position)
        return f"Particle([{pos}], best={self.best_value:.6f})"


class PSO:
    """
    Оптимизатор на основе алгоритма роя частиц.

    Ищет глобальный МИНИМУМ целевой функции.
    """

    def __init__(self,
        func: Callable[..., float],
        dimensions: int,
        bounds: List[Tuple[float, float]],
        num_particles: int = 30,
        max_iterations: int = 200,
        w: float = 0.7298,
        c1: float = 1.49618,
        c2: float = 1.49618,
        seed: Optional[int] = None,
    ):
        """
        Args:
            func: целевая функция, принимающая *args (по одному на измерение).
            dimensions: количество измерений.
            bounds: ограничения [(min1, max1), (min2, max2), ...].
            num_particles: размер роя.
            max_iterations: максимальное число итераций.
            w: коэффициент инерции.
            c1: когнитивный коэффициент (влияние личного лучшего).
            c2: социальный коэффициент (влияние глобального лучшего).
            seed: фиксация генератора случайных чисел (для воспроизводимости).
        """
        if seed is not None:
            random.seed(seed)

        self.func = func
        self.dimensions = dimensions
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # Инициализация роя
        self.swarm: List[Particle] = [
            Particle(dimensions, bounds) for _ in range(num_particles)
        ]

        # Глобальные лучшие значения
        self.global_best_position: List[float] = [0.0] * dimensions
        self.global_best_value: float = float("inf")

        # История лучших значений по итерациям (для анализа сходимости)
        self.history: List[float] = []

    # ------------------------------------------------------------------
    # Основной цикл
    # ------------------------------------------------------------------

    def optimize(self, verbose: bool = False) -> Tuple[List[float], float]:
        """
        Запуск оптимизации.

        Args:
            verbose: если True, печатает прогресс каждые 10 % итераций.

        Returns:
            (лучшая_позиция, лучшее_значение)
        """
        # Первичная оценка всех частиц
        self._evaluate_swarm()

        for iteration in range(1, self.max_iterations + 1):
            for particle in self.swarm:
                self._update_velocity(particle)
                self._update_position(particle)

            self._evaluate_swarm()
            self.history.append(self.global_best_value)

            if verbose and (
                iteration % max(1, self.max_iterations // 10) == 0
                or iteration == 1
            ):
                print(
                    f"  Итерация {iteration:>5d}/{self.max_iterations}  "
                    f"лучшее значение = {self.global_best_value:.10f}"
                )

        return list(self.global_best_position), self.global_best_value

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    def _evaluate_swarm(self) -> None:
        """Вычисляет значение функции для каждой частицы и обновляет лучшие."""
        for particle in self.swarm:
            value = self.func(*particle.position)

            # Обновление личного лучшего
            if value < particle.best_value:
                particle.best_value = value
                particle.best_position = list(particle.position)

            # Обновление глобального лучшего
            if value < self.global_best_value:
                self.global_best_value = value
                self.global_best_position = list(particle.position)

    def _update_velocity(self, particle: Particle) -> None:
        """Обновляет скорость частицы по формуле PSO."""
        for i in range(self.dimensions):
            r1 = random.random()
            r2 = random.random()

            cognitive = self.c1 * r1 * (
                particle.best_position[i] - particle.position[i]
            )
            social = self.c2 * r2 * (
                self.global_best_position[i] - particle.position[i]
            )

            particle.velocity[i] = (
                self.w * particle.velocity[i] + cognitive + social
            )

    def _update_position(self, particle: Particle) -> None:
        """Обновляет позицию частицы и ограничивает её рамками bounds."""
        for i in range(self.dimensions):
            particle.position[i] += particle.velocity[i]

            lo, hi = self.bounds[i]
            if particle.position[i] < lo:
                particle.position[i] = lo
                particle.velocity[i] *= -0.5  # отскок от границы
            elif particle.position[i] > hi:
                particle.position[i] = hi
                particle.velocity[i] *= -0.5