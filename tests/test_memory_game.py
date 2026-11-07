import unittest
import numpy as np
from boltzmann_lab.core.config import DT, WORLD_W
from boltzmann_lab.core.memory_game import MemoryGame, find_wells, preset_potential


class MemoryGameTests(unittest.TestCase):
    def make_game(self, wells=2):
        game = MemoryGame(seed=2026)
        game.set_potential(preset_potential(wells))
        return game

    def finish(self, game):
        while game.phase == "evolving":
            game.advance(DT)

    def test_presets_have_expected_basins(self):
        for count in (2, 3, 4):
            wells = find_wells(preset_potential(count))
            self.assertEqual(len(wells), count)
            for i, well in enumerate(wells):
                self.assertAlmostEqual(well.minimum, (i + 0.5) / count, delta=0.004)
                self.assertAlmostEqual(well.left, i / count, delta=0.004)
                self.assertAlmostEqual(well.right, (i + 1) / count, delta=0.004)

    def test_custom_wells_and_small_drawing_ripples(self):
        x = np.linspace(0, 1, 1024)
        potential = np.cos(3 * np.pi * x) ** 2 + 0.004 * np.sin(130 * np.pi * x)
        self.assertEqual(len(find_wells(potential)), 3)
        for invalid in (np.zeros(1024), (x - 0.5) ** 2, np.full(1024, np.nan), np.ones(10)):
            with self.assertRaises(ValueError):
                find_wells(invalid)

    def test_initial_particles_stay_in_selected_well_without_overlaps(self):
        game = self.make_game(3)
        for origin in (0, 1, 2):
            game.clear_round()
            game.start(origin=origin, temperature=0.02, count=60, duration=0.1)
            well = game.wells[origin]
            x = game.sim.positions[:, 0] / WORLD_W
            self.assertTrue(np.all((x > well.left) & (x < well.right)))
            delta = game.sim.positions[:, None] - game.sim.positions[None, :]
            distances = np.sum(delta ** 2, axis=2)
            np.fill_diagonal(distances, np.inf)
            self.assertGreaterEqual(distances.min(), (2 * game.sim.radius) ** 2)
            self.assertAlmostEqual(game.sim.effective_temperature(), 0.02)
            self.assertTrue(np.array_equal(game.initial_positions, game.sim.positions))

    def test_round_stops_exactly_at_target_and_cannot_be_answered_early(self):
        game = self.make_game()
        game.start(duration=0.1, count=20)
        with self.assertRaises(ValueError):
            game.answer(0)
        with self.assertRaises(ValueError):
            game.set_potential(preset_potential(3))
        self.finish(game)
        self.assertEqual(game.phase, "guess")
        self.assertEqual(game.elapsed, 0.1)
        positions = game.sim.positions.copy()
        game.advance(10)
        self.assertTrue(np.array_equal(positions, game.sim.positions))
        self.assertEqual(game.initial_positions.shape, positions.shape)

    def test_score_counts_each_round_once_and_tracks_chance(self):
        game = self.make_game()
        game.start(duration=0.1, count=20, origin=1)
        initial = game.initial_positions.copy()
        self.finish(game)
        result = game.answer(1)
        self.assertTrue(result.correct)
        self.assertEqual((game.correct, game.played, game.streak), (1, 1, 1))
        self.assertTrue(np.array_equal(game.initial_positions, initial))
        with self.assertRaises(ValueError):
            game.answer(1)
        game.set_potential(preset_potential(3))
        game.start(duration=0.1, count=20, origin=2)
        self.finish(game)
        self.assertFalse(game.answer(0).correct)
        self.assertEqual((game.correct, game.played, game.streak), (1, 2, 0))
        self.assertAlmostEqual(game.chance_sum, 1 / 2 + 1 / 3)

    def test_random_origin_is_seeded_and_covers_all_wells(self):
        game = self.make_game(3)
        seen = set()
        for _ in range(20):
            game.clear_round()
            game.start(duration=0.1, count=10)
            seen.add(game.origin)
        self.assertEqual(seen, {0, 1, 2})

    def test_edge_wells_and_multiple_custom_wells(self):
        x = np.linspace(0, 1, 1024)
        edge = find_wells(1 - (2 * x - 1) ** 2)
        self.assertEqual(len(edge), 2)
        self.assertEqual(edge[0].minimum, 0)
        self.assertEqual(edge[1].minimum, 1)
        self.assertEqual(len(find_wells(np.cos(6 * np.pi * x) ** 2)), 6)
        with self.assertRaises(ValueError):
            find_wells(np.cos(7 * np.pi * x) ** 2)

    def test_invalid_parameters_do_not_start_round(self):
        for kwargs in ({"temperature": 0}, {"duration": 0}, {"height": float("inf")}, {"origin": 8}):
            game = self.make_game()
            with self.assertRaises(ValueError):
                game.start(**kwargs)
            self.assertEqual(game.phase, "setup")
