import importlib
import subprocess
import sys
import unittest


def _loaded_modules(code: str) -> set[str]:
    script = code + '\nimport sys\nfor name in sorted(sys.modules):\n    print(name)\n'
    output = subprocess.check_output([sys.executable, '-c', script], text=True)
    return set(output.split())


class SearchLazyExportTest(unittest.TestCase):
    EXPECTED = {
        'MCTSNode': 'mcts', 'mcts_search': 'mcts', 'mcts_search_v3': 'mcts_v3',
        'mcts_search_v32': 'mcts_v32', 'mcts_search_v321': 'mcts_v321',
        'mcts_search_v4': 'mcts_v4', 'mcts_search_v5': 'mcts_v5', 'mcts_search_v6': 'mcts_v6',
    }

    def test_all_names_importable_and_identical(self):
        import search
        self.assertEqual(set(search.__all__), set(self.EXPECTED))
        for name, module_name in self.EXPECTED.items():
            namespace = {}
            exec(f'from search import {name}', namespace)
            submodule = importlib.import_module(f'search.{module_name}')
            self.assertIs(namespace[name], getattr(submodule, name), name)
            self.assertIs(getattr(search, name), getattr(submodule, name), name)

    def test_star_import_and_unknown_name(self):
        namespace = {}
        exec('from search import *', namespace)
        self.assertTrue(set(self.EXPECTED) <= set(namespace))
        import search
        with self.assertRaises(AttributeError):
            search.not_an_export
        with self.assertRaises(ImportError):
            exec('from search import not_an_export', {})

    def test_package_import_is_lazy(self):
        loaded = _loaded_modules('import search')
        self.assertIn('search', loaded)
        for name in ('search.mcts', 'search.mcts_v5', 'search.mcts_v6',
                     'search.threat_patterns', 'search.threat_planning'):
            self.assertNotIn(name, loaded)

    def test_submodule_import_still_eager_for_its_own_dependencies(self):
        loaded = _loaded_modules('from search import mcts_search_v6')
        self.assertIn('search.mcts_v6', loaded)
        self.assertIn('search.threat_planning', loaded)


if __name__ == '__main__':
    unittest.main()
