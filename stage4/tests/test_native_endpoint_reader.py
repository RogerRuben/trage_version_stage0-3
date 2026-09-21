"""Small read-only integration checks against the explicitly frozen local tiles."""
import json
import unittest
from valhalla.baldr import GraphId
from stage4.tools.read_native_endpoint_candidates import NativeTiles, CONFIG


@unittest.skipUnless(CONFIG.is_file(), 'Frozen local Valhalla tiles required')
class NativeEndpointReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = NativeTiles(json.loads(CONFIG.read_text())['mjolnir']['tile_dir'])

    def test_native_owner_and_reciprocal_opposing_edge(self):
        edge = self.graph.edge(511101131120)
        reverse = self.graph.edge(edge['opposing_edge_id'])
        self.assertEqual(edge['source'], 207467075952)
        self.assertEqual(edge['way_id'], 162532779)
        self.assertEqual((reverse['source'], reverse['target']), (edge['target'], edge['source']))

    def test_out_of_range_identity_is_not_snapped(self):
        gid = GraphId(511101131120)
        invalid = int(GraphId(gid.tileid(), gid.level(), (1 << 21)-1).value)
        with self.assertRaisesRegex(RuntimeError, 'out of bounds'):
            self.graph.edge(invalid)
        with self.assertRaisesRegex(RuntimeError, 'out of bounds'):
            self.graph.node(invalid)

    def test_same_level_conflict_is_not_a_hierarchy_alias(self):
        edge = self.graph.edge(759773026672)
        self.assertEqual(edge['target'], 309606766960)
        self.assertNotEqual(edge['target'], 309438994800)
        self.assertNotIn(edge['target'], self.graph.transitions(309438994800))


if __name__ == '__main__':
    unittest.main()
