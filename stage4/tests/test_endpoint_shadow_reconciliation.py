import unittest
from stage4.tools.reconcile_native_endpoint_shadow import transition_component, classify_anchors


class FakeNative:
    def __init__(self, links): self.links = links
    def transitions(self, node): return self.links.get(node, [])


class EndpointShadowTests(unittest.TestCase):
    def test_only_reciprocal_native_transitions_form_aliases(self):
        self.assertEqual(transition_component(FakeNative({1:[2],2:[1,3],3:[2]}),1),[1,2,3])
        with self.assertRaisesRegex(RuntimeError,'Nonreciprocal'):
            transition_component(FakeNative({1:[2]}),1)

    def test_absent_node_is_not_certified_outside_complex(self):
        self.assertEqual(classify_anchors([1,2],{3:'n3'},{'n3':'c3'}),('NO_FROZEN_NODE_ANCHOR',[],[]))
        self.assertEqual(classify_anchors([1],{1:'n1'},{}),('KNOWN_FROZEN_NODES_WITHOUT_COMPLEX',[1],[]))

    def test_multiple_complexes_are_not_arbitrarily_selected(self):
        status, _, complexes = classify_anchors([1,2],{1:'n1',2:'n2'},{'n1':'c1','n2':'c2'})
        self.assertEqual(status,'CONFLICT_MULTIPLE_FROZEN_COMPLEXES')
        self.assertEqual(complexes,['c1','c2'])
        self.assertEqual(classify_anchors([1,2],{2:'n2'},{'n2':'c2'})[0],'UNIQUE_FROZEN_COMPLEX_ANCHOR')


if __name__=='__main__': unittest.main()
