import unittest
from stage4.tools.audit_native_node_complex_context import context_class


class NativeNodeContextTests(unittest.TestCase):
    def test_distance_is_review_not_assignment(self):
        self.assertEqual(context_class(10,{'c'}),'WITHIN_FROZEN_CANDIDATE_BUFFER_REVIEW')
        self.assertEqual(context_class(20,{'c'}),'WITHIN_CANDIDATE_PAIR_DISTANCE_REVIEW')

    def test_far_adjacency_does_not_choose_a_complex(self):
        self.assertEqual(context_class(21,{'a','b'}),'OUTSIDE_BUFFERS_MULTICOMPLEX_ADJACENCY')
        self.assertEqual(context_class(21,{'a'}),'OUTSIDE_BUFFERS_SINGLE_COMPLEX_ADJACENCY')
        self.assertEqual(context_class(21,set()),'OUTSIDE_BUFFERS_NO_DIRECT_COMPLEX_ANCHOR')


if __name__=='__main__': unittest.main()
