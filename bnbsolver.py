#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import heapq
import math
from functools import total_ordering
from types import *


#   graphs ----------------------------------------------------------------------------------------

class BitVector(list):
    """
    Quick lousy implementation of vector<bool>
    """

    BITS_PER_ELEMENT = int(math.log( long(sys.maxint)+1, 2 ))
    

    def __init__(self):
        list.__init__(self)
    
    def set_bit(self, index, value):
        element = index / BitVector.BITS_PER_ELEMENT
        bit_offset = index % BitVector.BITS_PER_ELEMENT
        
        if (len(self) <= element):
            self.extend( (0,) * (element - len(self) + 1) )
        
        if value:
            self[element] |= 1 << bit_offset
        else:
            self[element] &= ~(1 << bit_offset)
    
    def get_bit(self, index):
        element = index / BitVector.BITS_PER_ELEMENT
        bit_offset = index % BitVector.BITS_PER_ELEMENT
        
        if (len(self) <= element):
            return False
        
        return 0 != (self[element] & (1 << bit_offset))
        

class AdjMatrix(BitVector):

    def __init__(self, node_count):
        BitVector.__init__(self)
        self.node_count = node_count
        # hack, for bitvector allocation
        if node_count > 1:
            self.set_edge(node_count - 2, node_count - 1, False)
    
    def __cell_position(self, nodeX, nodeY):
        if nodeX >= self.node_count or nodeY >= self.node_count:
            raise RuntimeError('bad node id')
        if nodeX > nodeY:
            (nodeX, nodeY) = (nodeY, nodeX)
        
        return (nodeX * self.node_count - nodeX*(nodeX + 1)/2   # start of row offset
                            + (nodeY - nodeX - 1))              # offset in the row
    
    def set_edge(self, nodeX, nodeY, is_present):
        if nodeX == nodeY:
            return
        
        self.set_bit( self.__cell_position(nodeX, nodeY), is_present )
    
    def has_edge(self, nodeX, nodeY):
        if nodeX == nodeY:
            return True
        
        return self.get_bit( self.__cell_position(nodeX, nodeY) )


class ListAdjMatrix(list):
    
    def __init__(self):
        list.__init__(self)
    
    def set_edge(self, nodeX, nodeY, is_present):
        if nodeX == nodeY:
            return
        if nodeX > nodeY:
            (nodeX, nodeY) = (nodeY, nodeX)
        
        if len(self) <= nodeX:
            self.extend( [set() for i in xrange(nodeX - len(self) + 1)] )
        
        if is_present:
            self[nodeX].add(nodeY)
        else:
            self[nodeX].discard(nodeY)
    
    def has_edge(self, nodeX, nodeY):
        if nodeX == nodeY:
            return True
        if nodeX > nodeY:
            (nodeX, nodeY) = (nodeY, nodeX)
        
        if len(self) <= nodeX:
            return False
        
        return nodeY in self[nodeX]
        
    def simple_form(self, node_count):
        ret = [set() for i in xrange(node_count)]
        for nodeX in xrange(len(self)):
            for nodeY in self[nodeX]:
                ret[nodeX].add(nodeY)
                ret[nodeY].add(nodeX)
        
        return map(lambda set_: sorted(set_), ret)


#   branches and bounds ---------------------------------------------------------------------------

class DeikstraModel(object):
    
    def __init__(self, node_count, adj_matrix, start_vertex, target_vertex):
        self.node_count = node_count
        self.adj_matrix = adj_matrix
        self.start_vertex = start_vertex
        self.target_vertex = target_vertex
        
        self.cfg = {
            'generation_portion':   1,
        }
    
    def initial_decompose(self):
        return [ DeikstraModel.Node(self, None, self.start_vertex) ]
    
    @total_ordering
    class Node(object):
        """
        Node of a solution tree

        Instances of this are compared according to target function
              (node with the lowest target function is considered as the best solution)
        """
        def __init__(self, model, parent, current_vertex):
            self.model = model
            self.parent = parent
            self.current_vertex = current_vertex
            
            self.pathlen = 0 if None == parent else parent.pathlen + 1
            
        def is_solution(self):
            """
            Returns true if this node represents solution of the task (leaf of solution tree)
            NOTE: for solution nodes weight function and target function must return the same value
            """
            return self.model.target_vertex == self.current_vertex
        
        def decompose(self):
            """
            Decomposition of a node
            """
            #ret = []
            #for another in self.model.adj_matrix[self.current_vertex]:
                #if None == self.parent or another != self.parent.current_vertex:
                    #ret.append( DeikstraModel.Node(self.model, self, another) )
            #return ret
            return self
        
        def __iter__(self):
            for another in self.model.adj_matrix[self.current_vertex]:
                if None == self.parent or another != self.parent.current_vertex:
                    yield DeikstraModel.Node(self.model, self, another)
        
        def get_state(self):
            """
            Get representation of a solution state (used to cut branches) or None
            """
            return self.current_vertex
    
    
        def __lt__(self, other):
            return self.pathlen < other.pathlen
        def __eq__(self, other):
            if NoneType == type(other):
                return False
            return self.pathlen == other.pathlen
        def __ne__(self, other):
            if NoneType == type(other):
                return True
            return self.pathlen != other.pathlen
    
    
    @total_ordering
    class NodeWeightComparer(object):
        """
        Instances of this class
            - participate in current heap and serve as pointers to nodes in a solution tree
            - compare (in heap) nodes according to weight function - measurement of potential
              (node with the lowest weight is decomposed first)
        """
        def __init__(self, node):
            self.node = node
        
        def __lt__(self, other):
            return self.node.pathlen < other.node.pathlen
        def __eq__(self, other):
            return self.node.pathlen == other.node.pathlen
        def __ne__(self, other):
            return self.node.pathlen != other.node.pathlen


class BNBSolver(object):
    
    DEFAULT_CFG = {
        'generation_portion':       10,
        'parallel_explore_angles':  map(lambda x: float(x)/20, xrange(21)),
        'parallel_explore_nths':    [1, 3, -2, -4],
        'custom_filters':           [],
    }
    
    def __init__(self, model):
        
        self.model = model
        self.model_cls = model.__class__
        
        if hasattr(model, 'cfg'):
            self.cfg = dict(BNBSolver.DEFAULT_CFG)
            self.cfg.update(model.cfg)
        else:
            self.cfg = BNBSolver.DEFAULT_CFG
        
        self.__current_heap = []
        
        self.__best_result = None
        self.__best_result_weight = None    # cache
        
        self.__bests_bystate = dict()
        
        # stat
        self.__nodes_seen = 0
        self.__nodes_decomposed = 0

    def solve(self):
        """ returns solution node or None """
        
        parallel_explorer_list = self.cfg.get('parallel_explore_nths')
        if parallel_explorer_list:
            for nfilter in map(lambda n: BNBSolver.NodeFilterNthItem(n), parallel_explorer_list):
                self._explore_iteration(nfilter)
        
        parallel_explorer_list = self.cfg.get('parallel_explore_angles')
        if parallel_explorer_list:
            for nfilter in map(lambda angle: BNBSolver.NodeFilterDFSLine(angle), parallel_explorer_list):
                self._explore_iteration(nfilter)
        
        custom_explorer_list = self.cfg.get('custom_filters')
        if custom_explorer_list:
            for nfilter in custom_explorer_list:
                self._explore_iteration(nfilter)
        
        return self._explore_iteration( BNBSolver.NodeFilterAll() )
    
    def _explore_iteration(self, node_filter):
        
        # init iteration - partly reinit solver
        self.__current_heap = []
        
        decomposition = self._preprocess_decomposition(self.model.initial_decompose(), node_filter, 0)
        self._merge_decomposition(decomposition)

        while True:
            # check for result
            if self.__best_result:
                if not self.__current_heap or self.__best_result_weight <= self.__current_heap[0]:
                    return self.__best_result
            else:
                if not self.__current_heap:
                    return None
            
            decomposition = self._preprocess_decomposition(
                    self._decompose_node(self.__current_heap[0].node, node_filter),
                    node_filter,
                    self.__current_heap[0].node.slntree_depth)
            heapq.heappop(self.__current_heap)
            self._merge_decomposition(decomposition)

    def _decompose_node(self, node, node_filter):
        decomposition = node.decompose()
        if type(decomposition) == type(node):
            if BNBSolver.NodeFilterAll != type(node_filter):    # hack -
                decomposition = map(None, node)                 # fetch all anyway
                self.__nodes_decomposed += 1
            else:
                # generator protocol is activated
                decomposition = []
                if hasattr(node, 'active_iter'):    # storing iterator.. todo: do it without hack, in base class
                    it = node.active_iter
                else:
                    it = iter(node)
                    node.active_iter = it
                for i in xrange(self.cfg.get('generation_portion')):
                    n = next(it, None)
                    if None == n:
                        delattr(node, 'active_iter')
                        self.__nodes_decomposed += 1
                        break
                    decomposition.append(n)
                if None != n:   # got more nodes to decompose, node is put on hold
                    decomposition.append(node)
                    self.__nodes_seen -= 1  # and yet another hack..
            
        return decomposition
        
    def _preprocess_decomposition(self, node_list, node_filter, slntree_depth):
        self.__nodes_seen += len(node_list)
        node_list = node_filter.nfilter(node_list, slntree_depth)
        for node in node_list:
            node.slntree_depth = slntree_depth + 1
        
        return node_list
    
    def _merge_decomposition(self, node_list):
        
        for node in node_list:
            
            state = node.get_state()    # tryin to cut in special nodes of a solution tree
            if None != state:
                best_for_state = self.__bests_bystate.get(state, None)
                if None == best_for_state or node < best_for_state:
                    self.__bests_bystate[state] = node
                else:
                    if node != best_for_state:  # node > best_for_state
                        continue
            
            if node.is_solution():
                if None == self.__best_result or node < self.__best_result:
                    self.__best_result = node
                    self.__best_result_weight = self.model_cls.NodeWeightComparer(node)
            
            else:
                weight_cmp = self.model_cls.NodeWeightComparer(node)
                # tryin to cut using best known solution
                # (dont forget, weight function == target function for solutions..
                # ..and, weight always grows (while searching for minimum))
                if None != self.__best_result and weight_cmp >= self.__best_result_weight:
                    continue

                heapq.heappush(self.__current_heap, weight_cmp)
        

    def stat(self):
        print 'Nodes seen: %d' % self.__nodes_seen
        print 'Nodes decomposed: %d' % self.__nodes_decomposed
        print 'Nodes still in current set: %d' % len(self.__current_heap)
        
        node_ids = set()
        for node in map(lambda x: x.node, self.__current_heap) \
                + ([self.__best_result] if None != self.__best_result else []) \
                + self.__bests_bystate.values():
            while None != node:
                nid = id(node)
                if nid in node_ids:
                    break
                node_ids.add(nid)
                node = node.parent
        print 'Nodes still in memory: %d' % len(node_ids)
    
    
    class NodeFilterAll(object):
        """
        Accept-all decomposition filter
        """
        
        def nfilter(self, node_list, slntree_depth):
            return node_list
    
    class NodeFilterDFSLine(object):
        """
        Depth-first explorer (without backtracking, though)
        """
        
        def __init__(self, direction_angle):
            if not (0 <= direction_angle <= 1):
                raise RuntimeError('bad angle: %s' % direction_angle)
            self.direction_angle = direction_angle
        
        def nfilter(self, node_list, slntree_depth):
            if 0 == len(node_list):
                return node_list
            return [ node_list[ int((len(node_list)-1) * self.direction_angle) ] ]
    
    class NodeFilterNthItem(object):
        """
        Filter which always tries to choose nth node from decomposition
        """
        
        def __init__(self, n):
            self.n = n
        
        def nfilter(self, node_list, slntree_depth):
            if 0 == len(node_list):
                return node_list
            n = self.n
            if n >= 0:
                if n >= len(node_list):
                    n = int((len(node_list)-1) / 2)
            else:
                if abs(n) > len(node_list):
                    n = int((len(node_list)-1) / 2)
            return [ node_list[n] ]
        
    

m = ListAdjMatrix()
for i in xrange(100):
    if 0 != (i % 10):
        m.set_edge(i, i-1, True)
    if 9 != (i % 10):
        m.set_edge(i, i+1, True)
    if i >= 10:
        m.set_edge(i, i-10, True)
    if i < 90:
        m.set_edge(i, i+10, True)

for i in (0, 11, 33, 44, 55, 66, 77, 88):
    m.set_edge(i, i+11, True)

solver = BNBSolver( DeikstraModel(100, m.simple_form(100), 0, 99) )
sln = solver.solve()
while sln:
    print sln.current_vertex
    sln = sln.parent

print
solver.stat()



class MaxProductModel(object):
    
    def __init__(self, lst):
        self.lst = lst
        self.max_abs_position = sum(lst)
        
        self.cfg = {
            'generation_portion':   4,
        }
    
    def initial_decompose(self):
        return [ MaxProductModel.Node(self, None, 0, 0, 0, 0) ]
    
    @total_ordering
    class Node(object):
        def __init__(self, model, parent, current_idx, current_sub_idx, step, abs_position):
            self.model = model
            self.current_idx = current_idx
            self.current_sub_idx = current_sub_idx
            self.parent = parent
            self.step = step
            self.abs_position = abs_position
            
            self.product = 1 if None == parent else parent.product * step
            self.product = float(self.product) / 4
            
        def is_solution(self):
            return self.abs_position == self.model.max_abs_position
        
        def decompose(self):
            ret = []
            current_idx = self.current_idx
            current_sub_idx = self.current_sub_idx
            if current_sub_idx == self.model.lst[current_idx]:
                current_idx += 1
                current_sub_idx = 0
                if current_idx >= len(self.model.lst):
                    return []
            for step in (1, 2, 3):
                if current_sub_idx + step > self.model.lst[current_idx]:
                    # takeover situation
                    if current_sub_idx != self.model.lst[current_idx] - 1:
                        break
                    if current_idx+1 >= len(self.model.lst):
                        break
                    if step - 1 > self.model.lst[current_idx+1]:
                        break
                    new_current_idx = current_idx + 1
                    new_current_sub_idx = step - 1
                else:
                    # no takeover
                    new_current_idx = current_idx
                    new_current_sub_idx = current_sub_idx + step
                
                new_abs_position = self.abs_position + step
                if new_abs_position > self.model.max_abs_position:
                    break
                
                ret.append( MaxProductModel.Node(self.model, self, new_current_idx, new_current_sub_idx, step, new_abs_position) )
            
            return ret
        
        def get_state(self):
            return self.abs_position
    
    
        def __repr__(self):
            return str(self)
        def __str__(self):
            return '<pos %d, product %d>' % (self.abs_position, self.product)
            
        def __lt__(self, other):
            return self.product > other.product
        def __eq__(self, other):
            if NoneType == type(other):
                return False
            return self.product == other.product
        def __ne__(self, other):
            if NoneType == type(other):
                return True
            return self.product != other.product
    
    
    @total_ordering
    class NodeWeightComparer(object):
        def __init__(self, node):
            self.node = node
        
        def __lt__(self, other):
            return self.node.product > other.node.product
        def __eq__(self, other):
            return self.node.product == other.node.product
        def __ne__(self, other):
            return self.node.product != other.node.product


solver = BNBSolver( MaxProductModel([10]) )
sln = solver.solve()
while sln:
    print "%d (%f)" % (sln.abs_position, sln.product)
    sln = sln.parent

print
solver.stat()

