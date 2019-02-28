import zss
from zss import Node


#distance function for structural-only edit distance - assume all node labels are equal
def struct_only_dist(a, b):
	return 0   
#end struct_only_dist

#distance function taking node labels into account - distance is 0 if labels are exactly equal, 1 otherwise
def label_equality_dist(a, b):
	if a.get_label() == b.get_label():
		return 0
	else:
		return 1
#end label_equality_dist

#distance function that ignores node labels, but takes node time into account
#if node times are within 30 minutes of each other, distance is 0
#otherwise, distance is 1
def time_dist(a, b):
	if abs(a.get_time() - b.get_time()) <= 30:
		return 0
	else:
		return 1
#end time_dist


#basic insert and removal cost functions - either operation is a cost of 1
def remove_cost(a):
	return 1
def insert_cost(a):
	return 1


#custom node class for comment trees, with required method to get list of children
class CommentNode(object):
	def __init__(self, label, time):
		self.label = label
		self.children = list()
		self.time = time

	def get_children(self):
		return self.children

	def get_label(self):
		return self.label

	def get_time(self):
		return self.time

    #append child to end of list of children
	def append_child(self, node):
		self.children.append(node)
		return self     #return self so we can chain add operations

    #insert a child to the front of the list of children
	def prepend_child(self, node):
		self.children.insert(0, node)
		return self     #return self so we can chain add operations
#end CommentNode


#given a CommentNode tree root, print the tree comment times
#uses DFS to print the tree structure
def print_tree(root):
	stack = [(root, 0)]     #node processing stack, node and level

	while len(stack) != 0:
		curr, level = stack.pop()  #get last node added to stack
		print("    " * level + "%.3f" % curr.get_time())   #print this comment time

		#append children in reverse time order so final output is sorted
		stack.extend([(child, level+1) for child in curr.get_children()][::-1])   
	print("") 
#end print_tree


#define two test trees - same as in example above
A = (
	CommentNode("f", 0)
		.append_child(CommentNode("a", 32)
			.append_child(CommentNode("h", 37))
			.append_child(CommentNode("c", 65)
				.append_child(CommentNode("l", 123))))
		.append_child(CommentNode("e", 50))
	)
B = (
	CommentNode("f", 0)
		.append_child(CommentNode("a", 30)
			.append_child(CommentNode("d", 42))
			.append_child(CommentNode("c", 60)
				.append_child(CommentNode("b", 80))))
		.append_child(CommentNode("e", 47))
		.append_child(CommentNode("g", 126))
	)

print_tree(A)
print_tree(B)

#compute distance between A and B, with custom node methods and structure-only method
struct_dist = zss.distance(A, B, CommentNode.get_children, insert_cost, remove_cost, struct_only_dist)
print("tree edit distance for A,B struct_only_dist =", struct_dist)
#result should be 1, since B has one additional comment than A

#get distance taking node label equality into account
label_dist = zss.distance(A, B, CommentNode.get_children, insert_cost, remove_cost, label_equality_dist)
print("tree edit distance for A,B label_equality_dist =", label_dist)
#result should be 3, 1 for B's additional node and 2 from rename operations h->d and l->b

#get distance taking node times into account, but ignoring labels
time_dist = zss.distance(A, B, CommentNode.get_children, insert_cost, remove_cost, time_dist)
print("tree edit distance for A,B time_dist =", time_dist)
#result should be 2, 1 for B's additional node and 2 from the time difference between l and b
