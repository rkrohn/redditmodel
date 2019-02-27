import zss
from zss import Node

#basic example first - define two trees using library's default Node structure, compute distance
A = (
    Node("f")
        .addkid(Node("a")
            .addkid(Node("h"))
            .addkid(Node("c")
                .addkid(Node("l"))))
        .addkid(Node("e"))
    )
B = (
    Node("f")
        .addkid(Node("a")
            .addkid(Node("d"))
            .addkid(Node("c")
                .addkid(Node("b"))))
        .addkid(Node("e"))
    )

#answer should be 2 - rename A's h -> d and A's l -> b
dist = zss.simple_distance(A, B)
print("tree edit distance for A,B =", dist)	


#custom tree format example - can define a node class for building trees, and functions for node distance

#first, define a string distance function - string edit distance if it exists, otherwise 0/1 on node names/labels
try:
    from editdist import distance as strdist
except ImportError:
    def strdist(a, b):
        if a == b:
            return 0
        else:
            return 1

#alternative string distance function, if we want node name edits to have greater weight
def weird_dist(A, B):
    return 10*strdist(A, B)

#custom node class, with required methods to get list of children and get node label
class WeirdNode(object):
    def __init__(self, label):
        self.my_label = label
        self.my_children = list()

    @staticmethod
    def get_children(node):
        return node.my_children

    @staticmethod
    def get_label(node):
        return node.my_label

    def addkid(self, node, before=False):
        if before:  self.my_children.insert(0, node)
        else:   self.my_children.append(node)
        return self
#end WeirdNode

#define two test trees - same as in example above
A = (
WeirdNode("f")
    .addkid(WeirdNode("d")
    .addkid(WeirdNode("a"))
    .addkid(WeirdNode("c")
        .addkid(WeirdNode("b"))
    )
    )
    .addkid(WeirdNode("e"))
)
B = (
WeirdNode("f")
    .addkid(WeirdNode("c")
    .addkid(WeirdNode("d")
        .addkid(WeirdNode("a"))
        .addkid(WeirdNode("b"))
    )
    )
    .addkid(WeirdNode("e"))
)

#compute distance between A and B, with custom node methods and weird distance method
#this time result should be 20, since the custom node distance is 10 * string edit distance
dist = zss.simple_distance(A, B, WeirdNode.get_children, WeirdNode.get_label, weird_dist)

print("tree edit distance for A,B weird_dist =", dist)