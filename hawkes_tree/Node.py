class Node:
	def __init__(self, id, parent):
		self.id = id
		self.parent = parent
		self.children = []			#list of Node objects
		self.children_ids = set()	#set of ids only

	def add_child(self, child_id):
		if child_id not in self.children_ids:
			self.children.append(Node(child_id, self.id))

	def add_children(self, children_ids):
		for child_id in children_ids:
			self.add_child(child_id)
		
