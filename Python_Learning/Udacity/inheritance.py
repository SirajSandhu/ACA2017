class Parent() :	
	def __init__(self, last_name, dukedom) :
		print("Parent Constructor Called")
		self.last_name = last_name
		self.dukedom = dukedom

class Child(Parent) : 
	def __init__(self, first_name, last_name, dukedom, succession_order) : 
		print("Child Constructor Called")
		Parent.__init__(self, last_name, dukedom)
		
		self.first_name = first_name
		self.succession_order = succession_order

adam = Child("Adam", "Rathbourne", "Whitehaven", "First Born")

#print "%s, %s of the house of %s, shall be the heir to the dukedom of %s." %(adam.first_name, adam.succession_order, adam.last_name, 											adam.dukedom)
