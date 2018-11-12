class Donor(object):
	"""docstring for ClassName"""
	def __init__(self, funds):
		self.funds = funds
	
	def makeContribution(self, contribution):
		if(self.funds >= contribution):
			self.funds -= contribution 
			return 1
		else:
			return 0