import numpy as np

LEN_FEATURE = 500

class Predictor(object):
	"""docstring for Predictor"""
	def __init__(self):
		super(Predictor, self).__init__()
		self.datas = np.zeros( (0,LEN_FEATURE) )
		self.classes = np.zeros( (0,2) )



