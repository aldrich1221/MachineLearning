class Keywords:
	def __init__(self, classId, ori_keyword, extend_keyword):
		self.classId = classId
		self.ori_keyword = ori_keyword
		self.extend_keyword = extend_keyword
	def __repr__(self):
		return repr((self.classId, self.ori_keyword, self.extend_keyword))