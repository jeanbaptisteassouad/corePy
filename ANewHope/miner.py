import re



def miner_lineIsNumeric(line):
	chars = list(line)
	cpt_numeric = 0
	cpt_other = 0
	for x in range(0,len(chars)):
		if chars[x].isdigit():
			cpt_numeric += 1
		elif chars[x] != " ":
			cpt_other += 1
		pass
	if cpt_numeric > cpt_other:
		return True
	else:
		return False

def miner_keepOnlyNumericLine(page):
	lines = re.split('\n',page)
	ans = ""
	for x in range(0,len(lines)):
		if miner_lineIsNumeric(lines[x]):
			ans += lines[x]
			ans += "\n"
		pass
	return ans

def miner_lineHaveThreeOrMoreColumns(line):
	list_elems = re.split(" {2,}",line);
	ans = 0
	for x in range(0,len(list_elems)):
		if list_elems[x] != '':
			ans += 1
			pass
		pass
	if ans >= 3:
		return True
	else:
		return False


def miner_keepOnlyThreeAndMoreColumnsLine(page):
	lines = re.split('\n',page)
	ans = ""
	for x in range(0,len(lines)):
		if miner_lineHaveThreeOrMoreColumns(lines[x]):
			ans += lines[x]
			ans += "\n"
		pass
	return ans



def main():
	f = open('Skf','r')
	content = f.read()
	f.close()
	pages = re.split('\f',content)



	number_page = 507
	lines = re.split('\n',pages[number_page])


	#######################
	# Formating for debug #
	#######################

	#2 espaces == separateurs de colonnes
	pages[number_page] = re.sub('(?P<data> {2,})',"  ",pages[number_page])

	#On enleve toutes les lignes vides
	pages[number_page] = re.sub('(?P<data>\n{1,})',"\n",pages[number_page])

	##############
	# Processing #
	##############
	#On garde les lignes ou il y au moins la moitier de chiffres
	pages[number_page] = miner_keepOnlyNumericLine(pages[number_page])

	#On garde les lignes avec au moins 3 colonnes
	pages[number_page] = miner_keepOnlyThreeAndMoreColumnsLine(pages[number_page])

	print pages[number_page]


def get_the_good_yosh(line_of_table,yosh):
	score = []
	for x in range(0,len(yosh)):
		score.append(0)
		for y in range(0,len(yosh[x])):
			score[x] += len(re.findall(" "+re.escape(yosh[x][y])+" "," "+line_of_table+" "))

	indiceMax = 0
	maxScore = -1

	for x in range(0,len(score)):
		if score[x] > maxScore:
			maxScore = score[x]
			indiceMax = x
			pass
		pass
	return indiceMax


def main2():
	f = open('Hpc.html','r')
	content = f.read()
	f.close()

	inside_doc_flag = re.findall('<doc>(.*)</doc>',content,re.DOTALL)
	pages_bbox = re.findall('<page.*?</page>',inside_doc_flag[0],re.DOTALL)

	f = open('Hpc','r')
	content = f.read()
	f.close()
	pages_layout = re.split('\f',content)


	for page_iterator in range(78,79):
		lines_layout = re.split('\n',pages_layout[page_iterator])
		tmp_page_dimension = re.findall("<page width=\"([^\"]*)\" height=\"([^\"]*)\">",pages_bbox[page_iterator])
		page_width = tmp_page_dimension[0][0]
		page_height = tmp_page_dimension[0][1]
		#On garde les lignes ou il y au moins la moitier de chiffres
		table = miner_keepOnlyNumericLine(pages_layout[page_iterator])

		#On garde les lignes avec au moins 3 colonnes
		table = miner_keepOnlyThreeAndMoreColumnsLine(table)



		yMin = re.findall("yMin=\"([^\"]*?)\"",pages_bbox[page_iterator])


		yMin_tmp = []
		for p in range(0,len(yMin)):
			isthere = False
			for j in range(0,len(yMin_tmp)):
				if yMin_tmp[j] == yMin[p]:
					isthere = True
					break
			if isthere == False:
				yMin_tmp.append(yMin[p])

		yMin_tmp.sort()
		yMin = list(yMin_tmp)

		lines_bbox = []
		for p in range(0,len(yMin)):
			lines_bbox.append( re.findall("<word[^>]*yMin=\""+str(yMin[p])+"\"[^>]*>([^<]*)</word>",pages_bbox[page_iterator]) )
			pass

		lines_of_table = re.split("\n",table)

		print pages_bbox[page_iterator]
		for p in range(0,len(lines_of_table)):
			print lines_of_table[p]
			print yMin[get_the_good_yosh(lines_of_table[p],lines_bbox)], lines_bbox[get_the_good_yosh(lines_of_table[p],lines_bbox)]
			print re.findall("<word xMin=\"([^>]*)\" yMin=\""+str(yMin[get_the_good_yosh(lines_of_table[p],lines_bbox)])+"\" xMax=\"([^>]*)\" yMax=\"([^>]*)\">[^<]*</word>",pages_bbox[page_iterator])
			exit(1)

		# for line_iterator in range(0,len(lines_layout)):
		# 	if lines_layout[line_iterator] != "":
		# 		words = re.split(' *',lines_layout[line_iterator])
		# 		tmp_words = []
		# 		for word_iterator in range(0,len(words)):
		# 			if words[word_iterator] != '':
		# 				tmp_words.append(words[word_iterator])
		# 		words = list(tmp_words)
		# 		print words
		# 		if len(words) >= 2:
		# 			print "<word.*?>"+words[0]+"<.*?>"+words[len(words)-1]+"<.*?word>"
		# 			current_line_bbox = re.findall("<word.*?>"+words[0]+"<.*?>"+words[len(words)-1]+"<.*?word>",pages_bbox[page_iterator],re.DOTALL)
		# 		if len(words) == 1:
		# 			current_line_bbox = re.findall("<word.*?>"+words[0]+"<.*?word>",pages_bbox[page_iterator],re.DOTALL)
		# 		if current_line_bbox != []:
		# 			current_line_bbox = current_line_bbox[0]
		# 			print current_line_bbox
		# 			xMin = min(re.findall("xMin=\"([^\"]*?)\"",current_line_bbox))
		# 			yMin = min(re.findall("yMin=\"([^\"]*?)\"",current_line_bbox))
		# 			xMax = max(re.findall("xMax=\"([^\"]*?)\"",current_line_bbox))
		# 			yMax = max(re.findall("yMax=\"([^\"]*?)\"",current_line_bbox))
		# 			print yMin


		pass
	# pass
	# for x in range(161600,161673):
	# 	print pages[x]
	# 	pass
	# pass


if __name__ == '__main__':
	main2()

