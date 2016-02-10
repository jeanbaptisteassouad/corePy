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


	for page_iterator in range(0,1):
		lines_layout = re.split('\n',pages_layout[page_iterator])
		tmp_page_dimension = re.findall("<page width=\"([^\"]*)\" height=\"([^\"]*)\">",pages_bbox[page_iterator])
		page_width = tmp_page_dimension[0][0]
		page_height = tmp_page_dimension[0][1]
		print pages_layout[page_iterator]

		for line_iterator in range(0,len(lines_layout)):
			if lines_layout[line_iterator] != "":
				words = re.split(' *',lines_layout[line_iterator])
				tmp_words = []
				for word_iterator in range(0,len(words)):
					if words[word_iterator] != '':
						tmp_words.append(words[word_iterator])
				words = list(tmp_words)
				xMin = 0
				yMin = 0
				xMax = 0
				yMax = 0
				if len(words) >= 2:
					current_line_bbox = re.findall("<word[^<>]*>"+words[0]+"<.*>"+words[len(words)-1]+"</word>",pages_bbox[page_iterator],re.DOTALL)
				if len(words) == 1:
					current_line_bbox = re.findall("<word[^<>]*>"+words[0]+"</word>",pages_bbox[page_iterator],re.DOTALL)
				if current_line_bbox != []:
					current_line_bbox = current_line_bbox[0]
					xMin = float(min(re.findall("xMin=\"([^\"]*?)\"",current_line_bbox)))/float(page_width)
					yMin = float(min(re.findall("yMin=\"([^\"]*?)\"",current_line_bbox)))/float(page_height)
					xMax = float(max(re.findall("xMax=\"([^\"]*?)\"",current_line_bbox)))/float(page_width)
					yMax = float(max(re.findall("yMax=\"([^\"]*?)\"",current_line_bbox)))/float(page_height)
				print line_iterator,xMin,yMin,xMax,yMax


		pass
	# pass
	# for x in range(161600,161673):
	# 	print pages[x]
	# 	pass
	# pass


if __name__ == '__main__':
	main2()

