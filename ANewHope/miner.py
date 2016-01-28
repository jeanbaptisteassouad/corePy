import re


def string_to_regex(string):
	ans = ''
	last_regex = ''
	for i in range(0,len(string)):
		if re.search('[0-9,]',string[i]) != None:
			if last_regex == '[0-9,]+':
				pass
			else:
				ans += '[0-9,]+'
				last_regex = '[0-9,]+'
		elif re.search('[A-Z]',string[i]) != None:
			if last_regex == '[A-Z]+':
				pass
			else:
				ans += '[A-Z]+'
				last_regex = '[A-Z]+'
		elif re.search('[a-z]',string[i]) != None:
			if last_regex == '[a-z]+':
				pass
			else:
				ans += '[a-z]+'
				last_regex = '[a-z]+'
		else:
			ans += string[i]
			last_regex = string[i]
	return ans;

def gen_regex(list_words):
	ans = []
	for x in range(0,len(list_words)):
		ans.append( string_to_regex(list_words[x]) )
	return ans


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



	number_page = 760
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

	# list_regex = []
	# for x in range(0,len(lines)):
	# 	tmp = re.split(' {2,}',lines[x])
	# 	# print tmp
	# 	list_regex.append( gen_regex(tmp) )
	# 	# for y in range(0,len(lines[x])):
	# 	# 	print lines[x][y], re.search('[0-9]',lines[x][y]) != None, re.search('[A-B]',lines[x][y]) != None, re.search('[a-z]',lines[x][y]) != None
	# 	# 	pass
	# 	pass

	# for x in range(0,len(list_regex)):
	# 	print list_regex[x]
	# 	pass

	# nb_spaces_per_lines = []
	# for x in range(0,len(lines)):
	# 	nb_spaces_per_lines.append( len(re.findall('[^ ] {2,}',lines[x])) )


	# for x in range(0,len(nb_spaces_per_lines)-1):
	# 	if nb_spaces_per_lines[x] != nb_spaces_per_lines[x+1]:
	# 		nb_spaces_per_lines[x] = 0
	# 	else:
	# 		if nb_spaces_per_lines[x] != 0:
	# 			print x, ":", nb_spaces_per_lines[x]


	pass

if __name__ == '__main__':
	main()

