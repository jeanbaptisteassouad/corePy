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


if __name__ == '__main__':
	main()

