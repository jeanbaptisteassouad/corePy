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


def main():
	f = open('Hpc','r')
	content = f.read()
	f.close()
	pages = re.split('\f',content)



	number_page = 99
	lines = re.split('\n',pages[number_page])

	#2 espaces == separateurs de colonnes
	pages[number_page] = re.sub('(?P<data> {2,})',"  ",pages[number_page])

	#On enleve toutes les lignes vides
	pages[number_page] = re.sub('(?P<data>\n{1,})',"\n",pages[number_page])

	# print pages[number_page]

	print re.sub('(?P<data> *([^ \n]+ +){10})',"\x1b[31m\g<data>\x1b[0m",pages[number_page])

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

