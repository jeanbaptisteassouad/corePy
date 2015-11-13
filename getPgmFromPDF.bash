#! /bin/bash

convertOnePdf()
{
	mkdir -p pgm/$1
	if [ ! -d pgm/$1/$2 ]; then
		printf "( ° o°):"
		printf '\x1b[31m'
	    echo $2 "not found"
	    printf '\x1b[0m'
	    mkdir -p pgm/$1/$2
	    pdftoppm -rx $1 -ry $1 -gray pdf/$2.pdf pgm/$1/$2/$2
	else
		printf "( ° o°):"
		printf '\x1b[32m'
		echo $2 "already there"
		printf '\x1b[0m'
	fi
}

convertAllPdf()
{
	printf "( ° o°):"
	printf '\x1b[1m'
	echo "Converting to" $1 "ppi"
	printf '\x1b[0m'
	# List all the pdf that should be convert
	convertOnePdf $1 HPC-T4-2013-GearsAndSprockets-GB
	#convertOnePdf $1 catalogue-direct-transmission-version-6-1-juin-2008-pdf-19-mo-dt-lcat0
	#convertOnePdf $1 SKF-rolling-bearings-catalogue
	#convertOnePdf $1 Timken-AP-Bearing-Catalog
}

# List all the ppi needed
convertAllPdf 50
