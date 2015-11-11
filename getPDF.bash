#! /bin/bash
mkdir -p pdf

downloadPDF()
{
	 if [ ! -f pdf/$2 ]; then
		printf "( ° o°):"
		printf '\x1b[31m'
	    echo $2 "not found"
	    printf '\x1b[0m'
	    wget $1/$2 -O pdf/$2
	else
		printf "( ° o°):"
		printf '\x1b[32m'
		echo $2 "already there"
		printf '\x1b[0m'
	fi
}

# HPC
downloadPDF http://www.hpceurope.com/cat13HPC HPC-T4-2013-GearsAndSprockets-GB.pdf
# Michaud Chailly
downloadPDF http://www.michaud-chailly.fr/custom/docs/catalogue/dt catalogue-direct-transmission-version-6-1-juin-2008-pdf-19-mo-dt-lcat0.pdf
# SKF
downloadPDF http://www.skf.com/binary/68-121486 SKF-rolling-bearings-catalogue.pdf
# Timken
downloadPDF http://www.timken.com/fr-fr/products/Documents Timken-AP-Bearing-Catalog.pdf

