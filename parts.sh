#!/usr/bin/env sh

set e-

# destination folder name
DEST=140
# text file source
TEXT_FILE=200.txt

# creates destination folder
mkdir "intervals/$DEST"

# open text file to get the file name of the images
cat "$TEXT_FILE" | while read line; do 				
	filename="$(echo $line | cut -d' ' -f1 )"
	letter="$(echo $filename | cut -d'_' -f1)"

	# source of the images
	SRC="dataset/$letter"

	# copies the image file to the destination folder
	cp "$SRC/$filename" "intervals/$DEST"

	echo "$filename"
done

echo "----- DONE -----"