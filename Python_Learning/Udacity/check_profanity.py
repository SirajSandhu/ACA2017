import urllib

def check_file() : 
	# to read the text in the file	
	opened_file = open("/home/sirajsandhu/Desktop/ACA-CML/Python_Learning/Udacity/movie_quotes.txt")
	file_contents = opened_file.read()
	print(file_contents)
	opened_file.close()

	check_profanity(file_contents)

def check_profanity(text_to_check) : 
	online_src = urllib.urlopen("http://www.wdylike.appspot.com/?q=+text_to_check")
	output = online_src.read()
	print "the above text contains profanity : %s" %output
	online_src.close()
	
check_file()
