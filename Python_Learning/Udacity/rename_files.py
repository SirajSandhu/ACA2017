import os

def rename_files() : 
	#get the file names
	file_list=os.listdir("/home/sirajsandhu/Desktop/prank")
	print(file_list)

	saved_path=os.getcwd()
	os.chdir("/home/sirajsandhu/Desktop/prank")

	#for each file rename file
	for file_name in file_list :
		print "renaming %s to " %file_name
		os.rename(file_name, file_name.translate(None,"0123456789"))
		print "%s\n" %file_name.translate(None,"0123456789")

	os.chdir(saved_path)


rename_files()
