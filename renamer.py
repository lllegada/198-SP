import os
import glob
n = 0
def rename_tool(folder_path,imtype,n):
	n = n
	files=[]
	
	print("The following are files found in " + folder_path + " with ." + imtype + " format:")
	# print(glob.glob(folder_path+'/'+'*/*.' + imtype))
	print("Renaming files...")
	for file in glob.glob(folder_path+'/'+'*/*.' + imtype):
		# n = n + 1
		files.append(file.replace('\\','/'))
		# print(x)
		# print(x.split('_')[2])
		# print(files)
	for x in files:
		n = n + 1
		a = str(n)
		old_name = (x.split('/')[2])
		new_name = (folder_path + '/' + x.split('/')[1] + '/' + x.split('/')[1] + '_' + a + "." + x.split('.')[1])
		# change_ext = new_name.split(.)[1]
		# print(change_ext)
		print("Old name: ",x)
		print("New name: ",new_name)
		# os.rename(x,new_name)
		pre, ext = os.path.splitext(new_name)
		os.replace(x, pre + ".jpg")
		print("pre: ",pre)
		print("ext: ",ext)


rename_tool("arranged","jpg",n)

# rename_tool("Dataset2","png",n)