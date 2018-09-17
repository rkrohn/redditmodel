
import file_utils
import tarfile
import os

#load all reddit and exogenous data, convert to fun python structures, and save as pickles
#code = {cyber, crypto, cve}, indicating reddit data to load
def load_model_data(code):

	if code == "cyber":

		#extract comment files from tar if not already
		if os.path.isdir("../2018DecCP/Reddit/Cyber/UNPACK_Tng_an_RC_Cyber_sent") == False:
			tar = tarfile.open("../2018DecCP/Reddit/Cyber/Tng_an_RC_Cyber_sent.tar")
			tar.extractall("../2018DecCP/Reddit/Cyber/UNPACK_Tng_an_RC_Cyber_sent")
			tar.close()
		#load each comment file
		for filename in sorted(os.listdir("../2018DecCP/Reddit/Cyber/UNPACK_Tng_an_RC_Cyber_sent")):
			data = file_utils.load_zipped_multi_json("../2018DecCP/Reddit/Cyber/UNPACK_Tng_an_RC_Cyber_sent/" + filename)
			print(len(data))
			print(data[0])
			break


	elif code == "crypto":
		print("no data for you")

	else:		#cve
		print("no data for you")






load_model_data("cyber")