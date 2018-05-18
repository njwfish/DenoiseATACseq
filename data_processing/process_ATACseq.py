"""
Takes in a single argument of a directory path of a folder with bedgraph files.
Reads all files in the folder and writesto data_output.txt the preprocessing for all the files.
Outputs a representation of a continuous 25000 base pair "chunk size" as a single line in the file.
The line in the file is tab delimited, with the first value being "file name: chromosome name: start position on chromsome".
The second value is either populated or sparse, depending on whether the sum of all read counts for that line exceeds SPARSE_LIMIT.
The rest of the values represent the maximum read count within a 25 base pair "chunk size" region of the overall 25000 base pairs. 
Therefore, there are 1000 values for each line of the output file. 
"""
import sys
import os 
import math

BUCKET_SIZE = 25
CHUNK_SIZE = 25000
SPARSE_LIMIT = 75

"""
Reads in the names of the files in the directory indicated by path and performs processFile on each.
"""
def main():
	path = sys.argv[1]
	dirs = os.listdir(path)
	output = open("data_output.txt", "w")
	for file in dirs:
		processFile(file, output, path)
	output.close()

"""
file: name of file being read in to process
output: name of output file
path: directory path of folder with all bedgraph files 

Opens file and outputs processed values for continuous 25000 base pair lengths, by binning maximum signal values for 25 bp intervals.
"""
def processFile(file, output, path):
	f = open(str(path) + "/" + str(file))
	prev_chrom = ""
	chunk_array = []
	for line in f:
		split_line = line.split()
		chrom = split_line[0]
		start = int(split_line[1])
		end = int(split_line[2])
		value = float(split_line[3])
		last_index = 0
		
		#start a new chunk
		if chrom == prev_chrom and len(chunk_array) >= CHUNK_SIZE/BUCKET_SIZE: 
			#output old chunk
			chunk_val = [float(val) for val in chunk_array]
			flag = "populated"
			if sum(chunk_val) < SPARSE_LIMIT:
				flag = "sparse"
			describe = str(file) + ":" + chrom + ":" + str(start_chunk)
			output.write(describe + "\t")
			output.write(flag + "\t")
			output.write('\t'.join(chunk_array) + "\n")
			
			#re-initialize values for new chunk
			start_chunk = start 
			chunk_array = []
			last_index = 0

		#start a new chunk on a new chromosome 
		if chrom != prev_chrom:
			prev_chrom = chrom
			start_chunk = start
			last_index = 0
			chunk_array = []
	
		start_index_for_line = int(float(start - start_chunk)/BUCKET_SIZE)
		
		if start_index_for_line == last_index:
			if len(chunk_array) == 0:
				chunk_array.append(str(value))
			else:
				newValue = str(max(float(chunk_array[last_index]), value))
				chunk_array[last_index] = newValue

		num_append = int(math.ceil(float(end - start)/BUCKET_SIZE) - 1)
		for _ in range(num_append):
			if len(chunk_array) < CHUNK_SIZE/BUCKET_SIZE:
				chunk_array.append(str(value))

		last_index = len(chunk_array) - 1
		
	f.close()



if __name__ == '__main__':
	main()
