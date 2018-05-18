import sys
import os 
import math

#Directory path of folder with all files is first argument 
#Output chunk as tab delimited row, first element of row is file name: chromosome name: start position of chunk on chromosome, second element of row is sparse or populated, depending if sum of row is greater than 75
#If multiple values in one bucket, takes max

BUCKET_SIZE = 25
CHUNK_SIZE = 25000
SPARSE_LIMIT = 75

def main():
	path = sys.argv[1]
	dirs = os.listdir(path)
	output = open("data_output.txt", "w")
	for file in dirs:
		processFile(file, output, path)
	output.close()

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
		if chrom == prev_chrom and len(chunk_array) >= CHUNK_SIZE/BUCKET_SIZE: #new chunk 
			chunk_val = [float(val) for val in chunk_array]
			flag = "populated"
			if sum(chunk_val) < SPARSE_LIMIT:
				flag = "sparse"
			describe = str(file) + ":" + chrom + ":" + str(start_chunk)
			output.write(describe + "\t")
			output.write(flag + "\t")
			output.write('\t'.join(chunk_array) + "\n")
			start_chunk = start 
			chunk_array = []
			last_index = 0
		if chrom != prev_chrom: #new chromosome 
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
