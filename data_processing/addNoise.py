import random

#changes NUM_TO_ONE values to 1.0, changes a random number (from 1/20 to 1/10 of the number of values that aren't 0.0) of values to 0.0

NUM_TO_ONE = 5

def main():
	random.seed(20)
	output = open("noisy_output.txt", "w")
	f = open("data_output.txt", "r")
	for line in f:
		split_line = line.split()
		length = len(split_line)
		pop_indices = [i for i in range(3, len(split_line)) if split_line[i] != "0.0"]
		num_change = random.randint(int(len(pop_indices)/20), int(len(pop_indices)/10))
		change_indices = random.sample(pop_indices, num_change)
		for index in change_indices:
			split_line[index] = "0.0"
		for _ in range(NUM_TO_ONE):
			index = random.randint(3, length - 1)
			split_line[index] = "1.0"
		split_line[1] = split_line[0] + split_line[1]
		output.write('\t'.join(split_line[1:]) + "\n")
	f.close()
	output.close()


if __name__ == '__main__':
	main()
