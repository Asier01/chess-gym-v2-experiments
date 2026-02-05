import matplotlib.pyplot as plt
import numpy

episodeLength = []
averageRewards = []
fig, (rw, leng) = plt.subplots(2,1)

#Get arrays of data out of log file
filename = input("Input log data to plot - ")
with open(filename) as file:
	for line in file:
		num = ""
		for char in line:
			if char==",":
				break
			num += char
		averageRewards.append(float(num))
		if "," not in line[-3:]:
			episodeLength.append(int(line[-3:]))
		else:
			episodeLength.append(int(line[-2]))


#print(len(averageRewards))
rw.plot(averageRewards, linewidth=0.05)
rw.set_title("Rewards")

leng.plot(episodeLength, linewidth=0.25)
leng.set_title("Episode length")

plt.tight_layout()
plt.show()
