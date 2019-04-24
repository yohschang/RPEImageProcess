
path = r"C:\Users\BT\Desktop\X_profile.txt"
with open(path,'r') as file:
    a = file.readlines()
    profile = dict()
    for line in a:
        index = int(line.split("\t")[0])
        phase = float(line.split("\t")[1].strip("\n"))
        profile.update({index:phase})

um_per_pixel = 5.5
mag = float(input("Mag:"))
index0 = int(input("index0:"))
index100 = int(input("index100:"))

phase100 = profile[index100]
phase0 = profile[index0]

phase10 = (phase100 - phase0)*0.1 + phase0
phase90 = (phase100 - phase0)*0.9 + phase0

for i in range(index0, index100+1):
    if profile[i] >= phase10:
        low10 = i-1
        up10 = i
        index10 = round(low10 + 1*(abs(phase10-profile[i-1])/abs(profile[i]-profile[i-1])),3)
        break
for i in range(index0, index100+1):
    if profile[i] >= phase90:
        low90 = i-1
        up90 = i
        index90 = round(low90 + 1*(abs(phase90-profile[i-1])/abs(profile[i]-profile[i-1])),3)
        break

for i in range(index0, index100+1):
    print(i,profile[i])
print('index10', str(index10))
print('index90', str(index90))
resolution = round(((index90 - index10)*um_per_pixel)/mag,3)
print("resolution is: ",str(resolution))

a = input("halt!")