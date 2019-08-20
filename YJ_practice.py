num = [2,7,11,15]
target = 9


flag = False
for i in range(len(num)):
    for j in range(len(num)):
        if (num[i]+num[j]==target):
            if flag == False:
                print (i,j)
                flag = True