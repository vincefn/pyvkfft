

imax = 5
a = 31
a = (1<<0 | 1<<3)


for i in range(imax):
    
    if not a&(1<<i):
        i += 1
        continue

    print(i)
