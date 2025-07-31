v = "welcome To ascediun"

s = ''.join(v.split(' '))

count = {}
for dig in s:
    if dig in count:
        count[dig] +=1
        if count[dig] == 2:
            print(f"{dig}")
    else:
        count[dig] =0