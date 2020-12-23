import numpy as np
c = np.arange(10)
c[5]=9
print(c)
pred_train_vot=np.arange(4000/100)
print(pred_train_vot)
if(pred_train_vot[1]==1):print("sss")
for b in range(0, 5):
    print(b)

def get_max(shuzu):
    s=[0,0]
    for i in range(0,10):
        if (shuzu[i]==0):s[0]=s[0]+1
        else:s[1]=s[1]+1
    print(s)
    if (s[0] > s[1]): return 0
    if (s[0] < s[1]): return 1
    if (s[0] == s[1]): return 2
ss=[0,0,0,0,1,1,0,0,1,1]

i=get_max(ss)
print(i)
