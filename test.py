import numpy as np
a=np.array([[1,2,3,4,5],[1,2,5,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]])
b=np.array([[1],[5],[5],[4],[3]])
c=np.hstack((a, b))
print(c)
#print(a)
# c = []
# c.append([1,2,3,4])
# c.append([5,6,7,8])
# c.append(9)
# print(c)
# latent_fake=a[-5]
# print(latent_fake)
# print(latent_fake)
#latent_fake=[]
# k,j=0
# for i in range(0,5):
#     #latent_fake.append(a[i])
#     latent_fake=np.vstack((latent_fake, a[i]))
#     print(latent_fake)