import numpy as np
filepath = 'D:/my bad/Suspicious object detection/data/CSV/'
filetype = '.csv'
filenames = []
trainfile = []
testfile = []
for j in ["0", "1M"]:  # "1S", "2S"
    for name in ['zb','tk']:
        for i in [i for i in range(0, 30)]:
            fn = filepath + name+"-2.5-M/" + name+"-"+ str(j) + "-" + str(i) + filetype
            filenames += [fn]
        np.random.shuffle(filenames)
        trainfile += filenames[:20]
        testfile += filenames[20:]
        filenames = []
trainfile = np.array(trainfile)#20*2
testfile = np.array(testfile)#10*2

print(trainfile)
print(testfile)