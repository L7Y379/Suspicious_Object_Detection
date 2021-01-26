import numpy as np
def file_array():
    filepath = 'D:/my bad/Suspicious object detection/data/CSV/'
    filetype = '.csv'
    filenames = []
    trainfile = []
    testfile = []
    for j in ["0", "1M","2M"]:  # "1S", "2S"
        for i in [i for i in range(0, 30)]:
            fn = filepath + "zb-2.5-M/" + "zb-" + str(j) + "-" + str(i) + filetype
            filenames += [fn]
        np.random.shuffle(filenames)
        if (j == "0"):
            trainfile += filenames[:20]
            testfile += filenames[20:]
        if(j=="1M"):
            trainfile += filenames[:10]
            testfile += filenames[25:]
        if (j == "2M"):
            trainfile += filenames[:10]
            testfile += filenames[25:]
        filenames = []
    trainfile = np.array(trainfile)#20*2
    testfile = np.array(testfile)#10*2
    #print(testfile);
    return trainfile, testfile

a,b=file_array()
print(a.shape)
print(a)
print(b.shape)
print(b)