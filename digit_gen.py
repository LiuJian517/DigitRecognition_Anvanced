from PIL import Image
import numpy as np
import os
f = open("test.csv", 'r')
a = f.readline()


if not os.path.exists("./digits_gen"):
    os.mkdir("./digits_gen")

i = 0
for a in f:
    data = np.matrix(a)
    data = np.reshape(data, (28, 28))
    print(data.shape)
    im = Image.fromarray(data.astype(np.uint8))
    im.show()
    im.save("./digits_gen/%s.bmp" % i, "bmp")
    i += 1
    if i == 10:
        break