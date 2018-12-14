import struct
import imghdr
import sys, os
# Plots a histogram of widhts and heights of all images
def test_jpeg(h, f):
    # SOI APP2 + ICC_PROFILE
    if h[0:4] == '\xff\xd8\xff\xe2' and h[6:17] == b'ICC_PROFILE':
        print "A"
        return 'jpeg'
    # SOI APP14 + Adobe
    if h[0:4] == '\xff\xd8\xff\xee' and h[6:11] == b'Adobe':
        return 'jpeg'
    # SOI DQT
    if h[0:4] == '\xff\xd8\xff\xdb':
        return 'jpeg'
imghdr.tests.append(test_jpeg)

def get_image_size(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            return
        what = imghdr.what(None, head)
        if what == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif what == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif what == 'jpeg':
            try:
                fhandle.seek(0) # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf or ftype in (0xc4, 0xc8, 0xcc):
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception: #IGNORE:W0703
                return
        else:
            return
        return width, height


tup = []
for path, subdirs, files in os.walk('/home/nct01/nct01003/.keras/datasets/Images/'):
        if (path != '/home/nct01/nct01003/.keras/datasets/Images/'):
                for name in files:
			res = get_image_size(path+'/'+name)
			if res is not None:
				tup.append(res)
w = [x[0] for x in tup]
h = [x[1] for x in tup]

##Store Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#Width and Height plot
plt.hist(w, bins=100, range=(0, 1000))
plt.title('Width')
plt.ylabel('pixels')
plt.xlabel('value')
plt.legend(['width'], loc='upper left')
plt.savefig('width.png')
plt.close()
plt.hist(h, bins=100, range=(0, 1000))
plt.title('Height')
plt.ylabel('pixels')
plt.xlabel('value')
plt.legend(['height'], loc='upper left')
plt.savefig('height.png')
