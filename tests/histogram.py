import cv2 as cv
from pathlib import Path
from matplotlib import pyplot as plt

def resize_image(image, image_name, procent):
    [height, width] = [image.shape[0],image.shape[1]]
    [height, width] = [procent*height, procent*width]
    cv.namedWindow(image_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(image_name, int(width), int(height))
    cv.imshow(image_name, image)


picPath = Path.cwd().parent.as_posix()+'/materials/GR-type_0_2.jpg'

img = cv.imread(picPath)

color = ('b','g','r')
b,g,r = cv.split(img)
for i,col in enumerate(color):
    histr = cv.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()

resize_image(b, 'blue channel', 0.4)
resize_image(r, 'red channel', 0.4)
resize_image(g, 'green channel', 0.4)
cv.waitKey(0)