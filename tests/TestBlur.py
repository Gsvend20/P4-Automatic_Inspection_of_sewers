import cv2 as cv
import pathlib
from Functions import Funcs as func
import materials.datafile as Vas
import pathlib

# Q for exit
# E for Blur Down
# R for Blur Up
# D for EgdeKernel Down
# f for EgdeKernel Up

Path = pathlib.Path().cwd().parent.as_posix()
img1 = cv.imread(r'C:\Users\Glenn\Desktop\UNI\P4/bgr_branch_3.png', cv.COLOR_BGR2HSV)
img2 = cv.imread(r'C:\Users\Glenn\Desktop\UNI\P4/bgr_branch_64.png', cv.COLOR_BGR2HSV)
img3 = cv.imread(r'C:\Users\Glenn\Desktop\UNI\P4/bgr_branch_108.png', cv.COLOR_BGR2HSV)
img4 = cv.imread(r'C:\Users\Glenn\Desktop\UNI\P4/bgr_branch_130.png', cv.COLOR_BGR2HSV)
Vid1 = 'sewer recordings/Training data/Branching pipe/1_bgr.avi'
input_vid = func.GetVid(Vid1)
gray = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)
img = gray


Gaus_img = cv.GaussianBlur(img, (Vas.Blur, Vas.Blur), cv.BORDER_DEFAULT)
Median_img = cv.medianBlur(img, Vas.Blur, cv.BORDER_DEFAULT)
Blur = cv.blur(img, (Vas.Blur, Vas.Blur), cv.BORDER_DEFAULT)

egdeInput = func.Sobel(img)
egdeGaus = func.Sobel(Gaus_img)
egdeMedian = func.Sobel(Median_img)
egdeBlur = func.Sobel(Blur)
# Mean_img = cv.mean(img, Vas.Blur)
Running = True
# Printing images!
while Running:

    Gaus_img = cv.GaussianBlur(img, (Vas.Blur, Vas.Blur), cv.BORDER_DEFAULT)
    Median_img = cv.medianBlur(img, Vas.Blur, cv.BORDER_DEFAULT)
    Blur = cv.blur(img, (Vas.Blur, Vas.Blur), cv.BORDER_DEFAULT)

    egdeInput = func.Sobel(img)
    egdeGaus = func.Sobel(Gaus_img)
    egdeMedian = func.Sobel(Median_img)
    egdeBlur = func.Sobel(Blur)

    Vas.PicNumb1 = 0
    Vas.PicNumb2 = 0
    func.PrintPic('input Img', img)
    func.PrintPic('Blur', Blur)
    func.PrintPic('Gaussian Blur', Gaus_img)
    func.PrintPic('Median Blur', Median_img)
    func.PrintPic('input Img Egde', egdeInput)
    func.PrintPic('Blur Egde', egdeBlur)
    func.PrintPic('Gaussian Blur Egde', egdeGaus)
    func.PrintPic('Median Blur Egde', egdeMedian)
    while True:
        if cv.waitKey(1) == ord('e') and Vas.Blur >= 2:
            Vas.Blur -= 2
            break
        elif cv.waitKey(1) == ord('r'):
            Vas.Blur += 2
            break
        elif cv.waitKey(1) == ord('d') and Vas.EgdeKernel >= 2:
            Vas.EgdeKernel -= 2
            break
        elif cv.waitKey(1) == ord('f'):
            Vas.EgdeKernel += 2
            break
        if cv.waitKey(1) == ord('q'):
            Running = False
            break

print(f'Vas.Blur = {Vas.Blur}')
print(f'Vas.EgdeKernel = {Vas.EgdeKernel}')
