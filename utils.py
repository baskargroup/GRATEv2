import numpy as np
import cv2
import matplotlib.pyplot as plt

##################### Utils ##################################

def invertBinaryImage(im):
    ## Flips the image pixel values(0 and 255) and returns the inverted image
    inv = np.zeros(im.shape)
    inv[im==0] = 255
    return inv

def ConcatAndShow(input, output, scale_percent,text):
    border = np.zeros((input.shape[0],75)) # Setting border between the concatenated images
    img=np.concatenate((input,border,output),axis=1)
    print(text + " \n")
    show_scalled_img(img,scale_percent)

def show_scalled_img(img_arr, scalePercent=100):
    ## shows the image at a scale percent of the origianl size.   
    scale_percent = scalePercent

    #calculate the 50 percent of original dimensions
    width = int(img_arr.shape[1] * scale_percent / 100)
    height = int(img_arr.shape[0] * scale_percent / 100)

    # resize image
    output = cv2.resize(img_arr, (width, height))

    fig = plt.figure(figsize = (50,50))
    plt.imshow(output, cmap='gray', vmin=0, vmax=255)
    plt.show()

def debugORSave(initial, final, params, concat, text):
    if params['show intermediate images'] == 1:
        if concat == 1:
            ConcatAndShow(initial, final, params['display image scaling'], text)
        else:
            print(text + "\n")
            show_scalled_img(final, params['display image scaling'])
    if params['save intermediate images'] == 1:
        cv2.imwrite(text+".png", final)
