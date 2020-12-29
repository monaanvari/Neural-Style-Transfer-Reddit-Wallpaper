import os

import transfer
import urllib
import util

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from PIL import Image
from subreddit import subreddit
from os.path import expanduser

home = expanduser("~")
jsonLimit = 50
min_width=1920
min_height=1080

if __name__== '__main__':


    main, style = util.getSubreddits()
    sub1 = subreddit(main)
    sub2 = subreddit(style)

    mainImageURL = sub1.getValidImage(jsonLimit, min_height, min_width)
    styleImageURL = sub2.getValidImage(jsonLimit, min_height, min_width)

    mainImage = Image.open(urllib.request.urlopen(mainImageURL))
    styleImage = Image.open(urllib.request.urlopen(styleImageURL))

    #image = transfer.transfer(mainImageURL, styleImageURL)
    image = transfer.transfer('https://i.pinimg.com/564x/ea/97/1a/ea971a1fb6df648975ca6a75c295087e.jpg', 'https://external-preview.redd.it/tE3IBYh4f73AILcPgOUi8RjHg9zHO_YSNewhlqTTUI8.jpg?width=640&crop=smart&auto=webp&s=8af196092868ec0c19440a8dda6ace7fb8b65907')

    filename = home+"/styledImage.jpg"

    image.save(filename)
    util.setWallpaper(filename)
