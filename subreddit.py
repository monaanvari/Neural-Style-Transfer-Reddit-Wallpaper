
import os
from os.path import expanduser

import requests
import urllib
from PIL import ImageFile

class submission:
    
    def __init__(self, URL, min_height, min_width):
        self.URL = URL
        self.min_height = min_height
        self.min_width = min_width

    def validURL(self):
        statusCode = requests.get(self.URL, headers = {'User-agent':'getWallpapers'}).status_code
        return statusCode != 404

    def knownURL(self):
        URL = self.URL
        return URL.lower().startswith('https://i.redd.it/') or URL.lower().startswith('http://i.redd.it/') or URL.lower().startswith('https://i.imgur.com/') or URL.lower().startswith('http://i.imgur.com/')

    def isImage(self):
        if self.URL.endswith(('.png', '.jpeg', '.jpg')):
            return True
        return False

    def isLandscape(self):
        file = urllib.request.urlopen(self.URL)
        size = file.headers.get('content-length')
        parser = ImageFile.Parser()

        while True:
            data=file.read(1024)
            if not data:
                break
            parser.feed(data)
            if parser.image:
                return parser.image.size[0] >= parser.image.size[1]
    
        file.close()
        return False

    def isHD(self):
        file = urllib.request.urlopen(self.URL)
        size = file.headers.get('content-length')
        parser = ImageFile.Parser()

        while True:
            data = file.read(1024)
            if not data:
                break
            parser.feed(data)
            if parser.image:
                return parser.image.size[0] >= self.min_width and parser.image.size[1] >= self.min_height

        file.close()
        return False

    def __del__(self): 
        pass


class subreddit:

    def __init__(self, name):
        self.name = name

    def getPosts(self, jsonLimit):
        allPosts = []
        URL = 'https://reddit.com/r/{}/top/.json?t=all&limit={}'.format(self.name, jsonLimit)
        posts = requests.get(URL, headers = {'User-agent':'YerAWizardHarry'}).json()

        for post in posts['data']['children']:
            allPosts.append(post)

        return allPosts

    def getValidImage(self, jsonLimit, min_height, min_width):

        allPosts = self.getPosts(jsonLimit)

        for p in allPosts:
            
            URL = p['data']['url']
            post = submission(URL, min_height, min_width)

            if post.validURL() and post.knownURL() and post.isImage() and post.isLandscape() and post.isHD():
                return post.URL

        print('Was not able to find any valid images from subreddit {} '.format(subreddit.name))
        sys.exit()

    def __del__(self): 
        pass


