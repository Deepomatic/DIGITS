import uuid
import numpy as np
import os.path
import string
import PIL.Image

VALID_CHARS = "-_.%s%s" % (string.ascii_letters, string.digits)

class Image(object):

    def __init__(self, stringIOOrFilenameOrNpArray):
        if isinstance(stringIOOrFilenameOrNpArray, np.ndarray):
            self.img = PIL.Image.fromarray(stringIOOrFilenameOrNpArray)
        else:   
            self.img = PIL.Image.open(stringIOOrFilenameOrNpArray)
        self.format = self.img.format

    @staticmethod
    def addNoise(image, noise_size, axe):
        if len(noise_size) == 3 and noise_size[2] == 1 and len(image.shape) == 2:
            noise_size = (noise_size[0], noise_size[1])

        noise = np.random.randint(0, 255, noise_size).astype('uint8')
        return np.concatenate((noise, image, noise), axis=axe)

    @staticmethod
    def addColor(image, color, color_size, axe):
        size = None
        if len(color_size) == 3 and color_size[2] == 1 and isinstance(color, tuple):
            color = [ Image.rgbtogray(color) ]
            if len(image.shape) == 2:
                color_size = (color_size[0], color_size[1])
            else:
                size = color_size[2]
        else:
            size = color_size[2]

        fill = np.zeros(color_size, 'uint8')
        if size:
            for i in range(size):
                fill[...,i] = color[i]
        else:
            fill[...] = color[0]
        return np.concatenate((fill, image, fill), axis=axe)        

    @staticmethod
    def getFill(img, size):
        resize_width, resize_height = size
        img_width, img_height = img.size
        height_diff = resize_height - img_height
        width_diff = resize_width - img_width

        if not height_diff and not width_diff:
            return None, None

        fill_size_height = None
        fill_size_width = None
        if width_diff:
            width_fill = width_diff / 2
            fill_size_width = (img_height, width_fill, len(img.mode))
            img_width += width_fill * 2

        if height_diff:
            fill_size_height = (height_diff / 2, img_width, len(img.mode))

        return fill_size_width, fill_size_height


    @staticmethod
    def rgbtogray(rgb):
        return int(0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2])

    def fillImageWithColor(self, size, color = (0, 0, 0)):
        fill_size = Image.getFill(self.img, size)
        if fill_size is (None, None):
            return

        left = 0
        top = size[1]
        right = size[0]
        bottom = 0

        image = np.array(self.img)
        if fill_size[0]:
            image = Image.addColor(image, color, fill_size[0], 1)
            left += fill_size[0][1]
            right -= fill_size[0][1]

        if fill_size[1]:
            image = Image.addColor(image, color, fill_size[1], 0)
            bottom += fill_size[1][0]
            top -= fill_size[1][0]

        self.img = PIL.Image.fromarray(image)
        return left, bottom, right, top


    def fillImageWithNoise(self, size):
        fill_size = Image.getFill(self.img, size)
        if fill_size is (None, None):
            return

        left = 0
        top = size[1]
        right = size[0]
        bottom = 0

        image = np.array(self.img)
        if fill_size[0]:
            image = Image.addNoise(image, fill_size[0], 1)
            left += fill_size[0][1]
            right -= fill_size[0][1]

        if fill_size[1]:
            image = Image.addNoise(image, fill_size[1], 0)
            bottom += fill_size[1][0]
            top -= fill_size[1][0]

        self.img = PIL.Image.fromarray(image)
        return left, bottom, right, top

    def isGrayscaleFromMode(self):
        return self.img.mode == "L"

    def isGrayscaleRGB(self):
        if self.img.mode != "RGB":
            raise Exception ("Must be an RGB image")

        w,h = self.img.size
        for i in range(w):
            for j in range(h):
                r,g,b = self.img.getpixel((i,j))
                if r != g != b: return False
        return True


    def resize(self, resize):
        if resize[0] > self.img.size[0] or resize[1] > self.img.size[1]:
            width, height = resize
            ratio = float(self.img.size[0]) / float(self.img.size[1])
            if ratio > 1:
                height = int(width / ratio)
            else:
                width = int(height * ratio)
            self.img = self.img.resize((width, height), PIL.Image.ANTIALIAS)
        else:
            self.img.thumbnail(resize, PIL.Image.ANTIALIAS)

    def convert(self, mode):
        if self.img.mode != mode:
            self.img = self.img.convert(mode)
        
    def saveWithCleanName(self, path, filename):

        filename, _ = os.path.splitext(filename.replace(" ", "_"))
        filename = ''.join(c for c in filename if c in VALID_CHARS)

        if not filename:
            filename = str(uuid.uuid4())

        filename += "." + self.format
        
        fullPath = os.path.join(path, filename)

        self.img.save(fullPath, self.format)

        return fullPath

    def toNumpyArray(self):
        return np.array(self.img)

    def crop(self, box):
        self.img = self.img.crop(box)

    @staticmethod
    def toCvRectFormat(rect):
        return rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]


    def crop_fill(self, new_size, ratioCrop = 0.5):
        width, height = self.img.size
        width_diff = width - new_size[0]
        height_diff = height - new_size[1]

        new_width, new_height = width, height
        if width_diff < 0:
            new_width = new_size[0] - width_diff

        if height_diff < 0:
            new_height = new_size[1] - height_diff

        if new_width != width or new_height != height:
            self.resize((new_width, new_height))
            width, height = self.img.size
            width_diff = width - new_size[0]
            height_diff = height - new_size[1]

        if ratioCrop:
            ratioCropSide = float(ratioCrop) / 2

            width_diff_ratio = int(width_diff * ratioCropSide)
            height_diff_ratio =  int(height_diff * ratioCropSide)

            box = width_diff_ratio, height_diff_ratio, width - width_diff_ratio, height - height_diff_ratio
            self.crop(box)

        self.resize(new_size)
        return self.fillImageWithNoise(new_size)

    def resize_grey(self, new_size, ratioCrop = 0.5):
        width, height = self.img.size
        width_diff = width - new_size[0]
        height_diff = height - new_size[1]

        new_width, new_height = width, height
        if width_diff < 0:
            new_width = new_size[0] - width_diff

        if height_diff < 0:
            new_height = new_size[1] - height_diff

        if new_width != width or new_height != height:
            self.resize((new_width, new_height))
            width, height = self.img.size
            width_diff = width - new_size[0]
            height_diff = height - new_size[1]

        if ratioCrop:
            ratioCropSide = float(ratioCrop) / 2

            width_diff_ratio = int(width_diff * ratioCropSide)
            height_diff_ratio =  int(height_diff * ratioCropSide)

            box = width_diff_ratio, height_diff_ratio, width - width_diff_ratio, height - height_diff_ratio
            self.crop(box)

        self.resize(new_size)
        return self.fillImageWithColor(new_size,color = (128, 128, 128))


    def show(self):
        self.img.show()

    def size(self):
        return self.img.size
    
    def size_up(self, resize):
        width,height = self.img.size
        scalar_1 = float(resize[0])/width
        if scalar_1*height > resize[1]:
            scalar = scalar_1
        else :
            scalar = float(resize[0])/height
        new_w = int(round(scalar*width))
        new_h = int(round(scalar*height))
        return new_w, new_h, scalar
