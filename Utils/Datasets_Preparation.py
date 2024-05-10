import shutil
from imageio import imsave
from imutils import paths
from math import floor
import os
from skimage.io import imsave
from skimage.transform import resize
from skimage import io


class Dataset_Prep:

    def __init__(self, dataset_path, test_size=.2, sample_per_class=5000, verbose=100):
        self.dataset_path = dataset_path
        self.test_size = test_size
        self._images_paths = list(paths.list_images(self.dataset_path))
        self.classes = dict()
        self.sample_per_class = sample_per_class
        self.verbose = verbose
        clss = self.get_classes()
        print(' Finding ', len(self._images_paths), ' Images  Belonging to ', len(clss), ' Classes ')
        for k in clss:
            print(k, ' : ', clss[k], ' images')

    def get_dirs(self):
        return self._images_paths

    def get_classes(self):
        for image in self.get_dirs():
            class_ = image.split('\\')[-2]
            if not (self.classes.keys().__contains__(class_)):
                self.classes[class_] = 1
            else:
                self.classes[class_] += 1
        return self.classes

    def _prepare_dirs(self, out_dir):
        for class_ in self.classes:
            os.makedirs(out_dir + class_, exist_ok=True)
            os.makedirs(out_dir + r'\Training\\' + class_, exist_ok=True)
            os.makedirs(out_dir + r'\Testing\\' + class_, exist_ok=True)
            os.makedirs(out_dir + r'\Validation\\' + class_, exist_ok=True)

    def shrink_dataset(self, out_dir):
        print(' Start Shrinking dataset each class contains %d' % self.sample_per_class)
        self._prepare_dirs(out_dir)
        imgs = 0
        for k in self.classes:
            self.classes[k] = self.sample_per_class
            imgs += self.classes[k]

        clss = self.classes.copy()

        i = 0
        for image in self.get_dirs():
            class_ = image.split('\\')[-2]
            if self.classes.keys().__contains__(class_):
                if self.classes[class_] <= 0:
                    continue
                else:
                    shutil.move(image, out_dir + class_ + r'\\' + image.split('\\')[-1])
                    self.classes[class_] -= 1
            i += 1
            if i % self.verbose == 0 or i == imgs:
                print('%d/%d' % (i, imgs), end=' ')
                for j in range(0, int((i / imgs) * 100), 10):
                    print('-', end='')
                print('> %1.2f%% finished' % ((i / imgs) * 100))

        print('\n--------- Shrinking Report-----------')
        for k in self.classes:
            print('{} : {}/{} Images Was Moved'.format(k,
                                                       self.sample_per_class - self.classes[k],
                                                       self.sample_per_class))
        print('\n--------------------')
        self.classes = clss.copy()
        self._images_paths = list(paths.list_images(out_dir))
        self.train_valid_test_split_dataset(out_dir)


    def resize_dataset(self, out_dir, new_size):
        print(' Start Resizing dataset each class contains %d' % self.sample_per_class)
        self._prepare_dirs(out_dir)
        imgs = 0
        for k in self.classes:
            self.classes[k] = self.sample_per_class
            imgs += self.classes[k]

        clss = self.classes.copy()

        i = 0
        for image in self.get_dirs():
            class_ = image.split('\\')[-2]
            if self.classes.keys().__contains__(class_):
                if self.classes[class_] <= 0:
                    continue
                else:
                    norm_image = io.imread(image)
                    resize_image = resize(norm_image,new_size)
                    imsave(out_dir + class_ + r'\\' + image.split('\\')[-1],resize_image)
                    self.classes[class_] -= 1
            i += 1
            if i % self.verbose == 0 or i == imgs:
                print('%d/%d' % (i, imgs), end=' ')
                for j in range(0, int((i / imgs) * 100), 10):
                    print('-', end='')
                print('> %1.2f%% finished' % ((i / imgs) * 100))

        print('\n--------- Resizing Report-----------')
        for k in self.classes:
            print('{} : {}/{} Images Was Resized'.format(k,
                                                       self.sample_per_class - self.classes[k],
                                                       self.sample_per_class))
        print('\n--------------------')
        self.classes = clss.copy()



    def train_valid_test_split_dataset(self, out_dir):
        print('---> Start Splitting dataset (train:%1.1f%% , validate:%1.1f%% , test:%1.1f%%)' %
              ((1 - self.test_size) * 100,
               self.test_size / 2 * 100, self.test_size / 2 * 100))
        clss = self.classes.copy()
        imgs = len(self.get_dirs())
        i = 0
        for image in self.get_dirs():
            class_ = image.split('\\')[-2]
            if self.classes.keys().__contains__(class_):
                if clss[class_] * self.test_size > self.classes[class_]:
                    if clss[class_] * (self.test_size / 2) > self.classes[class_]:
                        shutil.move(image, out_dir + r'\Testing\\' + class_ + '\\' + image.split('\\')[-1])
                        self.classes[class_] -= 1
                    else:
                        shutil.move(image, out_dir + r'\Validation\\' + class_ + '\\' + image.split('\\')[-1])
                        self.classes[class_] -= 1

                else:
                    shutil.move(image, out_dir + r'\Training\\' + class_ + '\\' + image.split('\\')[-1])
                    self.classes[class_] -= 1
            i += 1
            if i % self.verbose == 0 or i == imgs:
                print('%d/%d' % (i, imgs), end=' ')
                for j in range(0, int((i / imgs) * 100), 10):
                    print('-', end='')
                print('> %1.2f%% finished' % ((i / imgs) * 100))

        print('\n--------- Splitting Dataset Report-----------')
        for k in self.classes:
            print('Class {} : \n\t\t'
                  'training: {} Images Was Moved\n\t\t'
                  'validating: {} Images Was Moved\n\t\t'
                  'testing: {} Images Was Moved\n'.format(k,
                                                          floor((self.sample_per_class - self.classes[k]) * (
                                                                      1 - self.test_size)),
                                                          floor((self.sample_per_class - self.classes[k]) * (
                                                                      self.test_size / 2)),
                                                          floor((self.sample_per_class - self.classes[k]) * (
                                                                      self.test_size / 2))
                                                          ))
        print('\n--------------------')

        self.classes = clss
        for clss in self.classes:
            os.rmdir(out_dir + '\\' + clss)


# Dataset Path

dataset_path = r'C:\Users\Emran Al Hadad\Desktop\مشروع التخرج\مجلد جديد (2)\Dataset\lung_colon_image_set\colon_image_sets'

output_path = r'C:\Users\Emran Al Hadad\Desktop\ds4\\'


ds = Dataset_Prep(dataset_path)
ds.resize_dataset(output_path,(224,224,3))

