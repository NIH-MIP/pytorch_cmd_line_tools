import fastai  # pylint: disable=unused-import
from fastai.vision import *  # pylint: disable=unused-wildcard-import
from fastai.callbacks import *  # pylint: disable=unused-wildcard-import
from fastai.utils.mem import *  # pylint: disable=unused-wildcard-import
from typing import *
from fastai.vision.learner import cnn_config
from torchvision import transforms
from numbers import Integral
from torchvision.models import vgg16_bn
import pathlib
import pydicom
import imageio
import numpy as np


def get_data(src: ImageList, labels, tfms, bs:int, size:int):
    data = (src
            .label_from_func(labels.class_label)
            .transform(tfms, size=size, tfm_y=True)
            .databunch(bs=bs)
            .normalize(do_y=True))
    data.c = 3
    return data


def do_fit(learn, save_name, lr=1e-5, lrs=None, cycles=5, pct_start=0.9, rows=1, imgsize=5):
    # initialize arguments
    if lrs == None: lrs = slice(lr)

    # begin learning cycles
    learn.fit_one_cycle(cycles, lrs, pct_start=pct_start)
    learn.save(save_name)
    learn.show_results(rows=rows, imgsize=imgsize)




class NumpyReader(ImageList):

    def __init__(self, items, **kwargs):
        super().__init__(items, **kwargs)

    """ def open(self, fn):
        "open dicom using 'fn' subclass"
        path = str(fn)
        ds = pydicom.dcmread(path, force=True)
        ds.convert_pixel_data
        x = ds.pixel_array
        px_data = pil2tensor(x, np.float32)
        return DicomSlice(px_data) """

    def open(self, fn: PathOrStr):
        "open dicom using 'fn' subclass"
        x = np.load(fn)
        px_data = self.open_image(x, path_func=fn)
        return px_data


    def open_image(self, im: np.ndarray, path_func=None, div: bool = False,
                   cls: type = Image,
                   after_open: Callable = None) -> Image:
        "Return `Image` object created from image in file `fn`."
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # EXIF warning from TiffPlugin
            x = PIL.Image.fromarray(im).convert("RGB")
            if div: # normalization step
                #x/=np.max(x)
                x -= np.mean(x)
                x /= np.std(x)
                x = self.window(x)
                #x += 3 # may only need to do for writing out to dicom.
                #x /= 6
        if after_open: x = after_open(x)
        x = pil2tensor(x, np.float32)
        return cls(x)

    def window(self, x):
        t_mean = np.mean(x)
        t_max = np.max(x)
        t_min = np.min(x)
        #x = (x - t_min) / (t_max - t_min)
        x = (x - t_min) / (t_max)
        # x = pil2tensor(x,np.float32)
        return x

    def get(self, i):
        # fn = super().get(i)
        fn = self.items[i]
        res = self.open(fn)
        # self.sizes[i] = res.size
        return res

    def __getitem__(self, idxs: int) -> Any:
        idxs = try_int(idxs)
        if isinstance(idxs, Integral):
            return self.get(idxs)
        else:
            return self.new(self.items[idxs], inner_df=index_row(self.inner_df, idxs))


    # finish analyze_pre or decide if it needs to be finished or not.
    # Does not need to be completed processes results of predict prior to reconstruction
    # def analyze_pred(self, pred): return DicomSlice(pred.float())

    def filter_by_func(self, func: Callable) -> 'ItemList':
        "Only keep elements for which `func` returns `True`."
        self.items = array([o for o in self.items if func(o)])
        return self

    def filter_by_structure(self, include=None, exclude=None):
        """
        filters items given a file structure.
        file structures should appears as a list of lists. One list for each branch of the file tree you want to include or exclude.
        example:
        include = ['a', 'b', 'c']
        exclude = ['a', 'b', 'd']
        includes all files in branch ./a/b/c
        and excludes all files in branch ./a/b/d
        """
        paths_include = include
        paths_exclude = exclude

        def _inner(o):
            if isinstance(o, Path):
                n = o.relative_to(self.path).parts[:]
            else:
                n = o.split(os.path.sep)[len(str(self.path).split(os.path.sep))]
            if not paths_include == None:
                if any(all(x in n for x in p) for p in paths_include): return True
            if not paths_exclude == None:
                if any(all(x in n for x in p) for p in paths_exclude): return False
            if paths_include == None:
                return True
            return False

        return self.filter_by_func(_inner)

    @classmethod
    def from_folder(cls, path: PathOrStr, extensions: Collection[str] = None, recurse: bool = True,
                    include: Optional[Collection[str]] = None, processor=None, **kwargs) -> 'ItemList':
        """Create an `ItemList` in `path` from the filenames that have a suffix in `extensions`.
        `recurse` determines if we search subfolders."""
        path = Path(path)
        extensions = None
        return cls(cls.get_files(path, extensions, recurse=recurse, include=include), path=path, processor=processor,
                   **kwargs)

    @classmethod
    def _get_files(cls, parent, p, f, extensions):
        p = Path(p)  # .relative_to(parent)
        res = [p / o for o in f if not o.startswith('.')]
        return res

    @classmethod
    def get_files(cls, path: PathOrStr, extensions: Collection[str] = None, recurse: bool = True,
                  include: Optional[Collection[str]] = None) -> FilePathList:
        "Return list of files in `path` that have a suffix in `extensions`; optionally `recurse`."
        if recurse:
            res = []
            for p, d, f in os.walk(path):
                # skip hidden dirs
                if include is not None:
                    d[:] = [o for o in d if o in include]
                else:
                    d[:] = [o for o in d if not o.startswith('.')]
                res += cls._get_files(path, p, f, extensions)
            return res
        else:
            f = [o.name for o in os.scandir(path) if o.is_file()]
            return cls._get_files(path, path, f, extensions)