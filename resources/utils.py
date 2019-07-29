import os
import os.path as osp
import pickle


class special_string():
    def __init__(self):
        self._str = ''

    def newline(self):
        self._str += '\n'

    def append_str(self, incoming_str):
        self._str += incoming_str

    def add_line(self, incoming_str):
        self.newline()
        self.append_str(incoming_str)

    def get_str(self):
        return self._str

    def set_str(self, incoming_str):
        self._str = incoming_str

    def __str__(self):
        return self._str


def save_obj(file_name, obj):
    file_name = file_name if '.pkl' in file_name else file_name + '.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(obj, file)


def load_obj(file_name):
    file_name = file_name if '.pkl' in file_name else file_name + '.pkl'
    with open(file_name, 'rb') as file:
        obj = pickle.load(file)
    return obj


def delete_dir(dir):
    try:
        import shutil
        shutil.rmtree(dir, ignore_errors=True)
    except:
        print("couldn't delete directory: ", dir)


def ensure_dir_exists(dir, delete_directory=False):
    if osp.exists(dir) and delete_directory:
        print("Deleting directory", dir)
        delete_dir(dir)
    if not osp.exists(dir):
        os.makedirs(dir)
    return dir
