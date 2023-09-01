import os, json
from sysconfig import get_paths
from pybind11 import get_include


PYTHON_INCLUDE_DIR = get_paths()['include']
PYBIND_INCLUDE_DIRS = get_include()

BASE_INCLUDE_DIRS = [PYTHON_INCLUDE_DIR, PYBIND_INCLUDE_DIRS]

SETUP_TEMPLATE = """\
from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "%s",
        sorted(glob("*.cpp")),
    ),
]

setup(cmdclass={"build_ext": build_ext}, ext_modules=ext_modules)
"""

CPPBIND_TEMPLATE = """\
#include <pybind11/pybind11.h>

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(%s, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
}
"""

def add_setup_file(dir_name, module_name) -> int:
    setup_path = os.path.join(dir_name, "setup.py")
    if os.path.exists(setup_path):
        print("ERROR - setup file already exist")
        return 1

    with open(setup_path, "w", encoding="utf-8") as f:
        f.write(SETUP_TEMPLATE % module_name)
    return 0

def add_bind_cpp(dir_name, module_name) -> int:
    bind_path = os.path.join(dir_name, "bind.cpp")
    if os.path.exists(bind_path):
        print("ERROR - bind file already exist")
        return 1

    with open(bind_path, "w", encoding="utf-8") as f:
        f.write(CPPBIND_TEMPLATE % module_name)
    return 0    

def create_vscode_setting(dirname):
    vscode_dir = os.path.join(dirname, ".vscode")
    if not os.path.exists(vscode_dir):
        os.makedirs(vscode_dir)
    setting_path = os.path.join(vscode_dir, 'settings.json')
    if not os.path.exists(setting_path):
        with open(setting_path, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=4, ensure_ascii=False)

def load_vscode_setting(dirname):
    create_vscode_setting(dirname)
    setting_path = os.path.join(dirname, ".vscode", 'settings.json')
    with open(setting_path, "r", encoding="utf-8") as f:
        setting = json.load(f)
    return setting

def save_vscode_setting(dirname, setting:dict):
    create_vscode_setting(dirname)
    setting_path = os.path.join(dirname, ".vscode", 'settings.json')
    with open(setting_path, "w", encoding="utf-8") as f:
        json.dump(setting, f, indent=4, ensure_ascii=False)

def add_clangd_include(dirname, include_dirs):

    setting = load_vscode_setting(dirname)
    if "clangd.fallbackFlags" not in setting:
        setting["clangd.fallbackFlags"] = []
    flags = setting["clangd.fallbackFlags"]
    for include_dir in include_dirs:
        if f"-I{include_dir}" not in flags:
            flags.append(f"-I{include_dir}")
    save_vscode_setting(dirname, setting)

def init_pybind(dirname, module_name, ext='clangd'):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if ext == 'clangd':
        add_clangd_include(dirname, BASE_INCLUDE_DIRS)
    add_bind_cpp(dirname, module_name)
    add_setup_file(dirname, module_name)



