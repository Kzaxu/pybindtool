import os, json, sys, sysconfig
from subprocess import Popen, PIPE
from sysconfig import get_paths


WIN = sys.platform.startswith("win32") and "mingw" not in sysconfig.get_platform()
MACOS = sys.platform.startswith("darwin")

PYTHON_INCLUDE_DIR = get_paths()['include']

SETUP_TEMPLATE = """\
from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "%s",
        sorted(glob("*.cpp")),
        cxx_std=%d
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

def add_setup_file(dir_name, module_name, cxx_std=14) -> int:
    setup_path = os.path.join(dir_name, "setup.py")
    if os.path.exists(setup_path):
        print("ERROR - setup file already exist")
        return 1

    with open(setup_path, "w", encoding="utf-8") as f:
        f.write(SETUP_TEMPLATE % (module_name, cxx_std))
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

def add_clangd_flags(dirname, flags):
    setting = load_vscode_setting(dirname)
    if "clangd.fallbackFlags" not in setting:
        setting["clangd.fallbackFlags"] = []
    fall_back_flags = setting["clangd.fallbackFlags"]
    for f in flags:
        if f not in fall_back_flags:
            fall_back_flags.append(f)
    save_vscode_setting(dirname, setting)    

def add_clangd_include(dirname, include_dirs):
    flags = []
    for include_dir in include_dirs:
        if f"-I{include_dir}" not in flags:
            flags.append(f"-I{include_dir}")
    add_clangd_flags(dirname, flags)

def build_setup(setup_path) -> int:
    if not os.path.exists(setup_path):
        print("ERROR - setup file not exist")
        return 1
    
    cmd = ["python", setup_path, "build_ext", "--inplace"]
    proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    if proc.wait() == 1:
        out = proc.stdout.read().decode()
        err = proc.stderr.read().decode()
        print("ERROR - build setup fail, info:")
        print(out)
        print(err)
        return 1
    return 0

def check_dylib(dirname):
    dylib_suffix = []
    if WIN:
        dylib_suffix.append("pyd")
    elif MACOS:
        dylib_suffix.append("so")
    else:
        dylib_suffix.append("so")
    
    files = os.listdir(dirname)
    for file in files:
        suffix = file.rpartition(".")[-1]
        if suffix in dylib_suffix:
            return True
    return False

def init_pybind(dirname, module_name, cxx_std=14, ext='clangd'):
    from pybind11 import get_include
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if ext == 'clangd':
        add_clangd_include(dirname, [PYTHON_INCLUDE_DIR, get_include()])
        add_clangd_flags(dirname, [f"-std=c++{cxx_std}"])
    add_bind_cpp(dirname, module_name)
    add_setup_file(dirname, module_name)



