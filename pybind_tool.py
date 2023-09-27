import os, json, sys, sysconfig, inspect, shutil, argparse
from typing import Literal
from subprocess import Popen, PIPE
from sysconfig import get_paths


WIN = sys.platform.startswith("win32") and "mingw" not in sysconfig.get_platform()
MACOS = sys.platform.startswith("darwin")
LINUX = sys.platform.startswith('linux')

PYTHON_INCLUDE_DIR = get_paths()['include']


class CodeTemplate:

    @classmethod
    def add_setup_file(cls, dir_name, module_name, cxx_std=14) -> int:
        setup_path = os.path.join(dir_name, "setup.py")
        if os.path.exists(setup_path):
            print("ERROR - setup file already exist")
            return 1

        with open(setup_path, "w", encoding="utf-8") as f:
            f.write(cls.SETUP_TEMPLATE % (module_name, cxx_std))
        return 0

    @classmethod
    def add_bind_cpp(cls, dir_name, module_name) -> int:
        bind_path = os.path.join(dir_name, "bind.cpp")
        if os.path.exists(bind_path):
            print("ERROR - bind file already exist")
            return 1

        with open(bind_path, "w", encoding="utf-8") as f:
            f.write(cls.CPPBIND_TEMPLATE % module_name)
        return 0    

class PybindTemplate(CodeTemplate):

    SETUP_TEMPLATE = """\
import os
from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

FILE_PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)

ext_modules = [
    Pybind11Extension(
        "%s",
        sorted(glob(os.path.join(DIR_PATH, "*.cpp"))),
        cxx_std=%d
    ),
]

setup(cmdclass={"build_ext": build_ext}, 
      ext_modules=ext_modules)
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
    dirname = os.path.dirname(setup_path)
    cmd = ["python", setup_path, "build_ext", "--inplace"]
    proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=dirname)
    out = proc.stdout.read()
    err = proc.stderr.read()
    if proc.wait() == 1:
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

def init_dylib(dirname, check=False):
    if check and check_dylib(dirname):
        return 0
    return build_setup(os.path.join(dirname, "setup.py"))

def clean_build(dirname):
    build_dir = os.path.join(dirname, "build")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    for file in os.listdir(dirname):
        file_suffix = file.rpartition(".")[-1]
        if file_suffix in ['so', 'dylib']:
            os.remove(os.path.join(dirname, file))

PKG_INIT_FILE_FUNCS = [
    build_setup, check_dylib, init_dylib
]
PKG_INIT_FILE_FUNCS_SRC = "\n".join([
    inspect.getsource(f)
    for f in PKG_INIT_FILE_FUNCS
])

PKG_INIT_FILE_TEMPLATE = f"""\
import os, sys, sysconfig
from subprocess import Popen, PIPE

# set False when release
DEBUG = True

WIN = sys.platform.startswith("win32") and "mingw" not in sysconfig.get_platform()
MACOS = sys.platform.startswith("darwin")
FILE_PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)

{PKG_INIT_FILE_FUNCS_SRC}

init_dylib(DIR_PATH, check=not DEBUG)

from .%s import *
"""

def add_init_file(dir_name, module_name):
    init_file_path = os.path.join(dir_name, "__init__.py")
    if os.path.exists(init_file_path):
        print("ERROR - init file already exist")
        return 1

    with open(init_file_path, "w", encoding="utf-8") as f:
        f.write(PKG_INIT_FILE_TEMPLATE % module_name)
    return 0    

def init_pybind(dirname, module_name, cxx_std=14, ext='clangd'):
    from pybind11 import get_include
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if ext == 'clangd':
        add_clangd_include(dirname, [PYTHON_INCLUDE_DIR, get_include()])
        add_clangd_flags(dirname, [f"-std=c++{cxx_std}"])
    PybindTemplate.add_bind_cpp(dirname, module_name)
    PybindTemplate.add_setup_file(dirname, module_name, cxx_std=cxx_std)
    add_init_file(dirname, module_name)

class TorchExtTemplate(CodeTemplate):
    
    SETUP_TEMPLATE = """\
import os
from glob import glob

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

FILE_PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)

setup(
    ext_modules=[
        CppExtension(
            name="%s", 
            sources=sorted(glob(os.path.join(DIR_PATH, "*.cpp"))),
            extra_compile_args=['-stdc++%d']
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)
"""

    CPPBIND_TEMPLATE = """\
#include <torch/extension.h>

using torch::Tensor;

Tensor trilinear_interpolation
(Tensor feats, Tensor point)
{
    return feats;
}

PYBIND11_MODULE(%s, m){
    m.def("trilinear_interpolation", &trilinear_interpolation, "Test forward");
}
"""

class TorchCUDATemplate(CodeTemplate):
    SETUP_TEMPLATE = """\
import os
from glob import glob

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

FILE_PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)

setup(
    ext_modules=[
        CUDAExtension(
            name="%s", 
            sources=[*glob(os.path.join(DIR_PATH, "*.cpp")),
                     *glob(os.path.join(DIR_PATH, "*.cu"))],
            extra_compile_args={'cxx': ['-O2', '-stdc++%d'],
                                'nvcc': ['-O2']}
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)

"""

    CPPBIND_TEMPLATE = """\
#include <torch/extension.h>
#include "include/utils.hpp"

using torch::Tensor;

Tensor trilinear_fw_cu
(Tensor feats, Tensor point);

Tensor trilinear_interpolation
(Tensor feats, Tensor point)
{   
    CHECK_CUDA(feats);
    CHECK_CUDA(point);
    return trilinear_fw_cu(feats, point);
}

PYBIND11_MODULE(%s, m){
    m.def("trilinear_interpolation", &trilinear_interpolation, "Test forward");
}
"""

    UTILS_TEMPLATE = """\
#ifndef __UTILS_HPP

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#endif
"""

    CUDA_FUNC_TEMPLATE = """\
#include <torch/extension.h>

using torch::Tensor;
using torch::PackedTensorAccessor32;
using torch::RestrictPtrTraits;

// __global__ cuda 关键字，修饰的函数在 cpu 上调用，gpu 上执行
// __host__ cuda 关键字，修饰的函数在 cpu 上调用，cpu 上执行
// __device__ cuda 关键字，修饰的函数在 gpu 上调用，gpu 上执行
// f<<<blocks, threads>>> 函数 f 并行的在 blocks, threads 密铺结构上进行运算

// accessor - Tensor 获取数据的索引器，cpu使用，packer_accessor - gpu 使用
template<typename scalar_t>
__global__ void trilinear_fw_kernel(    
    const PackedTensorAccessor32<scalar_t, 3, RestrictPtrTraits> feats,
    const PackedTensorAccessor32<scalar_t, 2, RestrictPtrTraits> points,
    PackedTensorAccessor32<scalar_t, 2, RestrictPtrTraits> feat_interp
) {
    // blockIdx - block 所在坐标，threadIdx - 在当前block 中 thread 所在坐标
    // n, f 为密铺block 中 thread 所在的总坐标
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;

    const int N = feats.size(0), F = feats.size(2);
    if (n >= N && f >= F) {
        return;
    }

    // 计算 feat_interp[n][f] 的值
    // point -1 ~ 1 之间，先进性标准化
    const scalar_t u = (points[n][0]+1)/2;
    const scalar_t v = (points[n][1]+1)/2;
    const scalar_t w = (points[n][2]+1)/2;
    
    const scalar_t a = (1-v)*(1-w);
    const scalar_t b = (1-v)*w;
    const scalar_t c = v*(1-w);
    const scalar_t d = 1-a-b-c;
    feat_interp[n][f] = (1-u)*(a*feats[n][0][f] +
                               b*feats[n][1][f] +
                               c*feats[n][2][f] +
                               d*feats[n][3][f]) + 
                            u*(a*feats[n][4][f] +
                               b*feats[n][5][f] +
                               c*feats[n][6][f] +
                               d*feats[n][7][f]);
};
 

// 三线性插值，feats shape: [N, 8, F]，point shape: [N, 3]，
// N - sample dim, F - feature dim
Tensor trilinear_fw_cu
(Tensor feats, Tensor points)
{   
    const int N = feats.size(0), F = feats.size(2);

    // 设定返回结果 Tensor，同时 dtype，device 和 feats 一样
    Tensor feat_interp = torch::empty({N, F}, feats.options());
    // 宁外一种写法，手动设置torch 类型和设备
    // torch::zeros({N, F}, torch::dtype(torch::kInt32).device(feats.device()));

    // 单 block 的线程形状
    const dim3 threads(16, 16); // 线程数最多 256

    // 密铺多少 block 可以覆盖 N * F 的向量
    const dim3 blocks((N + threads.x - 1) / threads.x, (F + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES(feats.scalar_type(), "trilinear_fw_cu", [&]{
        trilinear_fw_kernel<scalar_t><<<blocks, threads>>>(
            feats.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),
            points.packed_accessor32<scalar_t, 2, RestrictPtrTraits>(),
            feat_interp.packed_accessor32<scalar_t, 2, RestrictPtrTraits>()
        );
    });
    return feat_interp;
}
"""

    @classmethod
    def add_static_files(cls, dirname):
        include_dir = os.path.join(dirname, "include")
        if not os.path.exists(include_dir):
            os.makedirs(include_dir)
        utils_path = os.path.join(include_dir, "utils.hpp")
        with open(utils_path, 'w', encoding='utf-8') as f:
            f.write(cls.UTILS_TEMPLATE)

        cuda_func_path = os.path.join(dirname, "interpolation_kernel.cu")
        with open(cuda_func_path, 'w', encoding='utf-8') as f:
            f.write(cls.CUDA_FUNC_TEMPLATE)        

def init_torch_ext(dirname, module_name, cxx_std=17, ext='clangd', cuda=False):
    """
    before import torch cpp extension, should import torch first
    """
    from torch.utils.cpp_extension import include_paths
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if ext == 'clangd':
        add_clangd_include(dirname, [PYTHON_INCLUDE_DIR, *include_paths(cuda)])
        ext_flags = [f"-std=c++{cxx_std}"]
        if cuda:
            from torch.utils.cpp_extension import CUDA_HOME
            if not CUDA_HOME:
                print("ERROR - can't find cuda")
            cuda_path_flag = f"--cuda-path={CUDA_HOME}"
            ext_flags.append(cuda_path_flag)
        add_clangd_flags(dirname, ext_flags)
    if not cuda:
        TorchExtTemplate.add_bind_cpp(dirname, module_name)
        TorchExtTemplate.add_setup_file(dirname, module_name, cxx_std=cxx_std)
    else:
        TorchCUDATemplate.add_bind_cpp(dirname, module_name)
        TorchCUDATemplate.add_setup_file(dirname, module_name, cxx_std=cxx_std)
        TorchCUDATemplate.add_static_files(dirname)
    add_init_file(dirname, module_name)


def init_dispatch(
    module_name:str,
    module_type:Literal["pybind", "torch"],
    dirname:str,
    cxx_std:int,
    vscode_ext:Literal["clangd"],
    cuda:bool
) -> int:

    if module_type == "pybind":
        init_pybind(dirname, module_name, cxx_std=cxx_std, ext=vscode_ext)
    elif module_type == "torch":
        init_torch_ext(dirname, module_name, cxx_std, vscode_ext, cuda)
    else:
        print(f"ERROR - only support [pybind, torch_ext], actually {module_type}")
        return 1
    return 0

def main():
    parser = argparse.ArgumentParser(
        prog='pybind_tool', 
        description='simple stage for python pybind extension', 
        epilog='Copyright(r), 2023'
    )
    parser.add_argument('module_name', nargs=1, help="module name for cpp extension")
    parser.add_argument('-t', '--type', dest="module_type",
                        choices=['pybind', 'torch'], default='pybind',
                        help="module type for cpp extension, support pybind, torch")
    parser.add_argument('-cxx_std', dest='cxx_std', type=int, default=14,
                        help="cpp std")
    parser.add_argument('-o', '--out', dest='dirname', default='.',
                        help="output dir for cpp extension")
    parser.add_argument('-e', '--ext', dest='vscode_ext', default='clangd', choices=['clangd'])
    parser.add_argument('--cuda', dest='cuda', action='store_true',
                        help='only in torch cpp extension, if build cuda ext')

    # parser.add_argument
    args = parser.parse_args()
    norm_args = {**args.__dict__}
    norm_args['module_name'] = args.module_name[0]
    init_dispatch(**norm_args)


if __name__ == '__main__':
    main()