import os, logging, tempfile, argparse

import numpy as np
import cupy as cp

script_dir = os.path.join(os.path.dirname(__file__))
cusvna_include_dir = os.path.realpath(os.path.join(script_dir, '..'))

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

stream = logging.StreamHandler()
stream.setFormatter(formatter)
stream.setLevel(os.environ.get("CUSVNA_LOG_LEVEL", "DEBUG"))

logger = logging.getLogger("cusvna")
logger.propagate = False
logger.addHandler(stream)
logger.setLevel(os.environ.get("CUSVNA_LOG_LEVEL", "DEBUG"))

def _extract_cuda_version(v):
    return f"{v // 1000}.{(v % 1000) // 10}"

int_driver_version = cp.cuda.runtime.driverGetVersion()
int_cuda_version = cp.cuda.runtime.runtimeGetVersion()
driver_version = _extract_cuda_version(int_driver_version)
cuda_version = _extract_cuda_version(int_cuda_version)

if int_driver_version < 11000:
    msg = (
        "You are using an outdated NVIDIA driver, expected a CUDA >= 11.0 compatible driver but found version "
        f"{driver_version} ({int_driver_version}). Please upgrade your NVidia driver to the latest stable "
        "driver supported by your device at https://www.nvidia.fr/Download/index.aspx."
    )
    logger.warning(msg)

if int_cuda_version < 11000:
    msg = (
        "You are using an outdated CUDA version, expected at least CUDA 11.0 but found version "
        f"{cuda_version} ({int_cuda_version}). Please download and install lastest CUDA release from "
        "https://developer.nvidia.com/cuda-downloads and upgrade cupy."
    )
    logger.warning(msg)

msg = (
    f"Loaded cuda runtime {cuda_version} ({int_cuda_version}), "
    f"driver version is {driver_version} ({int_driver_version})."
)
logger.info(msg)

def determine_cuda_include_dir(logger):
    if "CUSVNA_CUDA_ROOT" in os.environ:
        cuda_root = os.environ["CUSVNA_CUDA_ROOT"].strip()
        if not os.path.isdir(cuda_root):
            msg = f"CUSVNA_CUDA_ROOT has been set to '{cuda_root}' but this directory does not exist."
            logger.error(msg)
            raise RuntimeError(msg)
    else:
        cuda_root = cp.cuda.get_cuda_path()
        if not os.path.isdir(cuda_root):
            msg = f"Cupy reports cuda path '{cuda_root}' but this directory does not exist."
            logger.error(msg)
            raise RuntimeError(msg)

    cuda_include_dir = os.path.join(cuda_root, "include")

    if not os.path.isdir(cuda_include_dir):
        msg = (
            f"Could not find cuda include directory '{cuda_include_dir}', please "
            "set the CUDA_ROOT environment variable to your cuda root directory "
            "or rebuild pycuda by providing a correct CUDA_ROOT during configuration."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    cuda_include_dir = os.path.realpath(cuda_include_dir)

    logger.info(f"Cuda include directory is '{cuda_include_dir}'.")
    return cuda_include_dir


cuda_include_dir = determine_cuda_include_dir(logger);
# fast math (flush to zero, approximate division, sqrt and transcendentals) voids cusvna guarantees
default_build_options = (f'-I{cuda_include_dir}', f'-I{cusvna_include_dir}', '--std=c++17',
                            '--restrict', '--ptxas-options', '-O3')


def without_fast_math(flags):
    """Remove fast math related nvrtc flags, they void cusvna guarantees."""
    fast_math_flags = ('fast_math', 'fast-math', 'ftz=true', 'prec-div=false', 'prec-sqrt=false')
    return tuple(f for f in flags if not any(ff in f for ff in fast_math_flags))


def math_flags(flags, fast):
    """Remove fast math flags, and ignore rounding errors if fast is set (CUSVNA_IGNORE_ROUNDING, not rigorous).
    Fast math stays forbidden in both cases: its approximate math functions break interval marching."""
    return without_fast_math(flags) + (('-DCUSVNA_IGNORE_ROUNDING',) if fast else ())


def parse_example_arguments(description):
    """Command line of the graphical examples: affine forms and rounding mode."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('form', nargs='?', default='af1', choices=('af1', 'aa'),
                        help="affine forms: af1 for cusvna::AF1 (single error term, default) or aa for cusvna::AA "
                             "(full affine arithmetic)")
    parser.add_argument('--fast', action='store_true',
                        help="ignore rounding errors (CUSVNA_IGNORE_ROUNDING), the results are no longer rigorous")
    return parser.parse_args()


def nvrtc_arch_flags():
    """Target the current device, NVRTC compiles for compute_52 by default (much slower code on recent GPUs)."""
    from cupy.cuda import nvrtc
    cc = int(cp.cuda.Device().compute_capability)
    archs = [arch for arch in nvrtc.getSupportedArchs() if arch <= cc]
    return (f'-arch=compute_{max(archs)}',) if archs else ()


def build_module_and_extract_kernels_from_source(
    logger,
    source_code,
    kernel_names,
    build_options = default_build_options,
):
    logger.info(f"Building kernel(s) {', '.join(kernel_names)}...")

    def debug_dump_kernel():
        debug_dump_file = os.path.join(tempfile.gettempdir(), "cusvna_kernel.cu")
        debug_msg = f"Build failure, kernel source code has been dumped to file://{debug_dump_file}."
        with open(debug_dump_file, "w") as f:
            f.write(source_code)
        logger.error(debug_msg)

    try:
        module = cp.RawModule(
            code=source_code,
            name_expressions=kernel_names,
            backend="nvrtc",
            options=build_options,
        )
    except Exception as e:
        debug_dump_kernel()
        logger.error(f"Failed to build module, build options were {build_options}.")
        logger.error(e)
        raise e

    try:
        kernels = tuple(map(module.get_function, kernel_names))
    except Exception as e:
        debug_dump_kernel()
        logger.error(
            f"Failed to extract kernels, build options were {build_options}."
        )
        logger.error(e)
        raise e

    return kernels


dtype_to_ctype = {
    np.bool_: "bool",
    np.int8: "char",
    np.int16: "short",
    np.int32: "int",
    np.int64: "long long",
    np.uint8: "unsigned char",
    np.uint16: "unsigned short",
    np.uint32: "unsigned int",
    np.uint64: "unsigned long long",
    np.float16: "half",
    np.float32: "float",
    np.float64: "double",
    np.complex64: "complex<float>",
    np.complex128: "complex<double>",
}
