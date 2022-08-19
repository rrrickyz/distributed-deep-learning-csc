## I cannot build Horovod with other needed frameworks:
## this is the slurm output:
 
raise ValueError('Neither MPI nor Gloo support has been built. Try reinstalling Horovod ensuring that '
ValueError: Neither MPI nor Gloo support has been built. Try reinstalling Horovod ensuring that either MPI is installed (MPI) or CMake is installed (Gloo)

## command for installing Horovod:
> python -m pip install --user horovod[all-frameworks]

> horovodrun --check-build --verbose

Checking whether extension tensorflow was built.

Traceback (most recent call last):

  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 83, in _target_fn
  
    ext = importlib.import_module('.' + ext_base_name, 'horovod')
    
  File "/appl/opt/anaconda3/2019.3/envs/default/lib/python3.7/importlib/__init__.py", line 127, in import_module
  
    return _bootstrap._gcd_import(name[level:], package, level)
    
  File "<frozen importlib._bootstrap>", line 1006, in _gcd_import
  
  File "<frozen importlib._bootstrap>", line 983, in _find_and_load
  
  File "<frozen importlib._bootstrap>", line 967, in _find_and_load_unlocked
  
  File "<frozen importlib._bootstrap>", line 677, in _load_unlocked
  
  File "<frozen importlib._bootstrap_external>", line 728, in exec_module
  
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/tensorflow/__init__.py", line 24, in <module>
  
    check_extension('horovod.tensorflow', 'HOROVOD_WITH_TENSORFLOW', __file__, 'mpi_lib')
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 59, in check_extension
  
    ext_name, full_path, ext_env_var
  
ImportError: Extension horovod.tensorflow has not been built: /users/tongwenz/.local/lib/python3.7/site-packages/horovod/tensorflow/mpi_lib.cpython-37m-x86_64-linux-gnu.so not found
  
If this is not expected, reinstall Horovod with HOROVOD_WITH_TENSORFLOW=1 to debug the build error.
  
Extension tensorflow was NOT built.
  
Checking whether extension torch was built.
  
Extension horovod.torch has not been built: /users/tongwenz/.local/lib/python3.7/site-packages/horovod/torch/mpi_lib/_mpi_lib.cpython-37m-x86_64-linux-gnu.so not found
  
If this is not expected, reinstall Horovod with HOROVOD_WITH_PYTORCH=1 to debug the build error.
  
Warning! MPI libs are missing, but python applications are still available.
  
Extension torch was built.
  
Checking whether extension mxnet was built.
  
Traceback (most recent call last):
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 83, in _target_fn
  
    ext = importlib.import_module('.' + ext_base_name, 'horovod')
  
  File "/appl/opt/anaconda3/2019.3/envs/default/lib/python3.7/importlib/__init__.py", line 127, in import_module
  
    return _bootstrap._gcd_import(name[level:], package, level)
  
  File "<frozen importlib._bootstrap>", line 1006, in _gcd_import
  
  File "<frozen importlib._bootstrap>", line 983, in _find_and_load
  
  File "<frozen importlib._bootstrap>", line 967, in _find_and_load_unlocked
  
  File "<frozen importlib._bootstrap>", line 677, in _load_unlocked
  
  File "<frozen importlib._bootstrap_external>", line 728, in exec_module
  
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/mxnet/__init__.py", line 19, in <module>
    __file__, 'mpi_lib')
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 59, in check_extension<br>
    ext_name, full_path, ext_env_var
  
ImportError: Extension horovod.mxnet has not been built: /users/tongwenz/.local/lib/python3.7/site-packages/horovod/mxnet/mpi_lib.cpython-37m-x86_64-linux-gnu.so not found
  
If this is not expected, reinstall Horovod with HOROVOD_WITH_MXNET=1 to debug the build error.
  
Extension mxnet was NOT built.
  
Checking whether extension tensorflow was built with MPI.
  
Traceback (most recent call last):
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 83, in _target_fn
  
    ext = importlib.import_module('.' + ext_base_name, 'horovod')
  
  File "/appl/opt/anaconda3/2019.3/envs/default/lib/python3.7/importlib/__init__.py", line 127, in import_module
  
    return _bootstrap._gcd_import(name[level:], package, level)
  
  File "<frozen importlib._bootstrap>", line 1006, in _gcd_import
  
  File "<frozen importlib._bootstrap>", line 983, in _find_and_load
  
  File "<frozen importlib._bootstrap>", line 967, in _find_and_load_unlocked
  
  File "<frozen importlib._bootstrap>", line 677, in _load_unlocked
  
  File "<frozen importlib._bootstrap_external>", line 728, in exec_module
  
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/tensorflow/__init__.py", line 24, in <module>
  
    check_extension('horovod.tensorflow', 'HOROVOD_WITH_TENSORFLOW', __file__, 'mpi_lib')
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 59, in check_extension
    ext_name, full_path, ext_env_var
  
ImportError: Extension horovod.tensorflow has not been built: /users/tongwenz/.local/lib/python3.7/site-packages/horovod/tensorflow/mpi_lib.cpython-37m-x86_64-linux-gnu.so not found
  
If this is not expected, reinstall Horovod with HOROVOD_WITH_TENSORFLOW=1 to debug the build error.
  
Extension tensorflow was NOT built with MPI.
  
Checking whether extension torch was built with MPI.
  
Extension horovod.torch has not been built: /users/tongwenz/.local/lib/python3.7/site-packages/horovod/torch/mpi_lib/_mpi_lib.cpython-37m-x86_64-linux-gnu.so not found
  
If this is not expected, reinstall Horovod with HOROVOD_WITH_PYTORCH=1 to debug the build error.
  
Warning! MPI libs are missing, but python applications are still available.
  
Traceback (most recent call last):
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 84, in _target_fn
  
    result = fn(ext)
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 139, in <lambda>
  
    built_fn = lambda ext: ext.mpi_built()
  
AttributeError: module 'horovod.torch' has no attribute 'mpi_built'
  
Extension torch was NOT built with MPI.
  
Checking whether extension mxnet was built with MPI.
  
Traceback (most recent call last):
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 83, in _target_fn
  
    ext = importlib.import_module('.' + ext_base_name, 'horovod')
  
  File "/appl/opt/anaconda3/2019.3/envs/default/lib/python3.7/importlib/__init__.py", line 127, in import_module
  
    return _bootstrap._gcd_import(name[level:], package, level)
  
  File "<frozen importlib._bootstrap>", line 1006, in _gcd_import
  
  File "<frozen importlib._bootstrap>", line 983, in _find_and_load
  
  File "<frozen importlib._bootstrap>", line 967, in _find_and_load_unlocked
  
  File "<frozen importlib._bootstrap>", line 677, in _load_unlocked
  
  File "<frozen importlib._bootstrap_external>", line 728, in exec_module
  
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/mxnet/__init__.py", line 19, in <module>
  
    __file__, 'mpi_lib')
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 59, in check_extension
  
    ext_name, full_path, ext_env_var
  
ImportError: Extension horovod.mxnet has not been built: /users/tongwenz/.local/lib/python3.7/site-packages/horovod/mxnet/mpi_lib.cpython-37m-x86_64-linux-gnu.so not found
  
If this is not expected, reinstall Horovod with HOROVOD_WITH_MXNET=1 to debug the build error.
  
Extension mxnet was NOT built with MPI.
  
Checking whether extension tensorflow was built with Gloo.
  
Traceback (most recent call last):
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 83, in _target_fn
  
    ext = importlib.import_module('.' + ext_base_name, 'horovod')
  
  File "/appl/opt/anaconda3/2019.3/envs/default/lib/python3.7/importlib/__init__.py", line 127, in import_module
  
    return _bootstrap._gcd_import(name[level:], package, level)
  
  File "<frozen importlib._bootstrap>", line 1006, in _gcd_import
  
  File "<frozen importlib._bootstrap>", line 983, in _find_and_load
  
  File "<frozen importlib._bootstrap>", line 967, in _find_and_load_unlocked
  
  File "<frozen importlib._bootstrap>", line 677, in _load_unlocked
  
  File "<frozen importlib._bootstrap_external>", line 728, in exec_module
  
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/tensorflow/__init__.py", line 24, in <module>
  
    check_extension('horovod.tensorflow', 'HOROVOD_WITH_TENSORFLOW', __file__, 'mpi_lib')
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 59, in check_extension
  
    ext_name, full_path, ext_env_var
  
ImportError: Extension horovod.tensorflow has not been built: /users/tongwenz/.local/lib/python3.7/site-packages/horovod/tensorflow/mpi_lib.cpython-37m-x86_64-linux-gnu.so not found
  
If this is not expected, reinstall Horovod with HOROVOD_WITH_TENSORFLOW=1 to debug the build error.
  
Extension tensorflow was NOT built with Gloo.
  
Checking whether extension torch was built with Gloo.
  
Extension horovod.torch has not been built: /users/tongwenz/.local/lib/python3.7/site-packages/horovod/torch/mpi_lib/_mpi_lib.cpython-37m-x86_64-linux-gnu.so not found
  
If this is not expected, reinstall Horovod with HOROVOD_WITH_PYTORCH=1 to debug the build error.
  
Warning! MPI libs are missing, but python applications are still available.
  
Traceback (most recent call last):
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 84, in _target_fn
  
    result = fn(ext)
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 150, in <lambda>
  
    built_fn = lambda ext: ext.gloo_built()
  
AttributeError: module 'horovod.torch' has no attribute 'gloo_built'
  
Extension torch was NOT built with Gloo.
  
Checking whether extension mxnet was built with Gloo.
  
Traceback (most recent call last):
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 83, in _target_fn
  
    ext = importlib.import_module('.' + ext_base_name, 'horovod')
  
  File "/appl/opt/anaconda3/2019.3/envs/default/lib/python3.7/importlib/__init__.py", line 127, in import_module
  
    return _bootstrap._gcd_import(name[level:], package, level)
  
  File "<frozen importlib._bootstrap>", line 1006, in _gcd_import
  
  File "<frozen importlib._bootstrap>", line 983, in _find_and_load
  
  File "<frozen importlib._bootstrap>", line 967, in _find_and_load_unlocked
  
  File "<frozen importlib._bootstrap>", line 677, in _load_unlocked
  
  File "<frozen importlib._bootstrap_external>", line 728, in exec_module
  
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/mxnet/__init__.py", line 19, in <module>
    __file__, 'mpi_lib')
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 59, in check_extension
    ext_name, full_path, ext_env_var
  
ImportError: Extension horovod.mxnet has not been built: /users/tongwenz/.local/lib/python3.7/site-packages/horovod/mxnet/mpi_lib.cpython-37m-x86_64-linux-gnu.so not found
  
If this is not expected, reinstall Horovod with HOROVOD_WITH_MXNET=1 to debug the build error.
  
Extension mxnet was NOT built with Gloo.
  
Checking whether extension tensorflow was built with NCCL.
  
Traceback (most recent call last):
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 83, in _target_fn
  
    ext = importlib.import_module('.' + ext_base_name, 'horovod')
  
  File "/appl/opt/anaconda3/2019.3/envs/default/lib/python3.7/importlib/__init__.py", line 127, in import_module
  
    return _bootstrap._gcd_import(name[level:], package, level)
  
  File "<frozen importlib._bootstrap>", line 1006, in _gcd_import
  
  File "<frozen importlib._bootstrap>", line 983, in _find_and_load
  
  File "<frozen importlib._bootstrap>", line 967, in _find_and_load_unlocked
  
  File "<frozen importlib._bootstrap>", line 677, in _load_unlocked
  
  File "<frozen importlib._bootstrap_external>", line 728, in exec_module
  
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/tensorflow/__init__.py", line 24, in <module>
  
    check_extension('horovod.tensorflow', 'HOROVOD_WITH_TENSORFLOW', __file__, 'mpi_lib')
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 59, in check_extension
    ext_name, full_path, ext_env_var
  
ImportError: Extension horovod.tensorflow has not been built: /users/tongwenz/.local/lib/python3.7/site-packages/horovod/tensorflow/mpi_lib.cpython-37m-x86_64-linux-gnu.so not found
  
If this is not expected, reinstall Horovod with HOROVOD_WITH_TENSORFLOW=1 to debug the build error.
  
Extension tensorflow was NOT built with NCCL.
  
Checking whether extension torch was built with NCCL.
  
Extension horovod.torch has not been built: /users/tongwenz/.local/lib/python3.7/site-packages/horovod/torch/mpi_lib/_mpi_lib.cpython-37m-x86_64-linux-gnu.so not found
  
If this is not expected, reinstall Horovod with HOROVOD_WITH_PYTORCH=1 to debug the build error.
  
Warning! MPI libs are missing, but python applications are still available.
  
Traceback (most recent call last):
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 84, in _target_fn
  
    result = fn(ext)
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 160, in <lambda>
  
    built_fn = lambda ext: ext.nccl_built()
  
AttributeError: module 'horovod.torch' has no attribute 'nccl_built'
  
Extension torch was NOT built with NCCL.
  
Checking whether extension mxnet was built with NCCL.
  
Traceback (most recent call last):
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 83, in _target_fn
  
    ext = importlib.import_module('.' + ext_base_name, 'horovod')
  
  File "/appl/opt/anaconda3/2019.3/envs/default/lib/python3.7/importlib/__init__.py", line 127, in import_module
  
    return _bootstrap._gcd_import(name[level:], package, level)
  
  File "<frozen importlib._bootstrap>", line 1006, in _gcd_import
  
  File "<frozen importlib._bootstrap>", line 983, in _find_and_load
  
  File "<frozen importlib._bootstrap>", line 967, in _find_and_load_unlocked
  
  File "<frozen importlib._bootstrap>", line 677, in _load_unlocked
  
  File "<frozen importlib._bootstrap_external>", line 728, in exec_module
  
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/mxnet/__init__.py", line 19, in <module>
    __file__, 'mpi_lib')
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 59, in check_extension
    ext_name, full_path, ext_env_var
  
ImportError: Extension horovod.mxnet has not been built: /users/tongwenz/.local/lib/python3.7/site-packages/horovod/mxnet/mpi_lib.cpython-37m-x86_64-linux-gnu.so not found
  
If this is not expected, reinstall Horovod with HOROVOD_WITH_MXNET=1 to debug the build error.
  
Extension mxnet was NOT built with NCCL.
  
Checking whether extension tensorflow was built with DDL.
  
Traceback (most recent call last):
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 83, in _target_fn
  
    ext = importlib.import_module('.' + ext_base_name, 'horovod')
  
  File "/appl/opt/anaconda3/2019.3/envs/default/lib/python3.7/importlib/__init__.py", line 127, in import_module
  
    return _bootstrap._gcd_import(name[level:], package, level)
  
  File "<frozen importlib._bootstrap>", line 1006, in _gcd_import
  
  File "<frozen importlib._bootstrap>", line 983, in _find_and_load
  
  File "<frozen importlib._bootstrap>", line 967, in _find_and_load_unlocked
  
  File "<frozen importlib._bootstrap>", line 677, in _load_unlocked
  
  File "<frozen importlib._bootstrap_external>", line 728, in exec_module
  
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/tensorflow/__init__.py", line 24, in <module>
  
    check_extension('horovod.tensorflow', 'HOROVOD_WITH_TENSORFLOW', __file__, 'mpi_lib')
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 59, in check_extension
    ext_name, full_path, ext_env_var
  
ImportError: Extension horovod.tensorflow has not been built: /users/tongwenz/.local/lib/python3.7/site-packages/horovod/tensorflow/mpi_lib.cpython-37m-x86_64-linux-gnu.so not found
  
If this is not expected, reinstall Horovod with HOROVOD_WITH_TENSORFLOW=1 to debug the build error.
  
Extension tensorflow was NOT built with DDL.
  
Checking whether extension torch was built with DDL.
  
Extension horovod.torch has not been built: /users/tongwenz/.local/lib/python3.7/site-packages/horovod/torch/mpi_lib/_mpi_lib.cpython-37m-x86_64-linux-gnu.so not found
  
If this is not expected, reinstall Horovod with HOROVOD_WITH_PYTORCH=1 to debug the build error.
  
Warning! MPI libs are missing, but python applications are still available.
  
Traceback (most recent call last):
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 84, in _target_fn
  
    result = fn(ext)
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 170, in <lambda>
  
    built_fn = lambda ext: ext.ddl_built()
  
AttributeError: module 'horovod.torch' has no attribute 'ddl_built'
  
Extension torch was NOT built with DDL.
  
Checking whether extension mxnet was built with DDL.
  
Traceback (most recent call last):
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 83, in _target_fn
  
    ext = importlib.import_module('.' + ext_base_name, 'horovod')
  
  File "/appl/opt/anaconda3/2019.3/envs/default/lib/python3.7/importlib/__init__.py", line 127, in import_module
  
    return _bootstrap._gcd_import(name[level:], package, level)
  
  File "<frozen importlib._bootstrap>", line 1006, in _gcd_import
  
  File "<frozen importlib._bootstrap>", line 983, in _find_and_load
  
  File "<frozen importlib._bootstrap>", line 967, in _find_and_load_unlocked
  
  File "<frozen importlib._bootstrap>", line 677, in _load_unlocked
  
  File "<frozen importlib._bootstrap_external>", line 728, in exec_module
  
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/mxnet/__init__.py", line 19, in <module>
    __file__, 'mpi_lib')
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 59, in check_extension
    ext_name, full_path, ext_env_var
  
ImportError: Extension horovod.mxnet has not been built: /users/tongwenz/.local/lib/python3.7/site-packages/horovod/mxnet/mpi_lib.cpython-37m-x86_64-linux-gnu.so not found
  
If this is not expected, reinstall Horovod with HOROVOD_WITH_MXNET=1 to debug the build error.
  
Extension mxnet was NOT built with DDL.
  
Checking whether extension tensorflow was built with CCL.
  
Traceback (most recent call last):
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 83, in _target_fn
  
    ext = importlib.import_module('.' + ext_base_name, 'horovod')
  
  File "/appl/opt/anaconda3/2019.3/envs/default/lib/python3.7/importlib/__init__.py", line 127, in import_module
  
    return _bootstrap._gcd_import(name[level:], package, level)
  
  File "<frozen importlib._bootstrap>", line 1006, in _gcd_import
  
  File "<frozen importlib._bootstrap>", line 983, in _find_and_load
  
  File "<frozen importlib._bootstrap>", line 967, in _find_and_load_unlocked
  
  File "<frozen importlib._bootstrap>", line 677, in _load_unlocked
  
  File "<frozen importlib._bootstrap_external>", line 728, in exec_module
  
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/tensorflow/__init__.py", line 24, in <module>
  
    check_extension('horovod.tensorflow', 'HOROVOD_WITH_TENSORFLOW', __file__, 'mpi_lib')
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 59, in check_extension
    ext_name, full_path, ext_env_var
  
ImportError: Extension horovod.tensorflow has not been built: /users/tongwenz/.local/lib/python3.7/site-packages/horovod/tensorflow/mpi_lib.cpython-37m-x86_64-linux-gnu.so not found
  
If this is not expected, reinstall Horovod with HOROVOD_WITH_TENSORFLOW=1 to debug the build error.
  
Extension tensorflow was NOT built with CCL.
  
Checking whether extension torch was built with CCL.
  
Extension horovod.torch has not been built: /users/tongwenz/.local/lib/python3.7/site-packages/horovod/torch/mpi_lib/_mpi_lib.cpython-37m-x86_64-linux-gnu.so not found
  
If this is not expected, reinstall Horovod with HOROVOD_WITH_PYTORCH=1 to debug the build error.
  
Warning! MPI libs are missing, but python applications are still available.
  
Traceback (most recent call last):
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 84, in _target_fn
  
    result = fn(ext)
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 180, in <lambda>
  
    built_fn = lambda ext: ext.ccl_built()
  
AttributeError: module 'horovod.torch' has no attribute 'ccl_built'
  
Extension torch was NOT built with CCL.
  
Checking whether extension mxnet was built with CCL.
  
Traceback (most recent call last):
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 83, in _target_fn
  
    ext = importlib.import_module('.' + ext_base_name, 'horovod')
  
  File "/appl/opt/anaconda3/2019.3/envs/default/lib/python3.7/importlib/__init__.py", line 127, in import_module
  
    return _bootstrap._gcd_import(name[level:], package, level)
  
  File "<frozen importlib._bootstrap>", line 1006, in _gcd_import
  
  File "<frozen importlib._bootstrap>", line 983, in _find_and_load
  
  File "<frozen importlib._bootstrap>", line 967, in _find_and_load_unlocked
  
  File "<frozen importlib._bootstrap>", line 677, in _load_unlocked
  
  File "<frozen importlib._bootstrap_external>", line 728, in exec_module
  
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/mxnet/__init__.py", line 19, in <module>
    __file__, 'mpi_lib')
  
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 59, in check_extension
    ext_name, full_path, ext_env_var
  
ImportError: Extension horovod.mxnet has not been built: /users/tongwenz/.local/lib/python3.7/site-packages/horovod/mxnet/mpi_lib.cpython-37m-x86_64-linux-gnu.so not found
  
If this is not expected, reinstall Horovod with HOROVOD_WITH_MXNET=1 to debug the build error.
  
Extension mxnet was NOT built with CCL.

Horovod v0.25.0:

Available Frameworks:
  
    [ ] TensorFlow
  
    [X] PyTorch
  
    [ ] MXNet
  

Available Controllers:
  
    [ ] MPI
  
    [ ] Gloo
  

Available Tensor Operations:
  
    [ ] NCCL
  
    [ ] DDL
  
    [ ] CCL
  
    [ ] MPI
  
    [ ] Gloo  
  
## Commands I tried:
  > HOROVOD_WITH_TENSORFLOW=1 pip install --user horovod[tensorflow,keras]
  
Requirement already satisfied: horovod[keras,tensorflow] in ./.local/lib/python3.7/site-packages (0.25.0)
  
Requirement already satisfied: cloudpickle in /appl/opt/anaconda3/2019.3/envs/default/lib/python3.7/site-packages (from horovod[keras,tensorflow]) (0.8.0)
  
Requirement already satisfied: pyyaml in /appl/opt/anaconda3/2019.3/envs/default/lib/python3.7/site-packages (from horovod[keras,tensorflow]) (5.1)
  
Requirement already satisfied: psutil in /appl/opt/anaconda3/2019.3/envs/default/lib/python3.7/site-packages (from horovod[keras,tensorflow]) (5.6.1)
  
Requirement already satisfied: cffi>=1.4.0 in /appl/opt/anaconda3/2019.3/envs/default/lib/python3.7/site-packages (from horovod[keras,tensorflow]) (1.12.2)
  
Requirement already satisfied: keras!=2.0.9,!=2.1.0,!=2.1.1,>=2.0.8; extra == "keras" in ./.local/lib/python3.7/site-packages (from horovod[keras,tensorflow]) (2.9.0)
  
Requirement already satisfied: tensorflow; extra == "tensorflow" in ./.local/lib/python3.7/site-packages (from horovod[keras,tensorflow]) (2.9.1)
  
Requirement already satisfied: pycparser in /appl/opt/anaconda3/2019.3/envs/default/lib/python3.7/site-packages (from cffi>=1.4.0->horovod[keras,tensorflow]) (2.19)
  
Requirement already satisfied: typing-extensions>=3.6.6 in ./.local/lib/python3.7/site-packages (from tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (4.3.0)
  
Requirement already satisfied: grpcio<2.0,>=1.24.3 in ./.local/lib/python3.7/site-packages (from tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (1.47.0)
  
Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in ./.local/lib/python3.7/site-packages (from tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (0.26.0)
  
Collecting numpy>=1.20
  
  Using cached https://files.pythonhosted.org/packages/6d/ad/ff3b21ebfe79a4d25b4a4f8e5cf9fd44a204adb6b33c09010f566f51027a/numpy-1.21.6-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl
  
Requirement already satisfied: packaging in /appl/opt/anaconda3/2019.3/envs/default/lib/python3.7/site-packages (from tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (19.0)
  
Requirement already satisfied: tensorboard<2.10,>=2.9 in ./.local/lib/python3.7/site-packages (from tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (2.9.1)
  
Requirement already satisfied: gast<=0.4.0,>=0.2.1 in ./.local/lib/python3.7/site-packages (from tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (0.4.0)
  
Requirement already satisfied: flatbuffers<2,>=1.12 in ./.local/lib/python3.7/site-packages (from tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (1.12)
  
Requirement already satisfied: absl-py>=1.0.0 in ./.local/lib/python3.7/site-packages (from tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (1.2.0)
  
Requirement already satisfied: google-pasta>=0.1.1 in ./.local/lib/python3.7/site-packages (from tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (0.2.0)
  
Requirement already satisfied: tensorflow-estimator<2.10.0,>=2.9.0rc0 in ./.local/lib/python3.7/site-packages (from tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (2.9.0)
  
Requirement already satisfied: opt-einsum>=2.3.2 in ./.local/lib/python3.7/site-packages (from tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (3.3.0)
  
Requirement already satisfied: astunparse>=1.6.0 in ./.local/lib/python3.7/site-packages (from tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (1.6.3)
  
Requirement already satisfied: protobuf<3.20,>=3.9.2 in ./.local/lib/python3.7/site-packages (from tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (3.19.4)
  
Requirement already satisfied: six>=1.12.0 in /appl/opt/anaconda3/2019.3/envs/default/lib/python3.7/site-packages (from tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (1.12.0)
  
Requirement already satisfied: setuptools in /appl/opt/anaconda3/2019.3/envs/default/lib/python3.7/site-packages (from tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (40.8.0)
  
Requirement already satisfied: keras-preprocessing>=1.1.1 in ./.local/lib/python3.7/site-packages (from tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (1.1.2)
  
Requirement already satisfied: h5py>=2.9.0 in /appl/opt/anaconda3/2019.3/envs/default/lib/python3.7/site-packages (from tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (2.9.0)
  
Requirement already satisfied: wrapt>=1.11.0 in /appl/opt/anaconda3/2019.3/envs/default/lib/python3.7/site-packages (from tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (1.11.1)
  
Requirement already satisfied: libclang>=13.0.0 in ./.local/lib/python3.7/site-packages (from tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (14.0.6)
  
Requirement already satisfied: termcolor>=1.1.0 in ./.local/lib/python3.7/site-packages (from tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (1.1.0)
  
Requirement already satisfied: pyparsing>=2.0.2 in /appl/opt/anaconda3/2019.3/envs/default/lib/python3.7/site-packages (from packaging->tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (2.3.1)
  
Requirement already satisfied: requests<3,>=2.21.0 in /appl/opt/anaconda3/2019.3/envs/default/lib/python3.7/site-packages (from tensorboard<2.10,>=2.9->tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (2.21.0)
  
Collecting werkzeug>=1.0.1
  
  Using cached https://files.pythonhosted.org/packages/c8/27/be6ddbcf60115305205de79c29004a0c6bc53cec814f733467b1bb89386d/Werkzeug-2.2.2-py3-none-any.whl
Requirement already satisfied: google-auth<3,>=1.6.3 in ./.local/lib/python3.7/site-packages (from tensorboard<2.10,>=2.9->tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (2.10.0)
  
Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in ./.local/lib/python3.7/site-packages (from tensorboard<2.10,>=2.9->tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (0.4.6)
  
Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in ./.local/lib/python3.7/site-packages (from tensorboard<2.10,>=2.9->tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (0.6.1)
  
Requirement already satisfied: markdown>=2.6.8 in ./.local/lib/python3.7/site-packages (from tensorboard<2.10,>=2.9->tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (3.4.1)
  
Requirement already satisfied: wheel>=0.26 in /appl/opt/anaconda3/2019.3/envs/default/lib/python3.7/site-packages (from tensorboard<2.10,>=2.9->tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (0.33.1)
  
Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in ./.local/lib/python3.7/site-packages (from tensorboard<2.10,>=2.9->tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (1.8.1)
  
Requirement already satisfied: urllib3<1.25,>=1.21.1 in /appl/opt/anaconda3/2019.3/envs/default/lib/python3.7/site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (1.24.1)
  
Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /appl/opt/anaconda3/2019.3/envs/default/lib/python3.7/site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (3.0.4)
  
Requirement already satisfied: idna<2.9,>=2.5 in /appl/opt/anaconda3/2019.3/envs/default/lib/python3.7/site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (2.8)
  
Requirement already satisfied: certifi>=2017.4.17 in /appl/opt/anaconda3/2019.3/envs/default/lib/python3.7/site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (2019.3.9)
  
Collecting MarkupSafe>=2.1.1
  
  Using cached https://files.pythonhosted.org/packages/9f/83/b221ce5a0224f409b9f02b0dc6cb0b921c46033f4870d64fa3e8a96af701/MarkupSafe-2.1.1-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  
Requirement already satisfied: cachetools<6.0,>=2.0.0 in ./.local/lib/python3.7/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.10,>=2.9->tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (5.2.0)
  
Requirement already satisfied: rsa<5,>=3.1.4; python_version >= "3.6" in ./.local/lib/python3.7/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.10,>=2.9->tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (4.9)
  
Requirement already satisfied: pyasn1-modules>=0.2.1 in ./.local/lib/python3.7/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.10,>=2.9->tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (0.2.8)
  
Requirement already satisfied: requests-oauthlib>=0.7.0 in ./.local/lib/python3.7/site-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.10,>=2.9->tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (1.3.1)
  
Collecting importlib-metadata>=4.4; python_version < "3.10"
  
  Using cached https://files.pythonhosted.org/packages/d2/a2/8c239dc898138f208dd14b441b196e7b3032b94d3137d9d8453e186967fc/importlib_metadata-4.12.0-py3-none-any.whl
                                                          
Requirement already satisfied: pyasn1>=0.1.3 in ./.local/lib/python3.7/site-packages (from rsa<5,>=3.1.4; python_version >= "3.6"->google-auth<3,>=1.6.3->tensorboard<2.10,>=2.9->tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (0.4.8)
  
Requirement already satisfied: oauthlib>=3.0.0 in ./.local/lib/python3.7/site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.10,>=2.9->tensorflow; extra == "tensorflow"->horovod[keras,tensorflow]) (3.2.0)
  
Collecting zipp>=0.5
  
  Using cached https://files.pythonhosted.org/packages/f0/36/639d6742bcc3ffdce8b85c31d79fcfae7bb04b95f0e5c4c6f8b206a038cc/zipp-3.8.1-py3-none-any.whl
  
**ERROR: virtualenv 20.16.3 has requirement filelock<4,>=3.4.1, but you'll have filelock 3.0.10 which is incompatible.**
  
**ERROR: tensorflow-tensorboard 1.5.1 has requirement bleach==1.5.0, but you'll have bleach 3.1.0 which is incompatible.**
  
**ERROR: tensorflow-tensorboard 1.5.1 has requirement html5lib==0.9999999, but you'll have html5lib 1.0.1 which is incompatible.**
  
**ERROR: tensorboard 2.9.1 has requirement setuptools>=41.0.0, but you'll have setuptools 40.8.0 which is incompatible.**
  
Installing collected packages: numpy, MarkupSafe, werkzeug, zipp, importlib-metadata
  
Successfully installed MarkupSafe-2.1.1 importlib-metadata-4.12.0 numpy-1.21.6 werkzeug-2.2.2 zipp-3.8.1
  
## I guess the reason is because I have the incompatible versions of packages? So I tried reinstall and upgrade, but none of these works:
  
> pip install --user setuptools --upgrade
  
> pip install --user --upgrade tensorflow-gpu
  
> pip install --user --upgrade tensorflow
  
## They just keep showing me there are more conflicting versions of other packages. Some links I consulted, but none of these methods work for me.
  
ERROR: tensorboard 2.0.2 has requirement setuptools>=41.0.0, but you'll have setuptools 40.6.2 which is incompatible: https://stackoverflow.com/questions/59104396/error-tensorboard-2-0-2-has-requirement-setuptools-41-0-0-but-youll-have-set
https://stackoverflow.com/questions/50195901/how-to-resolve-bleach-1-5-0-html5lib-0-9999999-on-windows10
  

  
