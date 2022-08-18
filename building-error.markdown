I cannot build Horovod with other needed frameworks:
command for installing Horovod:
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
  File "/users/tongwenz/.local/lib/python3.7/site-packages/horovod/common/util.py", line 59, in check_extension
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
