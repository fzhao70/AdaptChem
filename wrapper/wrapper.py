import torch
import sys
import subprocess
from torch.utils.cpp_extension import include_paths, library_paths

def create_dynamic_library():
    """
    A simple function to create a dynamic library to call PyTorch model in the fortran code
    The function compiles the wrapper.cpp file into a shared library libtorch_fortran.so
    You can also use the FTorch library to load the model in Fortran if you need more flexibility and features.
    
    To use that in the Fortran code, you can use the following code:
    ```
    use iso_c_binding
    real(c_float), dimension(:), allocatable :: input_data, output_data
    integer(c_int64_t), dimension(:), allocatable :: input_dims, output_dims
    !Load the model
    model = load_model('scripted_model.pt'//c_null_char)
    !Call the forward function
    call forward(model, input_data, input_dims, size(input_dims, kind=c_int64_t), &
            output_data, output_dims, size(output_dims, kind=c_int64_t))
    !Destroy the model when done
    call destroy_model(model)
    ```
    """

    # Get include and library paths
    include_dirs = include_paths()
    library_dirs = library_paths()

    include_dirs_flags = ' '.join(['-I' + dir for dir in include_dirs])
    library_dirs_flags = ' '.join(['-L' + dir for dir in library_dirs])

    # Libraries to link against
    libraries = ['torch', 'torch_cpu', 'c10']
    libraries_flags = ' '.join(['-l' + lib for lib in libraries])
    rpath_flags = ' '.join(['-Wl,-rpath,' + dir for dir in library_dirs])

    # Compile the C++ code into a shared library
    cpp_files = 'wrapper.cpp'
    compile_command = f'g++ -std=c++14 -fPIC -shared {cpp_files} -o libtorch_fortran.so ' \
                    f'{include_dirs_flags} {library_dirs_flags} {libraries_flags} {rpath_flags}'

    subprocess.run(compile_command, shell=True, check=True)
    print(f"Dynamic library libtorch_fortran.so has been created successfully.")


def save_model(model, path):
    """
    Save a PyTorch model to a Torchscipt file
    So that it can be loaded in the Fortran code
    """
    torch.jit.script(model).save(path)