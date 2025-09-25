import os
import f90nml
import time
import subprocess
import argparse

class Case:
    def __init__(self,
                 name: str, n: int, m: int, k: int,
                 row: bool=False, print_summary: bool=False, print_array: bool=False):
        self.name: str = name
        self.n: int = n
        self.m: int = m
        self.k: int = k
        self.row: bool = row
        self.print_summary: bool = print_summary
        self.print_array: bool = print_array

        # input validation (done in main.f90)
        # assert m > 0
        # assert n > 0
        # assert k > 0
        # assert m <= n
        # assert k <= n
        # assert k <= m

    def make_nml(self, dst:str, group_name: str = "params"):
        print(f'\nSaving case={self.name} -> {dst}')
        nml = {
            "params": {
                "n": self.n,
                "m": self.m,
                "k": self.k,
                "row": self.row,
                "print_summary": self.print_summary,
                "print_array": self.print_array
            }   
        }

        os.makedirs(os.path.dirname(dst), exist_ok=True)

        if os.path.exists(dst):
            os.remove(dst)

        f90nml.write(nml, dst)

def compile(clean: bool = True, enable_par:bool=False):
    print('Compiling FORTRAN code')

    gfortran_base = ['gfortran', '-O3', '-std=f2018'] # base gfortran compiling settings
    
    app = 'app'
    src = 'src'
    build = 'build'
    bin = 'bin'

    binary_name = 'hw1'
    binary_path = os.path.join(bin, binary_name)
    
    os.makedirs(build, exist_ok=True)
    os.makedirs(bin, exist_ok=True)

    if clean: # then clear out files in build, src, and bin
        print(f'Cleaning: {build} {src} {bin}')
        for file in os.listdir(build):
           if file.endswith('.mod') or file.endswith('.o'):
               os.remove(os.path.join(build, file))

        for file in os.listdir(src):
            if file.endswith('.mod') or file.endswith('.o'):
                os.remove(os.path.join(src, file))

        if os.path.exists(binary_path):
            os.remove(binary_path)

    print('Compiling modules')
    # list of modules in /src/
    modules = ['input_arrays.f90', 'output_array.f90', 'mem_util.f90', 'path_util.f90']

    # compile modules into intermediate files
    o_list = []
    for mod in modules:
        in_path = os.path.join(src, mod)
        out_path = os.path.join(build, mod.replace('f90', 'o'))
        o_list.append(out_path)
        print(f'    {in_path} -> {out_path}')
        if enable_par:
            subprocess.run(gfortran_base + ['-c', in_path, '-J', 'build', '-o', out_path, '-fopenmp'], check=True)
        else:
            subprocess.run(gfortran_base + ['-c', in_path, '-J', 'build', '-o', out_path], check=True)

    # compile main app inot intermediate file
    print('Compiling main app')
    main_in = 'main.f90'
    main_out = main_in.replace('f90', 'o')

    in_path = os.path.join(app, main_in)
    out_path = os.path.join(build, main_out)
    o_list = [out_path] + o_list

    print(f'    {in_path} -> {out_path}')
    if enable_par:
        subprocess.run(gfortran_base + ['-c', in_path, '-I', build, '-fopenmp', '-o', out_path], check=True)
    else:
        subprocess.run(gfortran_base + ['-c', in_path, '-I', build, '-o', out_path], check=True)

    # compile binary using intermediate files
    print('Compiling binary')
    files_str = ' '.join(o_list)
    
    print(f'    {files_str} -> {binary_path}')
    if enable_par:
        subprocess.run(gfortran_base + o_list + ['-o', binary_path, '-fopenmp'], check=True)
    else:
        subprocess.run(gfortran_base + o_list + ['-o', binary_path], check=True)

    print('Done compiling \n')
    return binary_path

def main():
    # toggle if to compile
    # usage python hw.py --no-compile -> set run_compilation=False
    p = argparse.ArgumentParser()
    p.add_argument('--no-compile', action='store_true')
    p.add_argument('--enable-par', action='store_true')
    
    args = p.parse_args()
    if not args.no_compile:
        if args.enable_par:
            print('Building with Parallel')
            binary_path = compile(enable_par=True)
        else:
            print('Building with Single')
            binary_path = compile(enable_par=False)
    else:
        binary_path = './bin/hw1'
        
    # hw cases
    case_a = Case(name="a", n=100,   m=50,    k=44,    row=False, print_summary=True, print_array=False)
    case_b = Case(name="b", n=1000,  m=50,    k=88,    row=False, print_summary=True, print_array=False)
    case_c = Case(name="c", n=25000, m=12345, k=12346, row=False, print_summary=True, print_array=False)
    case_d = Case(name="d", n=90000, m=12345, k=12346, row=False, print_summary=True, print_array=False)

    case_q6_row = Case(name="q6_row", n=5, m=3, k=2, row=True, print_summary=True, print_array=True)
    case_q6_col = Case(name="q6_col", n=5, m=3, k=2, row=False, print_summary=True, print_array=True)

    # expected fails
    fail_1 = Case(name="fail_1", n=-1,  m=10, k=10)
    fail_2 = Case(name="fail_2", n=10,  m=-1, k=10)
    fail_3 = Case(name="fail_3", n=10,  m=10, k=-1)
    fail_4 = Case(name="fail_4", n=10,  m=11, k=10)
    fail_5 = Case(name="fail_5", n=10,  m=10, k=11)

    cases = [
        case_a,
        case_b,
        case_c,
        case_d, # this case requires a lot of ram

        case_q6_row,
        case_q6_col,

        # expected fails
        fail_1,
        fail_2,
        fail_3,
        fail_4,
        fail_5,
        ]

    for case in cases:
        directory = f"./cases/{case.name}/"
        params_dst = os.path.join(directory, "params.nml")

        case.make_nml(dst=params_dst)
        time_file = os.path.join(directory, "time.txt")

        # Run with enhanced timing and append to summary.txt
        cmd = ['/usr/bin/time', '-p', '-o', time_file, '--', binary_path, params_dst, directory]
        cmd = f'/usr/bin/time -p -o {time_file} -- {binary_path} {params_dst} {directory}'
        print(f"Running: {cmd}\n")

        # old simple way
        # os.system(cmd)

        start = time.perf_counter() # start time
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        end = time.perf_counter() # end time
        elapsed = end - start # elapsed time for subprocess
        
        
        if result.stderr:
            print(result.stdout + result.stderr)
        else:
            print(result.stdout)

        # append the timed subprocess execution to the time.txt file
        with open(time_file, 'a') as f:
            f.write(f'\nSubprocess time: {elapsed:.2E} [s]')

    


if __name__ == "__main__":
    main()