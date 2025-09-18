import os
import f90nml

class Case:
    def __init__(self,
                 name: str, n: int, m: int, k: int,
                 row: bool, print_summary: bool, print_array: bool):
        self.name: str = name
        self.n: int = n
        self.m: int = m
        self.k: int = k
        self.row: bool = row
        self.print_summary: bool = print_summary
        self.print_array: bool = print_array

    def make_nml(self, dst:str, group_name: str = "params"):
        print(f'Saving case={self.name} -> {dst}')
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

def compile(clean:bool=True):
    print('Compiling FORTRAN code')

    
    app = 'app'
    src = 'src'

    build = 'build'

    bin = 'bin'
    binary_name = 'hw1'
    binary_path = os.path.join(bin, binary_name)
    
    os.makedirs(build, exist_ok=True)
    os.makedirs(bin, exist_ok=True)

    if clean:
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
    modules = ['input_arrays.f90', 'output_array.f90', 'mem_util.f90', 'path_util.f90']

    # modules
    o_list = []
    for mod in modules:
        in_path = os.path.join(src, mod)
        out_path = os.path.join(build, mod.replace('f90', 'o'))
        o_list.append(out_path)
        print(f'    {in_path} -> {out_path}')
        os.system(f'gfortran -O3 -std=f2018 -c {in_path} -J build -o {out_path}')

    # main app
    print('Compiling main app')
    

    main_in = 'main.f90'
    main_out = main_in.replace('f90', 'o')

    in_path = os.path.join(app, main_in)
    out_path = os.path.join(build, main_out)
    o_list.append(out_path)

    print(f'    {in_path} -> {out_path}')
    os.system(f'gfortran -O3 -std=f2018 -c {in_path} -I {build} -o {out_path}')
    
    # binary
    print('Compiling binary')
    files_str = ' '.join(o_list)
    
    print(f'    {files_str} -> {binary_path}')
    os.system(f'gfortran -O3 -std=f2018 {files_str} -o {binary_path}')
    print('Done compiling \n')
    return binary_path

def main():
    binary_path = compile()

    case_a = Case(name="a", n=100,   m=50,    k=44,    row=False, print_summary=True, print_array=True)
    case_b = Case(name="b", n=1000,  m=50,    k=88,    row=False, print_summary=True, print_array=True)
    case_c = Case(name="c", n=25000, m=12345, k=12346, row=False, print_summary=True, print_array=False)
    case_d = Case(name="d", n=90000, m=12345, k=12346, row=False, print_summary=True, print_array=False)
    case_q6 = Case(name="q6", n=5, m=3, k=2, row=True, print_summary=True, print_array=True)


    cases = [
        case_a,
        case_b,
        case_c,
        case_q6,
        case_d # this case requires a lot of ram
        ]

    for case in cases:
        directory = f"./cases/{case.name}/"
        params_dst = os.path.join(directory, "params.nml")
        # time_dst = os.path.join(directory, "time.txt")

        case.make_nml(dst=params_dst)
        time_file = os.path.join(directory, "time.txt")

        # Run with enhanced timing and append to summary.txt
        cmd = f'/usr/bin/time -p -o {time_file} -- {binary_path} {params_dst} {directory}'
        print(f"Running: {cmd}\n")

        # # Add timing header to summary file
        # with open(summary_file, 'a') as f:
        #     f.write(f"\n=== TIMING RESULTS FOR CASE {case.name.upper()} ===\n")

        os.system(cmd)

        # Add timing footer
        # with open(summary_file, 'a') as f:
        #     f.write("=== END TIMING ===\n\n")

        # input()

    


if __name__ == "__main__":
    main()