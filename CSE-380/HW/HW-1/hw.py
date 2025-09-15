import os
import f90nml

class Case:
    def __init__(self, name: str, n: int, m: int, k: int, row: bool):
        self.name: str = name
        self.n: int = n
        self.m: int = m
        self.k: int = k
        self.row: bool = row

    def make_nml(self, dst:str, group_name: str = "params"):
        print(f'Saving case={self.name} -> {dst}')
        nml = {
            "params": {
                "n": self.n,
                "m": self.m,
                "k": self.k,
                "row": self.row
            }   
        }

        os.makedirs(os.path.dirname(dst), exist_ok=True)

        if os.path.exists(dst):
            os.remove(dst)

        f90nml.write(nml, dst)

def compile():
    print('Compiling FORTRAN code')
    os.makedirs('./build', exist_ok=True)
    os.makedirs('./bin', exist_ok=True)

    print('Compiling modules')
    src = 'src'
    build = 'build'
    modules = ['mem.f90', 'path.f90']

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
    app = 'app'

    main_in = 'main.f90'
    main_out = main_in.replace('f90', 'o')

    in_path = os.path.join(app, main_in)
    out_path = os.path.join(build, main_out)
    o_list.append(out_path)

    print(f'    {in_path} -> {out_path}')
    os.system(f'gfortran -O3 -std=f2018 -c {in_path} -I {build} -o {out_path}')
    
    # binary
    print('Compiling binary')

    bin = 'bin'
    binary_name = 'hw1'
    binary_path = os.path.join(bin, binary_name)

    files_str = ' '.join(o_list)
    
    print(f'    {files_str} -> {binary_path}')
    os.system(f'gfortran -O3 -std=f2018 {files_str} -o {binary_path}')
    print('Done compiling \n')
    return binary_path

def main():
    binary_path = compile()

    case_a = Case(name="a", n=100,   m=50,    k=44, row=False)
    case_b = Case(name="b", n=1000,  m=50,    k=88, row=False)
    case_c = Case(name="c", n=25000, m=12345, k=12346, row=False)
    case_d = Case(name="d", n=90000, m=12345, k=12346, row=False)


    cases = [
        case_a,
        case_b,
        case_c,
        case_d
        ]
    for case in cases:
        dst = f"./cases/{case.name}/params.nml"
        case.make_nml(dst=dst)
        output_dir = os.path.dirname(dst)
        # print(output_dir)
        cmd = f'{binary_path} {dst} {output_dir}'
        print(cmd + '\n')
        os.system(cmd)

    


if __name__ == "__main__":
    main()