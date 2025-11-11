import os
import subprocess
import sys
import time


def exec_case(case_name, case_dir, command, cwd, env_vars=None):
    """Execute a single test case with timing and live output.

    Args:
        case_name: Name of the case (e.g., "Case 1")
        case_dir: Directory path for the case
        command: Command to execute as a list
        cwd: Original working directory to return to
        env_vars: Optional dict of environment variables to set for this command
    """
    print(f'\n{"="*60}')
    print(f'Running {case_name}')
    print(f'{"="*60}')

    os.makedirs(case_dir, exist_ok=True)
    os.chdir(case_dir)

    print(f'Command: {" ".join(str(c) for c in command)}')
    print(f'Working directory: {case_dir}')
    if env_vars:
        print(f'Environment: {", ".join(f"{k}={v}" for k, v in env_vars.items())}')
    print(f'{"-"*60}\n')

    start_time = time.time()

    # Prepare environment
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    # Redirect stdout to file
    with open('./out.txt', 'w') as f:
        subprocess.run(command, check=True, text=True, stdout=f, stderr=subprocess.STDOUT, env=env)

    elapsed_time = time.time() - start_time

    # Append timing information to the stdout file
    with open('./out.txt', 'a') as f:
        f.write(f'\n{"-"*60}\n')
        f.write(f'{case_name} completed in {elapsed_time:.2f} seconds\n')
        f.write(f'{"-"*60}\n')

    print(f'\n{"-"*60}')
    print(f'{case_name} completed in {elapsed_time:.2f} seconds')
    print(f'{"="*60}\n')

    os.chdir(cwd)


def main():
    cwd = os.getcwd()
    sim_dst = os.path.abspath('./sim.py')
    python_exec = sys.executable

    # Case 1: no mpi (-np 1), no numba
    # exec_case(
    #     case_name='Case 1',
    #     case_dir=os.path.join(cwd, 'case_1'),
    #     command=['mpirun', '-np', '1', python_exec, sim_dst],
    #     cwd=cwd,
    #     env_vars={'USE_NUMBA': '0'}
    # )
    #
    # # Case 2a: mpi -np 4, no numba
    # exec_case(
    #     case_name='Case 2a',
    #     case_dir=os.path.join(cwd, 'case_2a'),
    #     command=['mpirun', '-np', '4', python_exec, sim_dst],
    #     cwd=cwd,
    #     env_vars={'USE_NUMBA': '0'}
    # )

    # Case 2b: mpi -np 8, no numba
    exec_case(
        case_name='Case 2b',
        case_dir=os.path.join(cwd, 'case_2b'),
        command=['mpirun', '-np', '8', python_exec, sim_dst],
        cwd=cwd,
        env_vars={'USE_NUMBA': '0'}
    )

    # Case 3: no mpi (-np 1), numba enabled
    exec_case(
        case_name='Case 3',
        case_dir=os.path.join(cwd, 'case_3'),
        command=['mpirun', '-np', '1', python_exec, sim_dst],
        cwd=cwd,
        env_vars={'USE_NUMBA': '1'}
    )

    # Case 4: mpi -np 8, numba enabled
    exec_case(
        case_name='Case 4',
        case_dir=os.path.join(cwd, 'case_4'),
        command=['mpirun', '-np', '8', python_exec, sim_dst],
        cwd=cwd,
        env_vars={'USE_NUMBA': '1'}
    )

if __name__ == '__main__':
    main()
    pass