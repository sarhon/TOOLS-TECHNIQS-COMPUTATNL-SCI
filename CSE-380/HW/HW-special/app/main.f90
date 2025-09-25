program name
    use, intrinsic :: iso_fortran_env, only: real32, output_unit
    use mem_util, only : nbytes, pbytes
    use path_util
    use input_arrays
    use output_array
    implicit none
    
    integer :: n, m, k
    type(InputArrays) :: inputs
    type(OutputArray) :: output
    logical :: row, print_summary, print_array
    integer :: i, j
    real(real32) :: sum_first_row, sum_kth_row, sum_first_col, sum_kth_col
    character(len=256) :: nml_file, output_dir, output_summary_dst, output_array_dst
    
    integer :: nml_unit, output_summary_unit, output_array_unit

    integer :: nargs

    ! Namelist definition
    namelist /params/ n, m, k, row, print_summary, print_array

    ! Defaults
    n = 100
    m = 50
    k = 25
    output_dir = '.'

    row = .false. ! Fortran is column major
    print_summary = .true.
    print_array = .false.

    ! default file units, likley overwritten
    output_summary_unit = 10
    output_array_unit = 11

    nargs = command_argument_count() ! number of arguments

    ! gets the first argument (input nml file)
    if (nargs > 0) then
        call get_command_argument(1, nml_file)
        ! print *, nml_file
        open(newunit=nml_unit, file=trim(nml_file), status='old')
        read(nml_unit, nml=params)
        close(nml_unit)
    end if

    ! Input validation
    if (n <= 0) then ! fail 1
        write(*, '(A,I0)') 'Error(1) n must be positive, n=', n
        stop 1
    end if

    if (m <= 0) then ! fail 2
        write(*, '(A,I0)') 'Error(2) m must be positive, m=', m
        stop 1
    end if

    if (k <= 0) then ! fail 3
        write(*, '(A,I0)') 'Error(3) k must be positive, k=', k
        stop 1
    end if

    if (m > n) then ! fail 4
        write(*, '(A,I0, A,I0)') 'Error(4) m must be less than n, m=', m, ' n=', n
        stop 1
    end if

    if (k > n) then ! fail 5
        write(*, '(A,I0, A,I0)') 'Error(5) k must be less than n, k=', k, ' n=', n
        stop 1
    end if

    ! print *, 'Running: Arrays'
    ! 3a I placed the class initializations here because it is after the defintion of 
    !    the n, m, and k variables but before the invokation of the computation method
    
    call new_input_arrays(self=inputs, n=n, m=m)                 ! 3a initilzing the input object
    call new_output_array(self=output, n=n)                      ! 3a intilizing the output object
    
    ! 3b The arrys are dynamically allocated on the heap at runtime 
    !    this also makes senst to allocate to the heap because if the size of some of the arrays 
    !    being 90k x 90k it would likely overflow the stack if allocated there
    
    ! 3c I used the standard dynamic allocation built into Fortran. This was done so that the size 
    !    of the arrays could be defined at runtime based on the input parameters

    ! 3d I expect the code to spend the most time on the in the method call below. This is because
    !    this is where the matrix operations take place, even though I am using the built in Fortran
    !    matrix operations, the underlying machine code needs still loop over every entry
    !                                         --- or ---
    !    (if it is enabled) the array save to a file at the bottom of this code, this takes forever because it 
    !    has to loop over every entry then write to storage (not cache or ram) which is very slow
    call output%compute(inputs=inputs, k=k, row=row)
    
    ! First and kth column sum (most efficient - contiguous memory access)
    sum_first_col = sum(output%y(:, 1))
    sum_kth_col = sum(output%y(:, k))

    ! First and kth row sum (less efficient but necessary - strided access)
    sum_first_row = sum(output%y(1, :))
    sum_kth_row = sum(output%y(k, :))
    
    ! gets the second argument (output directory)
    if (nargs > 1) then
        call get_command_argument(2, output_dir)
    end if

    output_summary_dst = join_path(output_dir, 'summary.txt')
    print *, "Saving: ", output_summary_dst

    open(newunit=output_summary_unit, file=output_summary_dst, status='replace', action='write')
    write(output_summary_unit, '(A, A)') 'x: ', pbytes(nbytes(inputs%x))
    write(output_summary_unit, '(A, A)') 'b: ', pbytes(nbytes(inputs%b))
    write(output_summary_unit, '(A, A)') 'y: ', pbytes(nbytes(output%y))
    write(output_summary_unit, '(A)') ''

    write(output_summary_unit,'(A,F12.0)') 'Sum of first row: ', sum_first_row
    write(output_summary_unit,'(A,F12.0)') 'Sum of kth row:   ', sum_kth_row
    write(output_summary_unit,'(A,F12.0)') 'Sum of first col: ', sum_first_col
    write(output_summary_unit,'(A,F12.0)') 'Sum of kth col:   ', sum_kth_col
    write(output_summary_unit, '(A)') ''

    close(output_summary_unit)

    if (print_array) then
        output_array_dst = join_path(output_dir, 'array.txt')
        print *, "Saving: ", output_array_dst

        open(newunit=output_array_unit, file=output_array_dst, status='replace', action='write')
        do i = 1, output%n
            write(output_array_unit,'(*(i1))') (int(output%y(i,j)), j=1, output%n)
        end do
        close(output_array_unit)
    end if

end program name