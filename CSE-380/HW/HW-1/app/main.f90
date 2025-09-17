program name
    use, intrinsic :: iso_fortran_env, only: real32, output_unit
    use mem_util, only : nbytes, pbytes
    use path_util
    use input_arrays
    use output_array
    implicit none
    
    integer :: n, m, k ! integers
    type(InputArrays) :: inputs
    type(OutputArray) :: output
    logical :: row
    integer :: i, j
    real(real32) :: sum_first_row, sum_kth_row, sum_first_col, sum_kth_col
    character(len=256) :: nml_file, output_dir, output_summary_dst, output_array_dst
    ! character(len=256) :: output_dir
    
    integer :: nml_unit, output_summary_unit, output_array_unit

    integer :: nargs

    ! Namelist definition
    namelist /params/ n, m, k, row

    ! Defaults
    n = 100
    m = 50
    k = 25
    output_dir = '.'

    row = .false.

    output_summary_unit = 10
    output_array_unit = 11

    nargs = command_argument_count()

    if (nargs > 0) then
        call get_command_argument(1, nml_file)
        ! print *, nml_file
        open(newunit=nml_unit, file=trim(nml_file), status='old')
        read(nml_unit, nml=params)
        close(nml_unit)
    end if

    ! write(*,'("n=",I0," m=",I0," k=",I0)') n, m, k

    
    print *, 'Running arrays'
    ! 3a I placed the class initializations here because it is after the defintion of 
    !    the n, m, and k variables but before the invokation of the computation method
    call new_input_arrays(self=inputs, n=n, m=m)                 ! 3a initilzing the input object
    call new_output_array(self=output, n=n)                      ! 3a intilizing the output object
    call output%compute(inputs=inputs, k=k, row=row)

    

    if (nargs > 1) then
        call get_command_argument(2, output_dir)
        ! print *, output_dir
    end if

    output_summary_dst = join_path(output_dir, 'summary.txt')
    print *, "Saving", output_summary_dst

    open(newunit=output_summary_unit, file=output_summary_dst, status='replace', action='write')
    write(output_summary_unit, '(A, A)') 'x: ', pbytes(nbytes(inputs%x))
    write(output_summary_unit, '(A, A)') 'b: ', pbytes(nbytes(inputs%b))
    write(output_summary_unit, '(A, A)') 'y: ', pbytes(nbytes(output%y))
    

    ! Compute sums efficiently (column-major friendly)
    ! First column sum (most efficient - contiguous memory access)
    sum_first_col = sum(output%y(:, 1))

    ! kth column sum (also efficient - contiguous memory access)
    sum_kth_col = sum(output%y(:, k))

    ! First row sum (less efficient but necessary - strided access)
    sum_first_row = sum(output%y(1, :))

    ! kth row sum (less efficient but necessary - strided access)
    sum_kth_row = sum(output%y(k, :))

    ! Output the results
    write(output_summary_unit, '(A)') ''
    write(output_summary_unit,'(A,F12.0)') 'Sum of first row: ', sum_first_row
    write(output_summary_unit,'(A,F12.0)') 'Sum of kth row:   ', sum_kth_row
    write(output_summary_unit,'(A,F12.0)') 'Sum of first col: ', sum_first_col
    write(output_summary_unit,'(A,F12.0)') 'Sum of kth col:   ', sum_kth_col



    close(output_summary_unit)

    ! output_array_dst = join_path(output_dir, 'array.txt')
    ! print *, "Saving", output_array_dst

    ! open(newunit=output_array_unit, file=output_array_dst, status='replace', action='write')
    ! do i = 1, output%n
    !     write(output_array_unit,'(100(f8.3,1x))') (output%y(i,j), j=1, output%n)
    ! end do
    ! close(output_array_unit)

end program name