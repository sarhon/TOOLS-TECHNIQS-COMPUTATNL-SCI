module input_arrays
    use, intrinsic :: iso_fortran_env, only: real32
    implicit none
    private
    public :: InputArrays, new_input_arrays

    type, public :: InputArrays
        integer :: n, m
        real(real32), allocatable :: x(:,:) 
        real(real32), allocatable :: b(:)
    end type InputArrays

    contains
        subroutine new_input_arrays(self, n, m)
            class(InputArrays), intent(out) :: self
            integer, intent(in) :: n, m

            self%n = n
            self%m = m

            allocate(self%x(n,n))
            self%x = 2.0_real32
            
            allocate(self%b(m))
            self%b = 1.0_real32
        end subroutine new_input_arrays
end module input_arrays

module output_array
    use input_arrays
    use, intrinsic :: iso_fortran_env, only: real32    
    implicit none
    private

    public :: OutputArray

    type, public :: OutputArray
        integer :: n
        real(real32), allocatable :: y(:,:)
        contains
            procedure :: compute
    end type OutputArray

    public :: new_output_array

    contains
        subroutine new_output_array(self, n)
            class(OutputArray), intent(out) :: self
            integer, intent(in) :: n

            self%n = n
            allocate(self%y(n,n))
            self%y = 1.0_real32

        end subroutine new_output_array

        subroutine compute(self, inputs, k, row)
            class(OutputArray), intent(inout) :: self
            class(InputArrays), intent(in) :: inputs
            integer, intent(in) :: k
            logical, intent(in) :: row

            ! I think this is faster than doing a loop
            ! This should compile to a tight double loop
            self%y = (self%y + 2.0_real32 * inputs%x) / 5.0_real32
            
            if (row) then
                self%y(k, 1:inputs%m) = self%y(k, 1:inputs%m) + inputs%b(1:inputs%m)
            else ! column major (Fortran)
                self%y(1:inputs%m, k) = self%y(1:inputs%m, k) + inputs%b(1:inputs%m)
            end if
            
        end subroutine compute

end module output_array

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
    character(len=256) :: nml_file, output_dir, output_size_dst, output_array_dst
    ! character(len=256) :: output_dir
    
    integer :: nml_unit, output_size_unit, output_array_unit

    integer :: nargs

    ! Namelist definition
    namelist /params/ n, m, k, row

    ! Defaults
    n = 100
    m = 50
    k = 25
    output_dir = '.'

    row = .false.

    output_size_unit = 10
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

    output_size_dst = join_path(output_dir, 'size.txt')
    print *, "Saving", output_size_dst

    open(newunit=output_size_unit, file=output_size_dst, status='replace', action='write')
    write(output_size_unit, '(A, A)') 'x: ', pbytes(nbytes(inputs%x))
    write(output_size_unit, '(A, A)') 'b: ', pbytes(nbytes(inputs%b))
    write(output_size_unit, '(A, A)') 'y: ', pbytes(nbytes(output%y))
    close(output_size_unit)

    ! output_array_dst = join_path(output_dir, 'array.txt')
    ! print *, "Saving", output_array_dst

    ! open(newunit=output_array_unit, file=output_array_dst, status='replace', action='write')
    ! do i = 1, output%n
    !     write(output_array_unit,'(100(f8.3,1x))') (output%y(i,j), j=1, output%n)
    ! end do

    close(output_array_unit)

end program name