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