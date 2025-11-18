module input_arrays
    use, intrinsic :: iso_fortran_env, only: int64, real32
    implicit none
    private
    public :: InputArrays, new_input_arrays

    type, public :: InputArrays
        integer :: n, m
        real(real32), allocatable :: x(:,:) 
        real(real32), allocatable :: b(:)
        integer(int64) :: x_bytes, b_bytes
    end type InputArrays

    contains
        subroutine new_input_arrays(self, n, m)
            class(InputArrays), intent(out) :: self
            integer, intent(in) :: n, m
            integer(int64) :: x_bytes, b_bytes
            integer :: i, j
            
            self%n = n
            self%m = m

            print *, "   Allocate x"
            allocate(self%x(n,n))
            print *, "   End allocate x"

            print *, "   Set x=2.0 (parallel)"
            !$omp parallel workshare
            self%x = 2.0_real32
            !$omp end parallel workshare
            print *, "   End set x"

            print *, "   Allocate b"
            allocate(self%b(m))
            print *, "   End allocate b"

            print *, "   Set b=1.0 (parallel)"
            !$omp parallel workshare
            self%b = 1.0_real32
            !$omp end parallel workshare
            print *, "   End set b"

            print *, "   Calculate x_bytes"
            self%x_bytes = int(self%n, int64) * int(self%n, int64) * storage_size(self%x) / 8_int64
            print *, "   End calculate"

            print *, "   Calculate b_bytes"
            self%b_bytes = int(self%m, int64) * storage_size(self%b) / 8_int64
            print *, "   End calculate"
        end subroutine new_input_arrays
end module input_arrays