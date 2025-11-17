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
            ! allocate(self%x(n,n), source=2.0_real32)
            print *, "   End allocate x"

            print *, "   Allocate m"
            allocate(self%b(m))
            ! allocate(self%b(m), source=1.0_real32)
            print *, "   End allocate m"
            
            print *, "   Set x=2.0"
            ! Column-major order initialization (optimal for Fortran)
            ! This reduces TLB misses and ensures first-touch optimization
            ! Each thread touches pages in the order they're laid out in memory
            !$omp parallel do schedule(static) private(i)
            do j = 1, self%n
                !$omp simd
                do i = 1, self%n
                    self%x(i,j) = 2.0_real32
                end do
            end do
            !$omp end parallel do
            print *, "   End set"

            print *, "   Set b=1.0"
            !$omp parallel do schedule(static)
            do i = 1, self%m
                self%b(i) = 1.0_real32
            end do
            !$omp end parallel do
            print *, "   End set"

            print *, "   Calculate x_bytes"
            self%x_bytes = int(self%n, int64) * int(self%n, int64) * storage_size(self%x) / 8_int64
            print *, "   End calculate"

            print *, "   Calculate b_bytes"
            self%b_bytes = int(self%m, int64) * storage_size(self%b) / 8_int64
            print *, "   End calculate"
        end subroutine new_input_arrays
end module input_arrays