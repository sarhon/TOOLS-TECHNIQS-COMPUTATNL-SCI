module output_array
    use input_arrays
    use, intrinsic :: iso_fortran_env, only: int64, real32
    implicit none
    private

    public :: OutputArray

    type, public :: OutputArray
        integer :: n
        real(real32), allocatable :: y(:,:)
        integer(int64) :: y_bytes
        contains
            procedure :: compute
    end type OutputArray

    public :: new_output_array

    contains
        subroutine new_output_array(self, n)
            class(OutputArray), intent(out) :: self
            integer, intent(in) :: n
            integer(int64) :: y_bytes
            integer :: i, j

            self%n = n

            print *, "   Allocate y"
            allocate(self%y(n,n))
            print *, "   End allocate y"

            print *, "   Set y=1.0 (parallel)"
            !$omp parallel workshare
            self%y = 1.0_real32
            !$omp end parallel workshare
            print *, "   End set y"

            print *, "   Calculate y_bytes"
            self%y_bytes = int(self%n, int64) * int(self%n, int64) * storage_size(self%y) / 8_int64
            print *, "   End calculate"

        end subroutine new_output_array

        subroutine compute(self, inputs, k, row)
            class(OutputArray), intent(inout) :: self
            class(InputArrays), intent(in) :: inputs
            integer, intent(in) :: k
            logical, intent(in) :: row
            integer :: i, j

            print *, "starting compute"
            print *, "    Start (y + 2.0_real32 * x) / 5.0_real32"
            !$omp parallel do
            do j = 1, self%n
                !$omp simd
                do i = 1, self%n
                    self%y(i,j) = 0.2_real32 * self%y(i,j) + 0.4_real32 * inputs%x(i,j)
                end do
            end do
            !$omp end parallel do
            print *, "    End"


            if (row) then
                print *, "    Start Row"
                !$omp parallel do
                do i = 1, inputs%m
                    self%y(k,i) = self%y(k,i) + inputs%b(i)
                end do
                !$omp end parallel do
                print *, "    End Row"
            else ! column major (Fortran)
                print *, "    Start Column"
                !$omp parallel do
                do i = 1, inputs%m
                    self%y(i,k) = self%y(i,k) + inputs%b(i)
                end do
                !$omp end parallel do
                print *, "    End Column"
            end if

            print *, "Ending compute"

        end subroutine compute

end module output_array