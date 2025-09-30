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

            self%n = n

            print *, "   Allocate y"
            allocate(self%y(n,n))
            print *, "   End allocate y"
            
            print *, "   Set y=1.0"
            !$omp parallel workshare
            self%y = 1.0_real32
            !$omp end parallel workshare
            print *, "   End set"

            print *, "   Calculate y_bytes"
            self%y_bytes = int(self%n, int64) * int(self%n, int64) * storage_size(self%y) / 8_int64
            print *, "   End calculate"
            
        end subroutine new_output_array

        subroutine compute(self, inputs, k, row)
            use omp_lib
            class(OutputArray), intent(inout) :: self
            class(InputArrays), intent(in) :: inputs
            integer, intent(in) :: k
            logical, intent(in) :: row
            integer :: num_threads, i, j

            print *, "starting compute"

            ! !$omp parallel
            ! !$omp single
            ! num_threads = omp_get_num_threads()
            ! ! call omp_set_num_threads(num_threads)
            ! print *, "Number of threads: ", num_threads
            ! !$omp end single
            ! !$omp end parallel

            ! I think this is faster than doing a loop
            ! This should compile to a tight double loop
            ! print *, "    (y + 2.0_real32 * x) / 5.0_real32"
            ! !$omp parallel workshare
            ! self%y = (self%y + 2.0_real32 * inputs%x) / 5.0_real32
            ! !$omp end parallel workshare
            ! print *, "    End"

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

            ! if (row) then
            !     !$omp parallel workshare
            !     self%y(k, 1:inputs%m) = self%y(k, 1:inputs%m) + inputs%b(1:inputs%m)
            !     !$omp end parallel workshare
            ! else ! column major (Fortran)
            !     !$omp parallel workshare
            !     self%y(1:inputs%m, k) = self%y(1:inputs%m, k) + inputs%b(1:inputs%m)
            !     !$omp end parallel workshare
            ! end if

        end subroutine compute

end module output_array