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
            use omp_lib
            class(OutputArray), intent(inout) :: self
            class(InputArrays), intent(in) :: inputs
            integer, intent(in) :: k
            logical, intent(in) :: row
            integer :: num_threads, i

            ! !$omp parallel
            ! !$omp single
            ! num_threads = omp_get_num_threads()
            ! ! call omp_set_num_threads(num_threads)
            ! print *, "Number of threads: ", num_threads
            ! !$omp end single
            ! !$omp end parallel

            ! I think this is faster than doing a loop
            ! This should compile to a tight double loop
            self%y = (self%y + 2.0_real32 * inputs%x) / 5.0_real32
            
            if (row) then
                !$omp parallel do
                do i = 1, inputs%m
                    self%y(k,i) = self%y(k,i) + inputs%b(i)
                end do
                !$omp end parallel do
            else ! column major (Fortran)
                !$omp parallel do
                do i = 1, inputs%m
                    self%y(i,k) = self%y(i,k) + inputs%b(i)
                end do
                !$omp end parallel do
            end if


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