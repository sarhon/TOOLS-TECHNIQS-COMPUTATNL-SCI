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