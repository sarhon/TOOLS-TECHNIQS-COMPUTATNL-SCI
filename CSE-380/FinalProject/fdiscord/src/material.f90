module material_mod
    use, intrinsic :: iso_fortran_env, only: int64, real64
    implicit none
    private
    public :: Material, NewMaterial

    type :: Material
        character(len=:), allocatable :: name
        real(real64) :: total, scatter, Q, left_bound, right_bound, absorption

    end type Material

    contains
        subroutine NewMaterial(self, name, total, scatter, Q, left_bound, right_bound)
            class(Material), intent(out) :: self
            character(len=*), intent(in) :: name
            real(real64), intent(in) :: total, scatter, Q, left_bound, right_bound

            self%name = name
            self%total = total
            self%scatter = scatter
            self%Q = Q
            self%left_bound = left_bound
            self%right_bound = right_bound

            self%absorption = total - scatter

        end subroutine NewMaterial

end module material_mod