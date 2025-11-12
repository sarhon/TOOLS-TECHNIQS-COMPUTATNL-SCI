module settings_mod
    use, intrinsic :: iso_fortran_env, only: int64, real64
    implicit none
    private
    public :: Settings, NewSettings

    type :: Settings
        real(real64) :: phiL        ! left boundary condition value
        real(real64) :: phiR        ! right boundary condition value
        character(len=10) :: phiL_type  ! 'vac', 'ref', or 'fixed'
        character(len=10) :: phiR_type  ! 'vac', 'ref', or 'fixed'
        integer(int64) :: num_nodes     ! number of spatial nodes
        integer(int64) :: sn            ! quadrature order
        real(real64), allocatable :: mu(:)  ! custom quadrature angles (optional)
        real(real64), allocatable :: w(:)   ! custom quadrature weights (optional)
    end type Settings

    contains
        subroutine NewSettings(self, phiL, phiR, phiL_type, phiR_type, num_nodes, sn)
            class(Settings), intent(out) :: self
            real(real64), intent(in) :: phiL, phiR
            character(len=*), intent(in) :: phiL_type, phiR_type
            integer(int64), intent(in) :: num_nodes, sn

            self%phiL = phiL
            self%phiR = phiR
            self%phiL_type = phiL_type
            self%phiR_type = phiR_type
            self%num_nodes = num_nodes
            self%sn = sn

        end subroutine NewSettings

end module settings_mod