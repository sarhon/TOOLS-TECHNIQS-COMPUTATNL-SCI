module right_flux
    use, intrinsic :: iso_fortran_env, only: int64, real64
    implicit none
    private
    public :: RightFlux, NewRightFlux, MakeMatrix, SolveMatrix

    type, public :: RightFlux
        real(real64), allocatable :: x(:)           ! position edges
        real(real64), allocatable :: sigma_t(:)     ! total cross section
        real(real64) :: mu                          ! direction cosine
        real(real64) :: dx                          ! spatial step
        real(real64), allocatable :: tau(:)         ! optical thickness
        integer(int64) :: N                         ! number of nodes

        ! Matrix diagonals
        real(real64), allocatable :: diag_center(:) ! main diagonal
        real(real64), allocatable :: diag_lower(:)  ! lower diagonal

        ! Solution
        real(real64), allocatable :: solved(:)      ! solution vector
    end type RightFlux

contains

    subroutine NewRightFlux(self, x, sigma_t, mu)
        class(RightFlux), intent(out) :: self
        real(real64), intent(in) :: x(:)
        real(real64), intent(in) :: sigma_t(:)
        real(real64), intent(in) :: mu

        integer(int64) :: i

        self%N = size(x)

        ! Allocate arrays
        allocate(self%x(self%N))
        allocate(self%sigma_t(self%N - 1))
        allocate(self%tau(self%N - 1))
        allocate(self%diag_center(self%N))
        allocate(self%diag_lower(self%N - 1))
        allocate(self%solved(self%N))

        ! Copy input data
        self%x = x
        self%sigma_t = sigma_t
        self%mu = mu

        ! Calculate dx (assuming uniform spacing)
        self%dx = x(2) - x(1)

        ! Calculate tau for each cell
        do i = 1, self%N - 1
            self%tau(i) = self%sigma_t(i) * self%dx / self%mu
        end do

    end subroutine NewRightFlux

    subroutine MakeMatrix(self)
        class(RightFlux), intent(inout) :: self
        integer(int64) :: i

        ! Main diagonal: all ones
        self%diag_center = 1.0_real64

        ! Lower diagonal: -exp(-tau)
        do i = 1, self%N - 1
            self%diag_lower(i) = -exp(-self%tau(i))
        end do

    end subroutine MakeMatrix

    subroutine SolveMatrix(self, phi0, Q_source, Q_scatter)
        class(RightFlux), intent(inout) :: self
        real(real64), intent(in) :: phi0
        real(real64), intent(in) :: Q_source(:)
        real(real64), intent(in) :: Q_scatter(:)

        real(real64), allocatable :: b(:)           ! RHS vector
        real(real64), allocatable :: diag_temp(:)   ! temporary diagonal for solver
        real(real64), allocatable :: lower_temp(:)  ! temporary lower diagonal
        integer(int64) :: i

        ! Allocate RHS vector
        allocate(b(self%N))
        allocate(diag_temp(self%N))
        allocate(lower_temp(self%N - 1))

        ! Build RHS: b = (Q_source/2 + Q_scatter) / sigma_t * (-expm1(-tau))
        ! Note: -expm1(-x) = -exp(-x) + 1 = 1 - exp(-x)
        do i = 1, self%N - 1
            b(i + 1) = ((Q_source(i) / 2.0_real64 + Q_scatter(i)) / self%sigma_t(i)) * &
                       (1.0_real64 - exp(-self%tau(i)))
        end do

        ! Boundary condition at left (index 1)
        b(1) = phi0

        ! Copy diagonals for solver (will be modified)
        diag_temp = self%diag_center
        lower_temp = self%diag_lower

        ! Solve tridiagonal system Ax = b
        ! For right flux: upper diagonal is zero, lower diagonal is -exp(-tau)
        call solve_tridiagonal(lower_temp, diag_temp, self%N, b, self%solved)

        deallocate(b, diag_temp, lower_temp)

    end subroutine SolveMatrix

    subroutine solve_tridiagonal(lower, diag, n, b, x)
        ! Bidiagonal solver for system with upper diagonal = 0
        ! This is for right flux (main diagonal + lower diagonal only)
        ! Matrix form: [diag(i) on main, lower(i-1) one below]
        ! No elimination needed - just forward substitution
        integer(int64), intent(in) :: n
        real(real64), intent(in) :: lower(n-1)
        real(real64), intent(in) :: diag(n)
        real(real64), intent(in) :: b(n)
        real(real64), intent(out) :: x(n)

        integer(int64) :: i

        ! Forward substitution (no elimination needed for bidiagonal)
        x(1) = b(1) / diag(1)
        do i = 2, n
            x(i) = (b(i) - lower(i-1) * x(i-1)) / diag(i)
        end do

    end subroutine solve_tridiagonal

end module right_flux


module left_flux
    use, intrinsic :: iso_fortran_env, only: int64, real64
    implicit none
    private
    public :: LeftFlux, NewLeftFlux, MakeMatrixLeft, SolveMatrixLeft

    type, public :: LeftFlux
        real(real64), allocatable :: x(:)           ! position edges
        real(real64), allocatable :: sigma_t(:)     ! total cross section
        real(real64) :: mu                          ! direction cosine
        real(real64) :: dx                          ! spatial step
        real(real64), allocatable :: tau(:)         ! optical thickness
        integer(int64) :: N                         ! number of nodes

        ! Matrix diagonals
        real(real64), allocatable :: diag_center(:) ! main diagonal
        real(real64), allocatable :: diag_upper(:)  ! upper diagonal

        ! Solution
        real(real64), allocatable :: solved(:)      ! solution vector
    end type LeftFlux

contains

    subroutine NewLeftFlux(self, x, sigma_t, mu)
        class(LeftFlux), intent(out) :: self
        real(real64), intent(in) :: x(:)
        real(real64), intent(in) :: sigma_t(:)
        real(real64), intent(in) :: mu

        integer(int64) :: i

        self%N = size(x)

        ! Allocate arrays
        allocate(self%x(self%N))
        allocate(self%sigma_t(self%N - 1))
        allocate(self%tau(self%N - 1))
        allocate(self%diag_center(self%N))
        allocate(self%diag_upper(self%N - 1))
        allocate(self%solved(self%N))

        ! Copy input data
        self%x = x
        self%sigma_t = sigma_t
        self%mu = mu

        ! Calculate dx (assuming uniform spacing)
        self%dx = x(2) - x(1)

        ! Calculate tau for each cell
        do i = 1, self%N - 1
            self%tau(i) = self%sigma_t(i) * self%dx / self%mu
        end do

    end subroutine NewLeftFlux

    subroutine MakeMatrixLeft(self)
        class(LeftFlux), intent(inout) :: self
        integer(int64) :: i

        ! Main diagonal: all ones
        self%diag_center = 1.0_real64

        ! Upper diagonal: -exp(-tau)
        do i = 1, self%N - 1
            self%diag_upper(i) = -exp(-self%tau(i))
        end do

    end subroutine MakeMatrixLeft

    subroutine SolveMatrixLeft(self, phiN, Q_source, Q_scatter)
        class(LeftFlux), intent(inout) :: self
        real(real64), intent(in) :: phiN
        real(real64), intent(in) :: Q_source(:)
        real(real64), intent(in) :: Q_scatter(:)

        real(real64), allocatable :: b(:)           ! RHS vector
        real(real64), allocatable :: diag_temp(:)   ! temporary diagonal for solver
        real(real64), allocatable :: upper_temp(:)  ! temporary upper diagonal
        integer(int64) :: i

        ! Allocate RHS vector
        allocate(b(self%N))
        allocate(diag_temp(self%N))
        allocate(upper_temp(self%N - 1))

        ! Build RHS: b = (Q_source/2 + Q_scatter) / sigma_t * (1 - exp(-tau))
        do i = 1, self%N - 1
            b(i) = ((Q_source(i) / 2.0_real64 + Q_scatter(i)) / self%sigma_t(i)) * &
                   (1.0_real64 - exp(-self%tau(i)))
        end do

        ! Boundary condition at right (index N)
        b(self%N) = phiN

        ! Copy diagonals for solver (will be modified)
        diag_temp = self%diag_center
        upper_temp = self%diag_upper

        ! Solve tridiagonal system Ax = b
        ! For left flux: lower diagonal is zero, upper diagonal is -exp(-tau)
        call solve_tridiagonal_upper(upper_temp, diag_temp, self%N, b, self%solved)

        deallocate(b, diag_temp, upper_temp)

    end subroutine SolveMatrixLeft

    subroutine solve_tridiagonal_upper(upper, diag, n, b, x)
        ! Bidiagonal solver for system with lower diagonal = 0
        ! This is for left flux (main diagonal + upper diagonal only)
        ! Matrix form: [diag(i) on main, upper(i) one above]
        ! No elimination needed - just backward substitution
        integer(int64), intent(in) :: n
        real(real64), intent(in) :: upper(n-1)
        real(real64), intent(in) :: diag(n)
        real(real64), intent(in) :: b(n)
        real(real64), intent(out) :: x(n)

        integer(int64) :: i

        ! Backward substitution (no elimination needed for bidiagonal)
        x(n) = b(n) / diag(n)
        do i = n-1, 1, -1
            x(i) = (b(i) - upper(i) * x(i+1)) / diag(i)
        end do

    end subroutine solve_tridiagonal_upper

end module left_flux