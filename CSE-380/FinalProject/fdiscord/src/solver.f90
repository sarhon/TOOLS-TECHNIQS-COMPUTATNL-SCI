module solver
    use, intrinsic :: iso_fortran_env, only: int64, real64
    use material_mod
    use right_flux
    use left_flux
    use settings_mod
    implicit none
    private
    public :: solve_flux, avg_flux, gauss_legendre

contains

    function avg_flux(xi, xe, phi_i, mu, Q, sigma_t) result(avg_flux_val)
        ! Calculate average flux in a cell using exponential differencing
        real(real64), intent(in) :: xi, xe      ! cell edges
        real(real64), intent(in) :: phi_i       ! incoming flux
        real(real64), intent(in) :: mu          ! direction cosine
        real(real64), intent(in) :: Q           ! source term
        real(real64), intent(in) :: sigma_t     ! total cross section
        real(real64) :: avg_flux_val

        real(real64) :: dx, tau, term

        dx = xe - xi
        tau = sigma_t * dx / mu

        ! Simplified formula: (Q/sigma_t) + (phi_i - Q/sigma_t) * (-expm1(-tau)/tau)
        ! where -expm1(-tau) = 1 - exp(-tau)
        if (abs(tau) < 1.0e-10_real64) then
            ! Taylor series for small tau to avoid numerical issues
            avg_flux_val = phi_i + Q * dx / (2.0_real64 * abs(mu))
        else
            term = (1.0_real64 - exp(-tau)) / tau
            avg_flux_val = (Q / sigma_t) + (phi_i - Q / sigma_t) * term
        end if

    end function avg_flux

    subroutine solve_flux(materials, num_materials, config, x_edges, total_flux, current, &
                         angular_flux_out, mu_out, w_out, x_centers_out)
        ! Main solver routine for discrete ordinates transport
        type(Material), intent(in) :: materials(:)
        integer(int64), intent(in) :: num_materials
        type(Settings), intent(inout) :: config
        real(real64), allocatable, intent(out) :: x_edges(:)
        real(real64), allocatable, intent(out) :: total_flux(:)
        real(real64), allocatable, intent(out) :: current(:)
        real(real64), allocatable, intent(out), optional :: angular_flux_out(:,:)
        real(real64), allocatable, intent(out), optional :: mu_out(:)
        real(real64), allocatable, intent(out), optional :: w_out(:)
        real(real64), allocatable, intent(out), optional :: x_centers_out(:)

        ! Local variables
        real(real64) :: L_bound, R_bound, dx
        real(real64), allocatable :: x_center(:)
        real(real64), allocatable :: nodal_total(:), nodal_scatter(:), nodal_Q(:)
        real(real64), allocatable :: Q_source(:), Q_scatter(:), Q_scatter_old(:)
        real(real64), allocatable :: left_mu(:), right_mu(:)
        real(real64), allocatable :: left_w(:), right_w(:)
        real(real64), allocatable :: mu(:), w(:)
        real(real64), allocatable :: angular_flux(:,:)
        real(real64), allocatable :: right_flux(:), left_flux(:)
        real(real64), allocatable :: right_avg_flux(:), left_avg_flux(:)
        real(real64), allocatable :: old_flux(:), old_avg_flux(:), total_avg_flux(:)
        real(real64) :: flux_error, Q_error, phiL0, phiRN, phi0, phiN
        real(real64) :: left_ordinance, left_weight, right_ordinance, right_weight
        integer(int64) :: i, j, s, idx, position_index
        integer(int64) :: max_step, num_angles, midpoint
        type(RightFlux) :: rf
        type(LeftFlux) :: lf

        ! Find spatial bounds from materials
        L_bound = huge(1.0_real64)
        R_bound = -huge(1.0_real64)

        do i = 1, num_materials
            if (materials(i)%left_bound < L_bound) then
                L_bound = materials(i)%left_bound
            end if
            if (materials(i)%right_bound > R_bound) then
                R_bound = materials(i)%right_bound
            end if
        end do

        ! Create spatial mesh
        allocate(x_edges(config%num_nodes + 1))
        allocate(x_center(config%num_nodes))

        dx = (R_bound - L_bound) / real(config%num_nodes, real64)

        do i = 1, config%num_nodes + 1
            x_edges(i) = L_bound + real(i - 1, real64) * dx
        end do

        do i = 1, config%num_nodes
            x_center(i) = L_bound + real(i - 1, real64) * dx + dx / 2.0_real64
        end do

        ! Map materials to nodes
        allocate(nodal_total(config%num_nodes))
        allocate(nodal_scatter(config%num_nodes))
        allocate(nodal_Q(config%num_nodes))

        do i = 1, config%num_nodes
            do j = 1, num_materials
                if (materials(j)%left_bound <= x_center(i) .and. &
                    x_center(i) <= materials(j)%right_bound) then
                    nodal_total(i) = materials(j)%total
                    nodal_scatter(i) = materials(j)%scatter
                    nodal_Q(i) = materials(j)%Q
                    exit
                end if
            end do
        end do

        ! Setup quadrature
        if (allocated(config%mu) .and. allocated(config%w)) then
            ! Use custom quadrature
            num_angles = size(config%mu)
            allocate(mu(num_angles))
            allocate(w(num_angles))
            mu = config%mu
            w = config%w
        else
            ! Use Gauss-Legendre quadrature
            num_angles = config%sn
            allocate(mu(num_angles))
            allocate(w(num_angles))
            call gauss_legendre(num_angles, mu, w)
        end if

        ! Split into left and right going angles
        midpoint = num_angles / 2
        allocate(left_mu(midpoint))
        allocate(right_mu(midpoint))
        allocate(left_w(midpoint))
        allocate(right_w(midpoint))

        left_mu = mu(1:midpoint)
        left_w = w(1:midpoint)
        right_mu = mu(midpoint+1:num_angles)
        right_w = w(midpoint+1:num_angles)

        ! Initialize arrays
        allocate(Q_source(config%num_nodes))
        allocate(Q_scatter(config%num_nodes))
        allocate(Q_scatter_old(config%num_nodes))
        allocate(angular_flux(num_angles, config%num_nodes + 1))
        allocate(right_flux(config%num_nodes + 1))
        allocate(left_flux(config%num_nodes + 1))
        allocate(right_avg_flux(config%num_nodes))
        allocate(left_avg_flux(config%num_nodes))
        allocate(total_flux(config%num_nodes + 1))
        allocate(total_avg_flux(config%num_nodes))
        allocate(old_flux(config%num_nodes + 1))
        allocate(old_avg_flux(config%num_nodes))
        allocate(current(config%num_nodes + 1))

        Q_source = nodal_Q
        Q_scatter = 1.0_real64
        total_flux = 0.0_real64
        total_avg_flux = 0.0_real64
        phiL0 = 0.0_real64
        phiRN = 0.0_real64

        max_step = 5000
        flux_error = 1.0_real64

        print *, 'Starting source iteration...'

        ! Source iteration loop
        do s = 1, max_step
            angular_flux = 0.0_real64
            right_flux = 0.0_real64
            right_avg_flux = 0.0_real64
            left_flux = 0.0_real64
            left_avg_flux = 0.0_real64
            position_index = 1

            ! Solve for left-going fluxes
            ! OpenMP parallelization: Each thread handles different angles
            !$omp parallel do private(i,j,left_ordinance,left_weight,phiN,lf) &
            !$omp& reduction(+:left_flux,left_avg_flux) schedule(dynamic)
            do i = 1, midpoint
                left_ordinance = left_mu(i)
                left_weight = left_w(i)
                ! Determine boundary condition
                if (config%phiR_type == 'ref') then
                    phiN = phiRN
                else if (config%phiR_type == 'vac') then
                    phiN = 0.0_real64
                else
                    phiN = config%phiR
                end if

                ! Create and solve left flux
                call NewLeftFlux(lf, x_edges, nodal_total, -left_ordinance)
                call MakeMatrixLeft(lf)
                call SolveMatrixLeft(lf, phiN, Q_source, Q_scatter)

                ! Store results
                angular_flux(i, :) = lf%solved
                left_flux = left_flux + lf%solved * left_weight

                ! Calculate average flux
                do j = 1, config%num_nodes
                    left_avg_flux(j) = left_avg_flux(j) + &
                        avg_flux(x_edges(j), x_edges(j+1), lf%solved(j), &
                                 left_ordinance, Q_source(j)/2.0_real64 + Q_scatter(j), &
                                 nodal_total(j)) * left_weight
                end do
            end do
            !$omp end parallel do

            ! Solve for right-going fluxes
            ! OpenMP parallelization: Each thread handles different angles
            !$omp parallel do private(i,j,right_ordinance,right_weight,phi0,rf) &
            !$omp& reduction(+:right_flux,right_avg_flux) schedule(dynamic)
            do i = 1, midpoint
                right_ordinance = right_mu(i)
                right_weight = right_w(i)
                ! Determine boundary condition
                if (config%phiL_type == 'ref') then
                    phi0 = phiL0
                else if (config%phiL_type == 'vac') then
                    phi0 = 0.0_real64
                else
                    phi0 = config%phiL
                end if

                ! Create and solve right flux
                call NewRightFlux(rf, x_edges, nodal_total, right_ordinance)
                call MakeMatrix(rf)
                call SolveMatrix(rf, phi0, Q_source, Q_scatter)

                ! Store results
                angular_flux(midpoint + i, :) = rf%solved
                right_flux = right_flux + rf%solved * right_weight

                ! Calculate average flux
                do j = 1, config%num_nodes
                    right_avg_flux(j) = right_avg_flux(j) + &
                        avg_flux(x_edges(j), x_edges(j+1), rf%solved(j), &
                                 right_ordinance, Q_source(j)/2.0_real64 + Q_scatter(j), &
                                 nodal_total(j)) * right_weight
                end do
            end do
            !$omp end parallel do

            ! Update total flux
            old_flux = total_flux
            old_avg_flux = total_avg_flux
            total_flux = right_flux + left_flux
            total_avg_flux = right_avg_flux + left_avg_flux

            ! Update boundary fluxes for reflective conditions
            phiL0 = left_flux(1)
            phiRN = right_flux(config%num_nodes + 1)

            ! Save old scattering source before updating
            Q_scatter_old = Q_scatter

            ! Update scattering source
            Q_scatter = nodal_scatter / 2.0_real64 * total_avg_flux

            ! Calculate convergence metrics
            flux_error = norm2((total_flux - old_flux) / (total_flux + 1.0e-16_real64))

            Q_error = norm2((Q_scatter - Q_scatter_old) / (Q_scatter + 1.0e-16_real64))

            print '(A,I5,A,ES12.4,A,ES12.4,A)', &
                '(', s, ') dPhi/di = ', flux_error * 100.0_real64, &
                ' % | dQ/di = ', Q_error * 100.0_real64, ' %'

            if (flux_error < 1.0e-10_real64) then
                print *, 'Converged!'
                exit
            end if
        end do

        print '(A,ES12.4)', 'End of solution: flux error: ', flux_error

        ! Calculate current
        current = 0.0_real64
        do i = 1, num_angles
            do j = 1, config%num_nodes + 1
                current(j) = current(j) + w(i) * mu(i) * angular_flux(i, j)
            end do
        end do

        ! Return optional outputs
        if (present(angular_flux_out)) then
            allocate(angular_flux_out(num_angles, config%num_nodes + 1))
            angular_flux_out = angular_flux
        end if

        if (present(mu_out)) then
            allocate(mu_out(num_angles))
            mu_out = mu
        end if

        if (present(w_out)) then
            allocate(w_out(num_angles))
            w_out = w
        end if

        if (present(x_centers_out)) then
            allocate(x_centers_out(config%num_nodes))
            x_centers_out = x_center
        end if

        ! Cleanup
        deallocate(x_center, nodal_total, nodal_scatter, nodal_Q)
        deallocate(Q_source, Q_scatter, Q_scatter_old)
        deallocate(mu, w, left_mu, right_mu, left_w, right_w)
        deallocate(angular_flux, right_flux, left_flux)
        deallocate(right_avg_flux, left_avg_flux)
        deallocate(old_flux, old_avg_flux, total_avg_flux)

    end subroutine solve_flux

    subroutine gauss_legendre(n, x, w)
        ! Gauss-Legendre quadrature points and weights on [-1, 1]
        integer(int64), intent(in) :: n
        real(real64), intent(out) :: x(n)
        real(real64), intent(out) :: w(n)

        integer(int64) :: i, j, m
        real(real64) :: z, z1, p1, p2, p3, pp
        real(real64), parameter :: pi = 3.141592653589793_real64
        real(real64), parameter :: eps = 1.0e-14_real64

        m = (n + 1) / 2

        do i = 1, m
            ! Initial guess for root
            z = cos(pi * (real(i, real64) - 0.25_real64) / (real(n, real64) + 0.5_real64))

            ! Newton-Raphson iteration
            do
                p1 = 1.0_real64
                p2 = 0.0_real64

                ! Evaluate Legendre polynomial
                do j = 1, n
                    p3 = p2
                    p2 = p1
                    p1 = ((2.0_real64 * real(j, real64) - 1.0_real64) * z * p2 - &
                          (real(j, real64) - 1.0_real64) * p3) / real(j, real64)
                end do

                ! Derivative of Legendre polynomial
                pp = real(n, real64) * (z * p1 - p2) / (z * z - 1.0_real64)

                z1 = z
                z = z1 - p1 / pp

                if (abs(z - z1) < eps) exit
            end do

            x(i) = -z
            x(n + 1 - i) = z
            w(i) = 2.0_real64 / ((1.0_real64 - z * z) * pp * pp)
            w(n + 1 - i) = w(i)
        end do

    end subroutine gauss_legendre

end module solver