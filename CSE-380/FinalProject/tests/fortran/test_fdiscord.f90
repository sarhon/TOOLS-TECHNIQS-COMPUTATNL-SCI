! Unit tests for Fortran discrete ordinates transport solver (FDiscord)
program test_fdiscord
    use, intrinsic :: iso_fortran_env, only: int64, real64
    use material_mod
    use settings_mod
    use solver
    implicit none

    integer :: num_passed, num_failed, total_tests

    num_passed = 0
    num_failed = 0
    total_tests = 0

    print *, '========================================='
    print *, 'Running FDiscord Unit Tests'
    print *, '========================================='
    print *, ''

    ! Run all test suites
    call test_gauss_legendre(num_passed, num_failed, total_tests)
    call test_avg_flux_function(num_passed, num_failed, total_tests)
    call test_single_material_vacuum(num_passed, num_failed, total_tests)
    call test_single_material_reflective(num_passed, num_failed, total_tests)
    call test_three_material_vacuum(num_passed, num_failed, total_tests)
    call test_symmetry(num_passed, num_failed, total_tests)

    ! Print summary
    print *, ''
    print *, '========================================='
    print *, 'Test Summary:'
    print *, '========================================='
    print '(A,I3)', '  Total tests:  ', total_tests
    print '(A,I3)', '  Passed:       ', num_passed
    print '(A,I3)', '  Failed:       ', num_failed
    print *, ''

    if (num_failed == 0) then
        print *, 'All tests PASSED!'
    else
        print *, 'Some tests FAILED!'
        stop 1
    end if

contains

    subroutine assert_true(condition, test_name, passed, failed, total)
        logical, intent(in) :: condition
        character(len=*), intent(in) :: test_name
        integer, intent(inout) :: passed, failed, total

        total = total + 1
        if (condition) then
            passed = passed + 1
            print '(A,A)', '  [PASS] ', trim(test_name)
        else
            failed = failed + 1
            print '(A,A)', '  [FAIL] ', trim(test_name)
        end if
    end subroutine assert_true

    subroutine assert_close(val1, val2, test_name, passed, failed, total, tol)
        real(real64), intent(in) :: val1, val2
        character(len=*), intent(in) :: test_name
        integer, intent(inout) :: passed, failed, total
        real(real64), intent(in), optional :: tol
        real(real64) :: tolerance
        logical :: condition

        if (present(tol)) then
            tolerance = tol
        else
            tolerance = 1.0e-10_real64
        end if

        condition = abs(val1 - val2) < tolerance
        total = total + 1
        if (condition) then
            passed = passed + 1
            print '(A,A)', '  [PASS] ', trim(test_name)
        else
            failed = failed + 1
            print '(A,A,ES12.4,A,ES12.4)', '  [FAIL] ', trim(test_name), val1, ' vs ', val2
        end if
    end subroutine assert_close

    subroutine test_gauss_legendre(passed, failed, total)
        integer, intent(inout) :: passed, failed, total
        real(real64), allocatable :: mu(:), w(:)
        integer(int64) :: i
        real(real64) :: sum_weights
        real(real64), parameter :: expected_mu_s2 = 0.5773502691896257_real64

        print *, ''
        print *, 'Testing Gauss-Legendre Quadrature:'
        print *, '---------------------------------'

        ! Test S2
        allocate(mu(2), w(2))
        call gauss_legendre(2_int64, mu, w)

        call assert_close(abs(mu(1)), expected_mu_s2, 'S2 mu value', passed, failed, total)
        call assert_close(mu(1), -mu(2), 'S2 symmetry', passed, failed, total)
        call assert_close(w(1), w(2), 'S2 weight symmetry', passed, failed, total)

        sum_weights = sum(w)
        call assert_close(sum_weights, 2.0_real64, 'S2 weights sum to 2', passed, failed, total)

        deallocate(mu, w)

        ! Test S4
        allocate(mu(4), w(4))
        call gauss_legendre(4_int64, mu, w)

        sum_weights = sum(w)
        call assert_close(sum_weights, 2.0_real64, 'S4 weights sum to 2', passed, failed, total)

        ! Check symmetry
        call assert_close(mu(1), -mu(4), 'S4 mu symmetry 1', passed, failed, total)
        call assert_close(mu(2), -mu(3), 'S4 mu symmetry 2', passed, failed, total)
        call assert_close(w(1), w(4), 'S4 weight symmetry 1', passed, failed, total)

        deallocate(mu, w)

        ! Test S8
        allocate(mu(8), w(8))
        call gauss_legendre(8_int64, mu, w)

        sum_weights = sum(w)
        call assert_close(sum_weights, 2.0_real64, 'S8 weights sum to 2', passed, failed, total)

        deallocate(mu, w)

    end subroutine test_gauss_legendre

    subroutine test_avg_flux_function(passed, failed, total)
        integer, intent(inout) :: passed, failed, total
        real(real64) :: result, expected
        real(real64) :: xi, xe, phi_i, mu_val, Q, sigma_t

        print *, ''
        print *, 'Testing Average Flux Function:'
        print *, '------------------------------'

        ! Test case 1: Small tau (should use Taylor series)
        xi = 0.0_real64
        xe = 0.001_real64
        phi_i = 1.0_real64
        mu_val = 1.0_real64
        Q = 0.0_real64
        sigma_t = 0.001_real64

        result = avg_flux(xi, xe, phi_i, mu_val, Q, sigma_t)
        call assert_true(result > 0.0_real64, 'avg_flux positive for small tau', passed, failed, total)

        ! Test case 2: Zero source
        xi = 0.0_real64
        xe = 1.0_real64
        phi_i = 1.0_real64
        mu_val = 1.0_real64
        Q = 0.0_real64
        sigma_t = 1.0_real64

        result = avg_flux(xi, xe, phi_i, mu_val, Q, sigma_t)
        call assert_true(result <= phi_i, 'avg_flux decays without source', passed, failed, total)

    end subroutine test_avg_flux_function

    subroutine test_single_material_vacuum(passed, failed, total)
        integer, intent(inout) :: passed, failed, total
        type(Material), allocatable :: materials(:)
        type(Settings) :: config
        real(real64), allocatable :: x_edges(:), total_flux(:), current(:)
        integer(int64) :: i, mid
        real(real64) :: max_flux

        print *, ''
        print *, 'Testing Single Material Vacuum BC:'
        print *, '----------------------------------'

        ! Setup
        allocate(materials(1))
        materials(1)%name = 'source'
        materials(1)%total = 1.0_real64
        materials(1)%scatter = 0.5_real64
        materials(1)%absorption = 0.5_real64
        materials(1)%Q = 1.0_real64
        materials(1)%left_bound = -5.0_real64
        materials(1)%right_bound = 5.0_real64

        config%phiL = 0.0_real64
        config%phiR = 0.0_real64
        config%phiL_type = 'vac'
        config%phiR_type = 'vac'
        config%num_nodes = 10
        config%sn = 4

        ! Solve
        call solve_flux(materials, 1_int64, config, x_edges, total_flux, current)

        ! Tests
        call assert_true(allocated(total_flux), 'total_flux allocated', passed, failed, total)
        call assert_true(size(total_flux) == config%num_nodes + 1, 'flux size correct', &
                        passed, failed, total)

        ! Check all flux values are positive
        call assert_true(all(total_flux >= 0.0_real64), 'all flux positive', passed, failed, total)

        ! Check symmetry (approximately)
        mid = (config%num_nodes + 1) / 2
        call assert_close(total_flux(1), total_flux(config%num_nodes + 1), &
                         'flux symmetry at boundaries', passed, failed, total, 0.01_real64)

        ! Check maximum is near center
        max_flux = maxval(total_flux)
        call assert_true(total_flux(mid) >= max_flux * 0.9_real64, &
                        'max flux near center', passed, failed, total)

        ! Cleanup
        deallocate(materials, x_edges, total_flux, current)

    end subroutine test_single_material_vacuum

    subroutine test_single_material_reflective(passed, failed, total)
        integer, intent(inout) :: passed, failed, total
        type(Material), allocatable :: materials(:)
        type(Settings) :: config
        real(real64), allocatable :: x_edges(:), total_flux(:), current(:)

        print *, ''
        print *, 'Testing Single Material Reflective BC:'
        print *, '--------------------------------------'

        ! Setup
        allocate(materials(1))
        materials(1)%name = 'source'
        materials(1)%total = 1.0_real64
        materials(1)%scatter = 0.9_real64
        materials(1)%absorption = 0.1_real64
        materials(1)%Q = 1.0_real64
        materials(1)%left_bound = -5.0_real64
        materials(1)%right_bound = 5.0_real64

        config%phiL = 0.0_real64
        config%phiR = 0.0_real64
        config%phiL_type = 'ref'
        config%phiR_type = 'ref'
        config%num_nodes = 10
        config%sn = 4

        ! Solve
        call solve_flux(materials, 1_int64, config, x_edges, total_flux, current)

        ! Tests
        call assert_true(allocated(total_flux), 'total_flux allocated', passed, failed, total)

        ! Check current at boundaries is near zero
        call assert_true(abs(current(1)) < 0.1_real64, 'left current near zero', &
                        passed, failed, total)
        call assert_true(abs(current(config%num_nodes + 1)) < 0.1_real64, &
                        'right current near zero', passed, failed, total)

        ! Cleanup
        deallocate(materials, x_edges, total_flux, current)

    end subroutine test_single_material_reflective

    subroutine test_three_material_vacuum(passed, failed, total)
        integer, intent(inout) :: passed, failed, total
        type(Material), allocatable :: materials(:)
        type(Settings) :: config
        real(real64), allocatable :: x_edges(:), total_flux(:), current(:)
        real(real64) :: max_flux, min_flux

        print *, ''
        print *, 'Testing Three Material Problem:'
        print *, '-------------------------------'

        ! Setup
        allocate(materials(3))
        materials(1)%name = 'scatter1'
        materials(1)%total = 2.0_real64
        materials(1)%scatter = 1.99_real64
        materials(1)%absorption = 0.01_real64
        materials(1)%Q = 0.0_real64
        materials(1)%left_bound = -15.0_real64
        materials(1)%right_bound = -5.0_real64

        materials(2)%name = 'source'
        materials(2)%total = 1.0_real64
        materials(2)%scatter = 0.0_real64
        materials(2)%absorption = 1.0_real64
        materials(2)%Q = 1.0_real64
        materials(2)%left_bound = -5.0_real64
        materials(2)%right_bound = 5.0_real64

        materials(3)%name = 'scatter2'
        materials(3)%total = 2.0_real64
        materials(3)%scatter = 1.99_real64
        materials(3)%absorption = 0.01_real64
        materials(3)%Q = 0.0_real64
        materials(3)%left_bound = 5.0_real64
        materials(3)%right_bound = 15.0_real64

        config%phiL = 0.0_real64
        config%phiR = 0.0_real64
        config%phiL_type = 'vac'
        config%phiR_type = 'vac'
        config%num_nodes = 27
        config%sn = 8

        ! Solve
        call solve_flux(materials, 3_int64, config, x_edges, total_flux, current)

        ! Tests
        max_flux = maxval(total_flux)
        min_flux = minval(total_flux)

        call assert_close(max_flux, 0.9997443_real64, 'three material max flux', &
                         passed, failed, total, 1.0e-4_real64)
        call assert_close(min_flux, 0.0293688_real64, 'three material min flux', &
                         passed, failed, total, 1.0e-4_real64)

        ! Cleanup
        deallocate(materials, x_edges, total_flux, current)

    end subroutine test_three_material_vacuum

    subroutine test_symmetry(passed, failed, total)
        integer, intent(inout) :: passed, failed, total
        type(Material), allocatable :: materials(:)
        type(Settings) :: config
        real(real64), allocatable :: x_edges(:), total_flux(:), current(:)
        integer(int64) :: i, n
        logical :: is_symmetric

        print *, ''
        print *, 'Testing Flux Symmetry:'
        print *, '---------------------'

        ! Setup symmetric problem
        allocate(materials(1))
        materials(1)%name = 'source'
        materials(1)%total = 1.0_real64
        materials(1)%scatter = 0.0_real64
        materials(1)%absorption = 1.0_real64
        materials(1)%Q = 1.0_real64
        materials(1)%left_bound = -10.0_real64
        materials(1)%right_bound = 10.0_real64

        config%phiL = 0.0_real64
        config%phiR = 0.0_real64
        config%phiL_type = 'vac'
        config%phiR_type = 'vac'
        config%num_nodes = 20
        config%sn = 4

        ! Solve
        call solve_flux(materials, 1_int64, config, x_edges, total_flux, current)

        ! Check symmetry
        n = size(total_flux)
        is_symmetric = .true.
        do i = 1, n/2
            if (abs(total_flux(i) - total_flux(n+1-i)) > 0.01_real64) then
                is_symmetric = .false.
                exit
            end if
        end do

        call assert_true(is_symmetric, 'symmetric flux for symmetric problem', &
                        passed, failed, total)

        ! Cleanup
        deallocate(materials, x_edges, total_flux, current)

    end subroutine test_symmetry

end program test_fdiscord