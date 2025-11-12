PROGRAM FDISCROD
    use, intrinsic :: iso_fortran_env, only: int64, real64
    use material_mod
    use settings_mod
    use solver
    use json_reader
    use json_writer
    implicit none

    ! Variables
    type(Material), allocatable :: materials(:)
    type(Settings) :: sim_settings
    real(real64), allocatable :: x_edges(:), total_flux(:), current(:)
    real(real64), allocatable :: angular_flux(:,:), mu(:), w(:), x_centers(:)
    integer(int64) :: i, num_materials
    character(len=256) :: input_file, output_file, arg
    integer :: num_args, iarg
    logical :: has_output

    ! Parse command-line arguments
    input_file = './input.json'  ! Default
    output_file = ''
    has_output = .false.

    num_args = command_argument_count()
    iarg = 1
    do while (iarg <= num_args)
        call get_command_argument(iarg, arg)
        select case (trim(arg))
        case ('-i', '--input')
            iarg = iarg + 1
            if (iarg <= num_args) then
                call get_command_argument(iarg, input_file)
            else
                print *, 'Error: -i/--input requires an argument'
                stop 1
            end if
        case ('-o', '--output')
            iarg = iarg + 1
            if (iarg <= num_args) then
                call get_command_argument(iarg, output_file)
                has_output = .true.
            else
                print *, 'Error: -o/--output requires an argument'
                stop 1
            end if
        case ('-h', '--help')
            print *, 'Usage: fdiscord [-i INPUT] [-o OUTPUT]'
            print *, ''
            print *, 'FDiscord - Fortran Discrete Ordinates Transport Solver'
            print *, ''
            print *, 'Options:'
            print *, '  -i, --input   Input JSON file (default: ../input.json)'
            print *, '  -o, --output  Output JSON file (optional)'
            print *, '  -h, --help    Show this help message'
            stop 0
        case default
            ! Assume it's the input file (for backward compatibility)
            input_file = trim(arg)
        end select
        iarg = iarg + 1
    end do

    ! Load configuration from JSON
    call load_config_from_json(input_file, materials, num_materials, sim_settings)

    print *, 'Loaded configuration from: ', trim(input_file)
    print *, ''

    ! Print material information
    print *, '========================================='
    print *, 'Material Configuration:'
    print *, '========================================='
    do i = 1, size(materials)
        print '(A,I1,A,A)', 'Material ', i, ': ', trim(materials(i)%name)
        print '(A,F8.4,A,F8.4,A)', '  Bounds: [', materials(i)%left_bound, ', ', &
              materials(i)%right_bound, ']'
        print '(A,F8.4)', '  Total XS:      ', materials(i)%total
        print '(A,F8.4)', '  Scatter XS:    ', materials(i)%scatter
        print '(A,F8.4)', '  Absorption XS: ', materials(i)%absorption
        print '(A,F8.4)', '  Source Q:      ', materials(i)%Q
        print *, ''
    end do

    ! Print settings information
    print *, '========================================='
    print *, 'Solver Settings:'
    print *, '========================================='
    print '(A,A)', '  Left BC:    ', trim(sim_settings%phiL_type)
    print '(A,A)', '  Right BC:   ', trim(sim_settings%phiR_type)
    print '(A,I5)', '  Num Nodes:  ', sim_settings%num_nodes
    print '(A,I5)', '  SN Order:   ', sim_settings%sn
    print *, ''

    ! Solve the flux (with optional outputs if needed for JSON)
    if (has_output) then
        call solve_flux(materials, int(size(materials), int64), sim_settings, &
                       x_edges, total_flux, current, angular_flux, mu, w, x_centers)
    else
        call solve_flux(materials, int(size(materials), int64), sim_settings, &
                       x_edges, total_flux, current)
    end if

    ! Print some results
    print *, '========================================='
    print *, 'Solution Summary:'
    print *, '========================================='
    print '(A,ES12.4)', '  Total flux at x=0:   ', total_flux(sim_settings%num_nodes/2 + 1)
    print '(A,ES12.4)', '  Current at x=0:      ', current(sim_settings%num_nodes/2 + 1)
    print '(A,ES12.4)', '  Max flux:            ', maxval(total_flux)
    print '(A,ES12.4)', '  Min flux:            ', minval(total_flux)
    print *, ''

    ! Write JSON output if requested
    if (has_output) then
        call write_output_json(output_file, materials, num_materials, sim_settings, &
                              x_edges, x_centers, total_flux, current, &
                              angular_flux, mu, w)
    end if

    ! Cleanup
    deallocate(materials)
    deallocate(x_edges, total_flux, current)
    if (allocated(angular_flux)) deallocate(angular_flux)
    if (allocated(mu)) deallocate(mu)
    if (allocated(w)) deallocate(w)
    if (allocated(x_centers)) deallocate(x_centers)

END PROGRAM FDISCROD