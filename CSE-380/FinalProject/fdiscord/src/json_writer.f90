module json_writer
    use, intrinsic :: iso_fortran_env, only: int64, real64
    use material_mod
    use settings_mod
    implicit none
    private
    public :: write_output_json

contains

    subroutine write_output_json(filename, materials, num_materials, config, &
                                 x_edges, x_centers, total_flux, current, &
                                 angular_flux, mu, w)
        character(len=*), intent(in) :: filename
        type(Material), intent(in) :: materials(:)
        integer(int64), intent(in) :: num_materials
        type(Settings), intent(in) :: config
        real(real64), intent(in) :: x_edges(:), x_centers(:)
        real(real64), intent(in) :: total_flux(:), current(:)
        real(real64), intent(in) :: angular_flux(:,:)
        real(real64), intent(in) :: mu(:), w(:)

        integer :: unit, i, j, num_edges, num_centers, num_angles
        real(real64) :: max_flux, min_flux, flux_center, current_center

        num_edges = size(x_edges)
        num_centers = size(x_centers)
        num_angles = size(mu)

        ! Calculate summary values
        max_flux = maxval(total_flux)
        min_flux = minval(total_flux)
        flux_center = total_flux(num_edges/2 + 1)
        current_center = current(num_edges/2 + 1)

        ! Open output file
        open(newunit=unit, file=trim(filename), status='replace', action='write')

        ! Write JSON structure
        write(unit, '(A)') '{'
        write(unit, '(A)') '  "problem_setup": {'
        write(unit, '(A,F12.6,A)') '    "bounds": [', x_edges(1), ','
        write(unit, '(A,F12.6,A)') '      ', x_edges(num_edges), '],'
        write(unit, '(A,I0,A)') '    "num_nodes": ', config%num_nodes, ','
        write(unit, '(A,I0,A)') '    "sn_order": ', config%sn, ','
        write(unit, '(A,A,A)') '    "phiL_type": "', trim(config%phiL_type), '",'
        write(unit, '(A,A,A)') '    "phiR_type": "', trim(config%phiR_type), '"'
        write(unit, '(A)') '  },'

        ! Write materials
        write(unit, '(A)') '  "materials": ['
        do i = 1, num_materials
            write(unit, '(A)') '    {'
            write(unit, '(A,A,A)') '      "name": "', trim(materials(i)%name), '",'
            write(unit, '(A,F12.6,A)') '      "total": ', materials(i)%total, ','
            write(unit, '(A,F12.6,A)') '      "scatter": ', materials(i)%scatter, ','
            write(unit, '(A,F12.6,A)') '      "absorption": ', materials(i)%absorption, ','
            write(unit, '(A,F12.6,A)') '      "Q": ', materials(i)%Q, ','
            write(unit, '(A,F12.6,A)') '      "bounds": [', materials(i)%left_bound, ','
            write(unit, '(A,F12.6,A)') '        ', materials(i)%right_bound, ']'
            if (i < num_materials) then
                write(unit, '(A)') '    },'
            else
                write(unit, '(A)') '    }'
            end if
        end do
        write(unit, '(A)') '  ],'

        ! Write quadrature
        write(unit, '(A)') '  "quadrature": {'
        write(unit, '(A)', advance='no') '    "mu": ['
        do i = 1, num_angles
            if (i < num_angles) then
                write(unit, '(ES23.16,A)', advance='no') mu(i), ', '
            else
                write(unit, '(ES23.16,A)') mu(i), '],'
            end if
        end do
        write(unit, '(A)', advance='no') '    "weights": ['
        do i = 1, num_angles
            if (i < num_angles) then
                write(unit, '(ES23.16,A)', advance='no') w(i), ', '
            else
                write(unit, '(ES23.16,A)') w(i), ']'
            end if
        end do
        write(unit, '(A)') '  },'

        ! Write mesh
        write(unit, '(A)') '  "mesh": {'
        write(unit, '(A)', advance='no') '    "x_edges": ['
        do i = 1, num_edges
            if (i < num_edges) then
                write(unit, '(ES23.16,A)', advance='no') x_edges(i), ', '
            else
                write(unit, '(ES23.16,A)') x_edges(i), '],'
            end if
        end do
        write(unit, '(A)', advance='no') '    "x_centers": ['
        do i = 1, num_centers
            if (i < num_centers) then
                write(unit, '(ES23.16,A)', advance='no') x_centers(i), ', '
            else
                write(unit, '(ES23.16,A)') x_centers(i), ']'
            end if
        end do
        write(unit, '(A)') '  },'

        ! Write solution
        write(unit, '(A)') '  "solution": {'
        write(unit, '(A)', advance='no') '    "scalar_flux": ['
        do i = 1, num_edges
            if (i < num_edges) then
                write(unit, '(ES23.16,A)', advance='no') total_flux(i), ', '
            else
                write(unit, '(ES23.16,A)') total_flux(i), '],'
            end if
        end do
        write(unit, '(A)', advance='no') '    "current": ['
        do i = 1, num_edges
            if (i < num_edges) then
                write(unit, '(ES23.16,A)', advance='no') current(i), ', '
            else
                write(unit, '(ES23.16,A)') current(i), '],'
            end if
        end do
        write(unit, '(A)') '    "angular_flux": ['
        do i = 1, num_angles
            write(unit, '(A)', advance='no') '      ['
            do j = 1, num_edges
                if (j < num_edges) then
                    write(unit, '(ES23.16,A)', advance='no') angular_flux(i,j), ', '
                else
                    if (i < num_angles) then
                        write(unit, '(ES23.16,A)') angular_flux(i,j), '],'
                    else
                        write(unit, '(ES23.16,A)') angular_flux(i,j), ']'
                    end if
                end if
            end do
        end do
        write(unit, '(A)') '    ]'
        write(unit, '(A)') '  },'

        ! Write summary
        write(unit, '(A)') '  "summary": {'
        write(unit, '(A,ES23.16,A)') '    "max_flux": ', max_flux, ','
        write(unit, '(A,ES23.16,A)') '    "min_flux": ', min_flux, ','
        write(unit, '(A,ES23.16,A)') '    "flux_at_center": ', flux_center, ','
        write(unit, '(A,ES23.16)') '    "current_at_center": ', current_center
        write(unit, '(A)') '  }'
        write(unit, '(A)') '}'

        close(unit)

        print *, 'Results written to: ', trim(filename)

    end subroutine write_output_json

end module json_writer