module json_reader
    use, intrinsic :: iso_fortran_env, only: int64, real64
    use material_mod
    use settings_mod
    implicit none
    private
    public :: load_config_from_json

contains

    subroutine load_config_from_json(filename, materials, num_materials, config)
        character(len=*), intent(in) :: filename
        type(Material), allocatable, intent(out) :: materials(:)
        integer(int64), intent(out) :: num_materials
        type(Settings), intent(out) :: config

        character(len=20000) :: json_content
        integer :: unit, ios
        character(len=1000) :: line
        integer :: num_mats, i
        character(len=100) :: name, phiL_type, phiR_type
        real(real64) :: total, scatter, Q, bound_left, bound_right
        real(real64) :: phiL, phiR
        integer(int64) :: num_nodes, sn

        ! Open and read JSON file
        open(newunit=unit, file=trim(filename), status='old', action='read', iostat=ios)
        if (ios /= 0) then
            print *, 'Error: Cannot open file ', trim(filename)
            stop
        end if

        ! Read entire file
        json_content = ''
        do
            read(unit, '(A)', iostat=ios) line
            if (ios /= 0) exit
            json_content = trim(json_content) // ' ' // trim(adjustl(line))
        end do
        close(unit)

        ! Count materials - simply count commas between materials + 1
        ! Assumes materials are listed as array elements separated by },\n{
        num_mats = count_occurrences(json_content(index(json_content, '"materials"'):index(json_content, '"settings"')), '},')
        if (num_mats > 0) num_mats = num_mats + 1  ! Add 1 for last material
        if (num_mats == 0) num_mats = 1  ! At least one material if materials section exists

        num_materials = num_mats
        allocate(materials(num_mats))

        ! Parse each material
        do i = 1, num_mats
            call parse_material_simple(json_content, i, name, total, scatter, Q, &
                                      bound_left, bound_right)
            call NewMaterial(materials(i), name, total, scatter, Q, &
                           bound_left, bound_right)
        end do

        ! Parse settings
        call parse_settings_simple(json_content, phiL_type, phiR_type, phiL, phiR, &
                                  num_nodes, sn)
        call NewSettings(config, phiL, phiR, phiL_type, phiR_type, num_nodes, sn)

    end subroutine load_config_from_json

    function count_occurrences(str, pattern) result(count)
        character(len=*), intent(in) :: str, pattern
        integer :: count, pos, offset

        count = 0
        offset = 1

        do
            pos = index(str(offset:), pattern)
            if (pos == 0) exit
            count = count + 1
            offset = offset + pos + len(pattern) - 1
            if (offset > len_trim(str)) exit
        end do
    end function count_occurrences

    subroutine parse_material_simple(json_str, mat_index, name, total, scatter, Q, &
                                     bound_left, bound_right)
        character(len=*), intent(in) :: json_str
        integer, intent(in) :: mat_index
        character(len=*), intent(out) :: name
        real(real64), intent(out) :: total, scatter, Q, bound_left, bound_right

        integer :: mat_start, mat_end, search_start, i
        character(len=2000) :: mat_section

        ! Find materials section
        search_start = index(json_str, '"materials"')
        mat_start = search_start

        ! Skip to the mat_index-th material block
        do i = 1, mat_index
            mat_start = index(json_str(mat_start+1:), '{') + mat_start
        end do

        ! Find end of this material block
        mat_end = index(json_str(mat_start+1:), '}') + mat_start

        mat_section = json_str(mat_start:mat_end)

        ! Extract fields
        call get_string_field(mat_section, 'name', name)
        call get_real_field(mat_section, 'total', total)
        call get_real_field(mat_section, 'scatter', scatter)
        call get_real_field(mat_section, 'Q', Q)
        call get_array_field(mat_section, 'bounds', bound_left, bound_right)

    end subroutine parse_material_simple

    subroutine parse_settings_simple(json_str, phiL_type, phiR_type, phiL, phiR, &
                                     num_nodes, sn)
        character(len=*), intent(in) :: json_str
        character(len=*), intent(out) :: phiL_type, phiR_type
        real(real64), intent(out) :: phiL, phiR
        integer(int64), intent(out) :: num_nodes, sn

        integer :: settings_start
        character(len=2000) :: settings_section

        ! Find settings block
        settings_start = index(json_str, '"settings"')
        settings_section = json_str(settings_start:)

        ! Extract fields
        call get_string_field(settings_section, 'phiL_type', phiL_type)
        call get_string_field(settings_section, 'phiR_type', phiR_type)
        call get_real_field(settings_section, 'phiL', phiL)
        call get_real_field(settings_section, 'phiR', phiR)
        call get_int_field(settings_section, 'num_nodes', num_nodes)
        call get_int_field(settings_section, 'sn', sn)

    end subroutine parse_settings_simple

    subroutine get_string_field(json_str, key, value)
        character(len=*), intent(in) :: json_str, key
        character(len=*), intent(out) :: value

        integer :: key_pos, start_quote, end_quote
        character(len=200) :: search_key

        search_key = '"' // trim(key) // '"'
        key_pos = index(json_str, trim(search_key))

        if (key_pos > 0) then
            ! Find the value after the colon
            start_quote = index(json_str(key_pos:), ':') + key_pos
            start_quote = index(json_str(start_quote:), '"') + start_quote
            end_quote = index(json_str(start_quote:), '"') + start_quote - 2

            value = json_str(start_quote:end_quote)
        else
            value = ''
        end if
    end subroutine get_string_field

    subroutine get_real_field(json_str, key, value)
        character(len=*), intent(in) :: json_str, key
        real(real64), intent(out) :: value

        integer :: key_pos, colon_pos, value_end
        character(len=200) :: search_key, value_str

        search_key = '"' // trim(key) // '"'
        key_pos = index(json_str, trim(search_key))

        if (key_pos > 0) then
            colon_pos = index(json_str(key_pos:), ':') + key_pos

            ! Find end of value (comma or closing brace)
            value_end = index(json_str(colon_pos:), ',')
            if (value_end == 0 .or. value_end > index(json_str(colon_pos:), '}')) then
                value_end = index(json_str(colon_pos:), '}')
            end if
            value_end = value_end + colon_pos - 2

            value_str = adjustl(json_str(colon_pos+1:value_end))
            read(value_str, *, iostat=key_pos) value
            if (key_pos /= 0) value = 0.0_real64
        else
            value = 0.0_real64
        end if
    end subroutine get_real_field

    subroutine get_int_field(json_str, key, value)
        character(len=*), intent(in) :: json_str, key
        integer(int64), intent(out) :: value

        integer :: key_pos, colon_pos, value_end
        character(len=200) :: search_key, value_str

        search_key = '"' // trim(key) // '"'
        key_pos = index(json_str, trim(search_key))

        if (key_pos > 0) then
            colon_pos = index(json_str(key_pos:), ':') + key_pos

            ! Find end of value
            value_end = index(json_str(colon_pos:), ',')
            if (value_end == 0 .or. value_end > index(json_str(colon_pos:), '}')) then
                value_end = index(json_str(colon_pos:), '}')
            end if
            value_end = value_end + colon_pos - 2

            value_str = adjustl(json_str(colon_pos+1:value_end))
            read(value_str, *, iostat=key_pos) value
            if (key_pos /= 0) value = 0
        else
            value = 0
        end if
    end subroutine get_int_field

    subroutine get_array_field(json_str, key, val1, val2)
        character(len=*), intent(in) :: json_str, key
        real(real64), intent(out) :: val1, val2

        integer :: key_pos, bracket_start, bracket_end, comma_pos
        character(len=200) :: search_key, array_str, val1_str, val2_str

        search_key = '"' // trim(key) // '"'
        key_pos = index(json_str, trim(search_key))

        if (key_pos > 0) then
            bracket_start = index(json_str(key_pos:), '[') + key_pos
            bracket_end = index(json_str(bracket_start:), ']') + bracket_start - 2

            array_str = json_str(bracket_start:bracket_end)
            comma_pos = index(array_str, ',')

            val1_str = adjustl(array_str(1:comma_pos-1))
            val2_str = adjustl(array_str(comma_pos+1:))

            read(val1_str, *) val1
            read(val2_str, *) val2
        else
            val1 = 0.0_real64
            val2 = 0.0_real64
        end if
    end subroutine get_array_field

end module json_reader