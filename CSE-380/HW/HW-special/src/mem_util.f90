module mem_util
    use, intrinsic :: iso_fortran_env, only : int64, real64
    implicit none

    contains
    pure function pbytes(bytes) result(s)
        integer(int64), intent(in) :: bytes
        character(:), allocatable :: s
        real(real64) :: value
        integer :: index
        character(len=3), parameter :: units(4) = ['B  ', 'KiB', 'MiB', 'GiB']
        character(len=32) :: buf

        value = real(bytes, real64) ! int32 -> real64
        index = 1

        if (value >= 1024.0_real64) then
            value = value / 1024.0_real64
            index = 2

            if (value >= 1024.0_real64) then
                value = value / 1024.0_real64
                index = 3

                if (value >= 1024.0_real64) then
                    value = value / 1024.0_real64
                    index = 4

                end if
            end if
        end if

        write(buf, '(F0.1,1X,A3)') value, units(index)
        
        s = trim(adjustl(buf))
    end function pbytes
end module mem_util