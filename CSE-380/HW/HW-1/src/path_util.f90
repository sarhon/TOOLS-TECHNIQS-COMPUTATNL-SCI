module path_util
    implicit none

    contains
    function join_path(directory, filename) result(full_path)
        character(len=*), intent(in) :: directory, filename
        character(len=256) :: full_path
        
        if (len_trim(directory) > 0) then
            if (directory(len_trim(directory):len_trim(directory)) == '/') then
                full_path = trim(directory) // trim(filename)
            else
                full_path = trim(directory) // '/' // trim(filename)
            end if
        else
            full_path = filename
        end if
    end function join_path

end module path_util