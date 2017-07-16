if !isdefined(Symbol("@stderrcapture"))
    macro stderrcapture(block)
        quote
            if ccall(:jl_generating_output, Cint, ()) == 0
                errororiginal = STDERR;
                (errR, errW) = redirect_stderr();
                errorreader = @async readstring(errR);
                evalvalue = $(esc(block))
                redirect_stderr(errororiginal);
                close(errW);
                close(errR);
                return evalvalue
            end
        end
    end
end

include("testfftrf.jl")
include("testrpcga.jl")
include("testrmf.jl")
include("testfd.jl")

:passed