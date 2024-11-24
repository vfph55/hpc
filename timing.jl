using BenchmarkTools

function ij!(C,A,B,N)
	@inbounds for i in 1:N, j in 1:N
		C[i,j] = A[i,j] + B[i,j]
	end
end

function ji!(C,A,B,N)
	@inbounds for j in 1:N, i in 1:N
		C[i,j] = A[i,j] + B[i,j]
	end
end

function ijk!(C,A,B,N)
  @inbounds for i in 1:N, j in 1:N, k in 1:N
    C[i,j,k] = A[i,j,k] + B[i,j,k]
  end
  return nothing
end

function ikj!(C,A,B,N)
  @inbounds for i in 1:N, k in 1:N, j in 1:N
    C[i,j,k] = A[i,j,k] + B[i,j,k]
  end
  return nothing
end

function jik!(C,A,B,N)
  @inbounds for j in 1:N, i in 1:N, k in 1:N
    C[i,j,k] = A[i,j,k] + B[i,j,k]
  end
  return nothing
end

function jki!(C,A,B,N)
  @inbounds for j in 1:N, k in 1:N, i in 1:N
    C[i,j,k] = A[i,j,k] + B[i,j,k]
  end
  return nothing
end

function kij!(C,A,B,N)
  @inbounds for k in 1:N, i in 1:N, j in 1:N
    C[i,j,k] = A[i,j,k] + B[i,j,k]
  end
  return nothing
end

function kji!(C,A,B,N)
  @inbounds for k in 1:N, j in 1:N, i in 1:N
    C[i,j,k] = A[i,j,k] + B[i,j,k]
  end
  return nothing
end

function idk!(C,A,B,N)
   @inbounds for I in eachindex(C,A,B)
      C[I] = A[I] + B[I]
   end
   return nothing
end

function bc!(C,A,B,N)
    C.=A.+B
    return nothing
end

function main(ARGS)

    N = parse(Int, ARGS[1])
    M = parse(Int, ARGS[2])

    BenchmarkTools.DEFAULT_PARAMETERS.evals = M
    #
    #   3D version
    #
    A = rand(Float64,N,N,N)
    B = rand(Float64,N,N,N)
    C = rand(Float64,N,N,N)

    keys = ["BC","CI","ijk","jik","kij","kji","ikj","jki"]
    funs = [bc!, idk!, ijk!,jik!,kij!,kji!,ikj!,jki!]
    timing = Dict{String,Float64}()
    for (key, fun) in zip(keys, funs)
        timing[key] = @belapsed $fun($C,$A,$B,$N)
    end
    print("3D version:\n")
    for key in keys
        print("\t$(key):\t $(timing[key]) s\n")
    end
    #
    #   2D version
    #
    A = rand(Float64,N,N)
    B = rand(Float64,N,N)
    C = rand(Float64,N,N)
    keys = ["BC","CI","ij","ji"]
    funs = [bc!, idk!, ij!,ji!]
    timing = Dict{String,Float64}()
    for (key, fun) in zip(keys, funs)
        timing[key] = @belapsed $fun($C,$A,$B,$N)
    end
    print("2D version:\n")
    for key in keys
        print("\t$(key):\t $(timing[key]) s\n")
    end

    return nothing
end
main(ARGS)
