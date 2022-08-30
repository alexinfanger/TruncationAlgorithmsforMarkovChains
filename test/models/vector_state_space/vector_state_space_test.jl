function testing_code_matrix(array_of_sizes::AbstractArray)
    nd = size(array_of_sizes,1)
    k = reduce(*, array_of_sizes)

    mat = zeros(k, nd)
    num_inner = 1
    for i=1:nd
        v = collect(1:array_of_sizes[end-i+1])
        if i>1
            num_inner = num_inner * array_of_sizes[end-i+2]
        end
        num_outer = reduce(*, array_of_sizes[1:end-i])
        mat[:,nd-i+1] = repeat(v, inner=(num_inner,1), outer=(num_outer,1))
    end
    return mat
end

function testing_code_matrix_using_functions(array_of_sizes::AbstractArray)
    nd = size(array_of_sizes,1)
    k = reduce(*, array_of_sizes)
    mat = zeros(k, nd)

    vss = VectorStateSpace(array_of_sizes)

    for i =1:k
        mat[i,:] = ind_to_vec(vss, i)
    end
    return mat
end

function test_vss(array_of_sizes::AbstractArray)
    println("Starting Test...")
    mat = testing_code_matrix(array_of_sizes)
    vss = VectorStateSpace(array_of_sizes)
    for i=1:size(mat,1)
        if (mat[i,:]!=ind_to_vec(vss, i))
            println(i)
            println("ind to vec disagrees with mat:")
            println("ind_to_vec:  ")
            println(ind_to_vec(vss,i))
            println("mat:  ")
            println(mat[i,:])
            return
        end
        if (vec_to_ind(vss, mat[i,:]) !=i)
            print("vec_to_ind disagrees with mat")
            return
        end
    end
    println("Test Success.")
end

test_vss([1,2,3])
test_vss([20,3])
test_vss([20,5,1,1])
test_vss([20,5,1,0])




