
struct RotationError{T,D} <: StaticVector{3,T}
    err::SVector{3,T}
    map::D
    @inline function RotationError(err::SVector{3,T}, map::D) where {T,D <: ErrorMap}
        new{T,D}(err, map)
    end
end

# Convert an error back to a rotation
function inverse_map(e::RotationError)::Rotation
    e.map(e.err)
end

# Compute the error
function rotation_error(R1::Rotation, R2::Rotation, error_map::ErrorMap)
    return RotationError(error_map(R2\R1), error_map)
end

function rotation_error(R1::Rotation, R2::Rotation, error_map::IdentityMap)
    err = SVector(R2\R1)
    if length(err) != 3
        throw(ArgumentError("R2\\R1 must be a three-dimensional parameterization, got $(length(err))"))
    end
    return RotationError(err, error_map)
end

#   set the default error map to the CayleyMap
@inline ⊖(R1::Rotation, R2::Rotation) = rotation_error(R1, R2, CayleyMap())

#   default to the identity map for Rodrigues Params and MRPs
@inline ⊖(R1::RodriguesParam, R2::RodriguesParam) = rotation_error(R1, R2, IdentityMap())
@inline ⊖(R1::MRP,            R2::MRP           ) = rotation_error(R1, R2, IdentityMap())


# Static Arrays interface
function (::Type{E})(t::NTuple{3}) where E <: RotationError
    E(t[1], t[2], t[3])
end
Base.@propagate_inbounds Base.getindex(e::RotationError, i::Int) = e.err[i]
@inline Base.Tuple(e::RotationError) = Tuple(e.err)

# Compose a rotation with an error
function add_error(R1::Rotation, e::RotationError)
    R1 * inverse_map(e)
end

function add_error(R1::R, e::RotationError{<:Any, IdentityMap}) where R <: Rotation
    # must be able to construct R from a SVector{3}
    R1 * R(e.err)
end

@inline ⊕(R1::Rotation, e::RotationError) = add_error(R1, e)
