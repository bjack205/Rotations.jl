import Base: +, -, *, /, \, exp, log, ≈, ==, inv, conj

abstract type QuatMap end

"""
    UnitQuaternion{T,D} <: Rotation

4-parameter attitute representation that is singularity-free. Quaternions with unit norm
represent a double-cover of SO(3). The `UnitQuaternion` does NOT strictly enforce the unit
norm constraint, but certain methods will assume you have a unit quaternion. The
`UnitQuaternion` type is parameterized by the linearization method, which maps quaternions
to the 3D plane tangent to the 4D unit sphere. Follows the Hamilton convention for quaternions.

There are currently 4 methods supported:
* `VectorPart` - uses the vector (or imaginary) part of the quaternion
* `ExponentialMap` - the most common approach, uses the exponential and logarithmic maps
* `CayleyMap` - or Rodrigues parameters (aka Gibbs vectors).
* `MRPMap` - or Modified Rodrigues Parameter, is a sterographic projection of the 4D unit sphere
onto the plane tangent to either the positive or negative real poles.

# Constructors
```julia
UnitQuaternion(args...)  # defaults to `CayleyMay`
UnitQuaternion{T<:Real}(args...)
UnitQuaternion{D<:QuatMap}(args...)
UnitQuaternion{T,D}(args...)
```
where `args...` can be any of the following:
- `w,x,y,z` specifying the scalar (real) part `w` and the vector (imaginary) part `x,y,z`
- `AbstractVector` with elements `[w,x,y,z]` as the first 4 elements
- `StaticVector{3}` with elements `[x,y,z]` and `w = 0`
"""
struct UnitQuaternion{T,D<:QuatMap} <: Rotation{3,T}
    w::T
    x::T
    y::T
    z::T

    @inline function UnitQuaternion{T,D}(w, x, y, z, normalize::Bool = true) where {T,D}
        if normalize
            inorm = inv(sqrt(w*w + x*x + y*y + z*z))
            new{T,D}(w*inorm, x*inorm, y*inorm, z*inorm)
        else
            new{T,D}(w, x, y, z)
        end
    end

    UnitQuaternion{T,D}(q::UnitQuaternion) where {T,D} = new{T,D}(q.w, q.x, q.y, q.z)
end

include("quaternion_maps.jl")

# ~~~~~~~~~~~~~~~ Constructors ~~~~~~~~~~~~~~~ #
# Use default map
UnitQuaternion(w::W,x::X,y::Y,z::Z, normalize::Bool = true) where {W,X,Y,Z} =
    UnitQuaternion{promote_type(W,X,Y,Z),DEFAULT_QMAP}(w,x,y,z, normalize)
(::Type{UnitQuaternion{T}})(w, x, y, z, normalize::Bool = true) where T =
    UnitQuaternion{T,DEFAULT_QMAP}(w, x, y, z, normalize)

# Provide a map
UnitQuaternion{D}(w::W,x::X,y::Y,z::Z, normalize::Bool = true) where {W,X,Y,Z,D<:QuatMap} =
    UnitQuaternion{promote_type(W,X,Y,Z),D}(w,x,y,z, normalize)

# Pass in Vectors
(::Type{Q})(q::SVector{4}, normalize::Bool = true) where {Q <: UnitQuaternion} =
    Q(q[1], q[2], q[3], q[4], normalize)
(::Type{Q})(q::SVector{3,T}, normalize::Bool = true) where {T,Q <: UnitQuaternion} =
    Q(zero(T), q[1], q[2], q[3], normalize)

# Copy constructors
UnitQuaternion(q::UnitQuaternion) = q
UnitQuaternion{D}(q::UnitQuaternion) where D = UnitQuaternion{D}(q.w, q.x, q.y, q.z)

# UnitQuaternion <=> Quat
(::Type{Q})(q::Quat) where Q <: UnitQuaternion = Q(q.w, q.x, q.y, q.z, false)
(::Type{Q})(q::UnitQuaternion) where Q <: Quat = Q(q.w, q.x, q.y, q.z, false)
const AllQuats{T} = Union{<:Quat{T}, <:UnitQuaternion{T}}


# ~~~~~~~~~~~~~~~ StaticArrays Interface ~~~~~~~~~~~~~~~ #
function (::Type{Q})(t::NTuple{9}) where Q<:UnitQuaternion
    #=
    This function solves the system of equations in Section 3.1
    of https://arxiv.org/pdf/math/0701759.pdf. This cheap method
    only works for matrices that are already orthonormal (orthogonal
    and unit length columns). The nearest orthonormal matrix can
    be found by solving Wahba's problem:
    https://en.wikipedia.org/wiki/Wahba%27s_problem as shown below.

    not_orthogonal = randn(3,3)
    u,s,v = svd(not_orthogonal)
    is_orthogoral = u * diagm([1, 1, sign(det(u * transpose(v)))]) * transpose(v)
    =#

    a = 1 + t[1] + t[5] + t[9]
    b = 1 + t[1] - t[5] - t[9]
    c = 1 - t[1] + t[5] - t[9]
    d = 1 - t[1] - t[5] + t[9]
    max_abcd = max(a, b, c, d)
    if a == max_abcd
        b = t[6] - t[8]
        c = t[7] - t[3]
        d = t[2] - t[4]
    elseif b == max_abcd
        a = t[6] - t[8]
        c = t[2] + t[4]
        d = t[7] + t[3]
    elseif c == max_abcd
        a = t[7] - t[3]
        b = t[2] + t[4]
        d = t[6] + t[8]
    else
        a = t[2] - t[4]
        b = t[7] + t[3]
        c = t[6] + t[8]
    end
    return Q(a, b, c, d)
end


function Base.getindex(q::UnitQuaternion, i::Int)
    if i == 1
        ww = (q.w * q.w)
        xx = (q.x * q.x)
        yy = (q.y * q.y)
        zz = (q.z * q.z)

        ww + xx - yy - zz
    elseif i == 2
        xy = (q.x * q.y)
        zw = (q.w * q.z)

        2 * (xy + zw)
    elseif i == 3
        xz = (q.x * q.z)
        yw = (q.y * q.w)

        2 * (xz - yw)
    elseif i == 4
        xy = (q.x * q.y)
        zw = (q.w * q.z)

        2 * (xy - zw)
    elseif i == 5
        ww = (q.w * q.w)
        xx = (q.x * q.x)
        yy = (q.y * q.y)
        zz = (q.z * q.z)

        ww - xx + yy - zz
    elseif i == 6
        yz = (q.y * q.z)
        xw = (q.w * q.x)

        2 * (yz + xw)
    elseif i == 7
        xz = (q.x * q.z)
        yw = (q.y * q.w)

        2 * (xz + yw)
    elseif i == 8
        yz = (q.y * q.z)
        xw = (q.w * q.x)

        2 * (yz - xw)
    elseif i == 9
        ww = (q.w * q.w)
        xx = (q.x * q.x)
        yy = (q.y * q.y)
        zz = (q.z * q.z)

        ww - xx - yy + zz
    else
        throw(BoundsError(r,i))
    end
end

function Base.Tuple(q::UnitQuaternion)
    ww = (q.w * q.w)
    xx = (q.x * q.x)
    yy = (q.y * q.y)
    zz = (q.z * q.z)
    xy = (q.x * q.y)
    zw = (q.w * q.z)
    xz = (q.x * q.z)
    yw = (q.y * q.w)
    yz = (q.y * q.z)
    xw = (q.w * q.x)

    # initialize rotation part
    return (ww + xx - yy - zz,
            2 * (xy + zw),
            2 * (xz - yw),
            2 * (xy - zw),
            ww - xx + yy - zz,
            2 * (yz + xw),
            2 * (xz + yw),
            2 * (yz - xw),
            ww - xx - yy + zz)
end

# ~~~~~~~~~~~~~~~ Getters ~~~~~~~~~~~~~~~ #
map_type(::UnitQuaternion{T,D}) where {T,D} = D
map_type(::Type{UnitQuaternion{T,D}}) where {T,D} = D

scalar(q::UnitQuaternion) = q.w
vector(q::UnitQuaternion{T}) where T = SVector{3,T}(q.x, q.y, q.z)

SVector(q::UnitQuaternion{T}) where T = SVector{4,T}(q.w, q.x, q.y, q.z)

# ~~~~~~~~~~~~~~~ Initializers ~~~~~~~~~~~~~~~ #
Base.rand(::Type{<:UnitQuaternion{T,D}}) where {T,D} =
    normalize(UnitQuaternion{T,D}(randn(T), randn(T), randn(T), randn(T)))
Base.rand(::Type{UnitQuaternion{T}}) where T = Base.rand(UnitQuaternion{T,DEFAULT_QMAP})
Base.rand(::Type{UnitQuaternion}) = Base.rand(UnitQuaternion{Float64,DEFAULT_QMAP})
Base.zero(::Type{Q}) where Q<:UnitQuaternion = Q(I)
Base.zero(q::Q) where Q<:UnitQuaternion = Q(I)
@inline Base.one(::Type{Q}) where Q <: UnitQuaternion = Q(1.0, 0.0, 0.0, 0.0)


# ~~~~~~~~~~~~~~~ Math Operations ~~~~~~~~~~~~~~~ #

# Inverses
conj(q::UnitQuaternion{T,D}) where {T,D} = UnitQuaternion{T,D}(q.w, -q.x, -q.y, -q.z)
inv(q::UnitQuaternion) = conj(q)
(-)(q::UnitQuaternion{T,D}) where {T,D} = UnitQuaternion{T,D}(-q.w, -q.x, -q.y, -q.z)

# Norms
LinearAlgebra.norm(q::UnitQuaternion) = sqrt(q.w^2 + q.x^2 + q.y^2 + q.z^2)
vecnorm(q::UnitQuaternion) = sqrt(q.x^2 + q.y^2 + q.z^2)

function LinearAlgebra.normalize(q::UnitQuaternion{T,D}) where {T,D}
    n = 1/norm(q)
    UnitQuaternion{T,D}(q.w*n, q.x*n, q.y*n, q.z*n)
end

# Identity
(::Type{Q})(I::UniformScaling) where Q <: UnitQuaternion = one(Q)

# # Equality
# (≈)(q::UnitQuaternion, u::UnitQuaternion) = q.w ≈ u.w && q.x ≈ u.x && q.y ≈ u.y && q.z ≈ u.z
# (==)(q::UnitQuaternion, u::UnitQuaternion) = q.w == u.w && q.x == u.x && q.y == u.y && q.z == u.z

# Exponentials and Logarithms
function exp(q::UnitQuaternion{T,D}) where {T,D}
    θ = vecnorm(q)
    sθ,cθ = sincos(θ)
    es = exp(q.w)
    M = es*sθ/θ
    UnitQuaternion{T,D}(es*cθ, q.x*M, q.y*M, q.z*M, false)
end

function expm(ϕ::SVector{3,T}) where T
    θ = norm(ϕ)
    sθ,cθ = sincos(θ/2)
    M = 0.5*sinc(θ/2π)
    UnitQuaternion{T,ExponentialMap}(cθ, ϕ[1]*M, ϕ[2]*M, ϕ[3]*M, false)
end

function log(q::UnitQuaternion{T,D}, eps=1e-6) where {T,D}
    # Assumes unit quaternion
    θ = vecnorm(q)
    if θ > eps
        M = atan(θ, q.w)/θ
    else
        M = (1-(θ^2/(3q.w^2)))/q.w
    end
    UnitQuaternion{T,D}(0.0, q.x*M, q.y*M, q.z*M, false)
end

function logm(q::UnitQuaternion{T}) where T
    # Assumes unit quaternion
    q = log(q)
    SVector{3,T}(2*q.x, 2*q.y, 2*q.z)
end

# Composition
"""
    (*)(q::UnitQuaternion, w::UnitQuaternion)

Quternion Composition

Equivalent to
```julia
Lmult(q) * SVector(w)
Rmult(w) * SVector(q)
```

Sets the output mapping equal to the mapping of `w`
"""
function (*)(q::UnitQuaternion{T1,D1}, w::UnitQuaternion{T2,D2}) where {T1,T2,D1,D2}
    T = promote_type(T1, T2)
    D = D2
    UnitQuaternion{T,D}(q.w * w.w - q.x * w.x - q.y * w.y - q.z * w.z,
                        q.w * w.x + q.x * w.w + q.y * w.z - q.z * w.y,
                        q.w * w.y - q.x * w.z + q.y * w.w + q.z * w.x,
                        q.w * w.z + q.x * w.y - q.y * w.x + q.z * w.w, false)
end

"""
    (*)(q::UnitQuaternion, r::StaticVector)

Rotate a vector

Equivalent to `Hmat()' Lmult(q) * Rmult(q)' Hmat() * r`
"""
function Base.:*(q::UnitQuaternion{Tq}, r::SVector{3}) where Tq
    w = q.w
    v = vector(q)
    (w^2 - v'v)*r + 2*v*(v'r) + 2*w*cross(v,r)
end

"""
    (*)(q::UnitQuaternion, w::Real)

Scalar multiplication of a quaternion. Breaks unit norm.
"""
function (*)(q::Q, w::Real) where Q<:UnitQuaternion
    return Q(q.w*w, q.x*w, q.y*w, q.z*w, false)
end
(*)(w::Real, q::Q) where Q<:UnitQuaternion = q*w



"Inverted composition. Equivalent to inv(q1)*q2"
(\)(q1::UnitQuaternion, q2::UnitQuaternion) = conj(q1)*q2

"Inverted composition. Equivalent to q1*inv(q2)"
(/)(q1::UnitQuaternion, q2::UnitQuaternion) = q1*conj(q2)

"Inverted rotation. Equivalent to inv(q)*r"
(\)(q::UnitQuaternion, r::SVector{3}) = conj(q)*r


# ~~~~~~~~~~~~~~~ Quaternion Differences ~~~~~~~~~~~~~~~ #
function (⊖)(q::UnitQuaternion{T,D}, q0::UnitQuaternion) where {T,D}
    D(q0\q)
end

function (⊖)(q::UnitQuaternion{T,IdentityMap}, q0::UnitQuaternion) where {T}
    SVector(q) - SVector(q0)
    # return SVector(q0\q)
end

# ~~~~~~~~~~~~~~~ Kinematics ~~~~~~~~~~~~~~~ $
"""
    kinematics(R::Rotation{3}, ω::AbstractVector)

The time derivative of the rotation R, according to the definition
    ``Ṙ = \\lim_{Δt → 0} \\frac{q(t + Δt) - q(t)}{Δt}``
"""
function kinematics(q::UnitQuaternion{T,D}, ω::AbstractVector) where {T,D}
    0.5*SVector(q*UnitQuaternion{T,D}(0.0, ω[1], ω[2], ω[3]))
end

# ~~~~~~~~~~~~~~~ Linear Algebraic Conversions ~~~~~~~~~~~~~~~ #
"""
    Lmult(q::UnitQuaternion)
    Lmult(q::StaticVector{4})
"""
function Lmult(q::UnitQuaternion)
    @SMatrix [
        q.w -q.x -q.y -q.z;
        q.x  q.w -q.z  q.y;
        q.y  q.z  q.w -q.x;
        q.z -q.y  q.x  q.w;
    ]
end
Lmult(q::SVector{4}) = Lmult(UnitQuaternion(q, false))
"""
    Rmult(q::UnitQuaternion)
    Rmult(q::StaticVector{4})

`Rmult(q1)*SVector(q2)` return a vector equivalent to `q2*q1` (quaternion composition)
"""
function Rmult(q::UnitQuaternion)
    @SMatrix [
        q.w -q.x -q.y -q.z;
        q.x  q.w  q.z -q.y;
        q.y -q.z  q.w  q.x;
        q.z  q.y -q.x  q.w;
    ]
end
Rmult(q::SVector{4}) = Rmult(UnitQuaternion(q, false))

"""
    Tmat()

`Tmat()*SVector(q)`return a vector equivalent to `inv(q)`, where `q` is a `UnitQuaternion`
"""
function Tmat()
    @SMatrix [
        1  0  0  0;
        0 -1  0  0;
        0  0 -1  0;
        0  0  0 -1;
    ]
end

"""
    Vmat()

`Vmat()*SVector(q)`` returns the imaginary
    (vector) part of the quaternion `q` (equivalent to `vector(q)``)
"""
function Vmat()
    @SMatrix [
        0 1 0 0;
        0 0 1 0;
        0 0 0 1
    ]
end

"""
    Hmat()
    Hmat(r::AbstractVector)

`Hmat()*r` or `Hmat(r)` converts `r` into a pure quaternion, where `r` is 3-dimensional.

`Hmat() == Vmat()'`
"""
function Hmat()
    @SMatrix [
        0 0 0;
        1 0 0;
        0 1 0;
        0 0 1.;
    ]
end

function Hmat(r)
    @assert length(r) == 3
    @SVector [0,r[1],r[2],r[3]]
end


# ~~~~~~~~~~~~~~~ Useful Jacobians ~~~~~~~~~~~~~~~ #
"""
    ∇differential(q::UnitQuaternion)

Jacobian of `Lmult(q) QuatMap(ϕ)`, when ϕ is near zero.

Useful for converting Jacobians from R⁴ to R³ and
    correctly account for unit norm constraint. Jacobians for different
    differential quaternion parameterization are the same up to a constant.
"""
function ∇differential(q::UnitQuaternion)
    1.0 * @SMatrix [
        -q.x -q.y -q.z;
         q.w -q.z  q.y;
         q.z  q.w -q.x;
        -q.y  q.x  q.w;
    ]
end

"""
    ∇²differential(q::UnitQuaternion, b::SVector{4})

Jacobian of `(∂/∂ϕ Lmult(q) QuatMap(ϕ))`b, evaluated at ϕ=0
"""
function ∇²differential(q::UnitQuaternion, b::AbstractVector)
    @assert length(b) == 4 "Length of `b` must be 4, got $(length(b))"
    b1 = -SVector(q)'b
    Diagonal(@SVector fill(b1,3))
end

"""
    ∇rotate(R::Rotation{3}, r::StaticVector)

Jacobian of `R*r` with respect to the rotation
"""
function ∇rotate(q::UnitQuaternion{T,D}, r::StaticVector{3}) where {T,D}
    rhat = UnitQuaternion{D}(r)
    R = Rmult(q)
    2Vmat()*Rmult(q)'Rmult(rhat)
end

"""
    ∇composition1(R2::Rotation{3}, R1::Rotation{3})

Jacobian of `R2*R1` with respect to `R1`
"""
function ∇composition1(q2::UnitQuaternion, q1::UnitQuaternion)
    Lmult(q2)
end

"""
    ∇composition2(R2::Rotation{3}, R1::Rotation{3})

Jacobian of `R2*R1` with respect to `R2`
"""
function ∇composition2(q2::UnitQuaternion, q1::UnitQuaternion)
    Rmult(q1)
end
