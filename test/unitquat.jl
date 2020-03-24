using Rotations
using LinearAlgebra
using StaticArrays
using Test
using ForwardDiff

import Rotations: jacobian, ∇rotate, ∇composition1, ∇composition2
import Rotations: UnitQuaternion, CayleyMap, ExponentialMap, MRPMap, IdentityMap, VectorPart,
    map_type, expm, logm, ⊖, kinematics
import Rotations: Vmat, Rmult, Lmult, Hmat, Tmat

@testset "Unit Quaternions" begin
    q1 = rand(UnitQuaternion)
    q2 = rand(UnitQuaternion)
    r = @SVector rand(3)
    ω = @SVector rand(3)

    # Constructors
    @test UnitQuaternion(1.0, 0.0, 0.0, 0.0) isa UnitQuaternion{Float64}
    @test UnitQuaternion(1.0, 0, 0, 0) isa UnitQuaternion{Float64}
    @test UnitQuaternion(1.0, 0, 0, 0) isa UnitQuaternion{Float64}
    @test UnitQuaternion(1,0,0,0) isa UnitQuaternion{Int}
    @test UnitQuaternion(1.0f0, 0,0,0) isa UnitQuaternion{Float32}

    q = normalize(@SVector rand(4))
    q32 = SVector{4,Float32}(q)
    @test UnitQuaternion(q) isa UnitQuaternion{Float64}
    @test UnitQuaternion(q32) isa UnitQuaternion{Float32}

    r = normalize(@SVector rand(3))
    r32 = SVector{3,Float32}(r)
    @test UnitQuaternion(r) isa UnitQuaternion{Float64}
    @test UnitQuaternion(r32) isa UnitQuaternion{Float32}
    @test UnitQuaternion(r).w == 0

    D = ExponentialMap
    @test UnitQuaternion{D}(1.0, 0.0, 0.0, 0.0) isa UnitQuaternion{Float64,D}
    @test UnitQuaternion{D}(1.0, 0, 0, 0) isa UnitQuaternion{Float64,D}
    @test UnitQuaternion{D}(1.0f0, 0, 0, 0) isa UnitQuaternion{Float32,D}
    @test UnitQuaternion{D}(q) isa UnitQuaternion{Float64,D}
    @test UnitQuaternion{D}(q32) isa UnitQuaternion{Float32,D}
    @test UnitQuaternion{D}(r) isa UnitQuaternion{Float64,D}
    @test UnitQuaternion{D}(r32) isa UnitQuaternion{Float32,D}

    @test UnitQuaternion{Float64}(1,0,0,0) isa UnitQuaternion{Float64}
    @test UnitQuaternion{Float32}(1,0,0,0) isa UnitQuaternion{Float32}
    @test UnitQuaternion{Float32}(1.0,0,0,0) isa UnitQuaternion{Float32}
    @test UnitQuaternion{Float64}(1f0,0,0,0) isa UnitQuaternion{Float64}

    # normalization
    @test UnitQuaternion(2.,0,0,0,true) == one(UnitQuaternion)
    @test UnitQuaternion(2q, true) ≈ UnitQuaternion(q)
    @test UnitQuaternion(2r, true) ≈ UnitQuaternion(r)

    # Copy constructors
    q = rand(UnitQuaternion)
    @test UnitQuaternion(q) === q
    @test UnitQuaternion{Float32}(q) isa UnitQuaternion{Float32,CayleyMap}
    @test UnitQuaternion{D}(q) isa UnitQuaternion{Float64,D}
    @test UnitQuaternion{Float32,D}(q) isa UnitQuaternion{Float32,D}

    # rand
    @test rand(UnitQuaternion) isa UnitQuaternion{Float64}
    @test rand(UnitQuaternion{Float32}) isa UnitQuaternion{Float32}
    @test rand(UnitQuaternion{Float32,D}) isa UnitQuaternion{Float32,D}

    # Test math
    @test UnitQuaternion(I) isa UnitQuaternion{Float64}
    @test UnitQuaternion{Float64,MRPMap}(I) isa UnitQuaternion{Float64,MRPMap}

    ϕ = ExponentialMap(q1)
    @test expm(ϕ*2) ≈ q1
    q = UnitQuaternion(ϕ, false)
    @test exp(q) ≈ q1

    q = UnitQuaternion((@SVector [1,2,3,4.]), false)
    @test 2*q == UnitQuaternion((@SVector [2,4,6,8.]), false)
    @test q*2 == UnitQuaternion((@SVector [2,4,6,8.]), false)

    # Axis-angle
    ϕ = 0.1*@SVector [1,0,0]
    q = expm(ϕ)
    @test logm(expm(ϕ)) ≈ ϕ
    @test expm(logm(q1)) ≈ q1
    @test rotation_angle(q) ≈ 0.1
    @test rotation_axis(q) == [1,0,0]

    @test norm(q1 * ExponentialMap(ϕ)) ≈ 1
    @test q1 ⊖ q2 isa SVector{3}
    @test (q1 * ExponentialMap(ϕ)) ⊖ q1 ≈ ϕ



    # Test inverses
    q3 = q2*q1
    @test q2\q3 ≈ q1
    @test q3/q1 ≈ q2
    @test inv(q1)*r ≈ q1\r
    @test r ≈ q3\(q2*q1*r)
    @test q3 ⊖ q2 ≈ CayleyMap(q1)
    q3 = UnitQuaternion{IdentityMap}(q3)
    @test q3 ⊖ q2 ≈ SVector(q3) - SVector(q2)

    q = q1
    rhat = UnitQuaternion(r,false)
    @test q*r ≈ Vmat()*Lmult(q)*Rmult(q)'*Vmat()'r
    @test q*r ≈ Vmat()*Lmult(q)*Rmult(q)'*Hmat(r)
    @test q*r ≈ Vmat()*Lmult(q)*Lmult(rhat)*Tmat()*SVector(q)
    @test q*r ≈ Vmat()*Rmult(q)'*Rmult(rhat)*SVector(q)
    @test q*r ≈ Hmat()'Rmult(q)'*Rmult(rhat)*SVector(q)

    @test Rmult(SVector(q)) == Rmult(q)
    @test Lmult(SVector(q)) == Lmult(q)
    @test Hmat(r) == SVector(UnitQuaternion(r,false))

    @test kinematics(q1,ω) isa SVector{4}

    @test ForwardDiff.jacobian(q->UnitQuaternion{VectorPart}(q,false)*r,SVector(q)) ≈ ∇rotate(q,r)

    @test ForwardDiff.jacobian(q->SVector(q2*UnitQuaternion{VectorPart}(q,false)),SVector(q1)) ≈
        ∇composition1(q2,q1)
    @test ForwardDiff.jacobian(q->SVector(UnitQuaternion{VectorPart}(q,false)*q1),SVector(q2)) ≈
        ∇composition2(q2,q1)

    b = @SVector rand(4)
    qval = SVector(q1)
    ForwardDiff.jacobian(q->∇composition1(q2,UnitQuaternion(q))'b, @SVector [1,0,0,0.])
    diffcomp =  ϕ->SVector(q2*CayleyMap(ϕ))
    ∇diffcomp(ϕ) = ForwardDiff.jacobian(diffcomp, ϕ)
    @test ∇diffcomp(@SVector zeros(3)) ≈ Rotations.∇differential(q2)
    @test ForwardDiff.jacobian(ϕ->∇diffcomp(ϕ)'b, @SVector zeros(3)) ≈
        Rotations.∇²differential(q2, b)

    @test Lmult(q) ≈ ∇composition1(q,q2)

    ϕ = @SVector zeros(3)
    @test Rotations.∇differential(q) ≈ Lmult(q)*jacobian(VectorPart,ϕ)
    @test Rotations.∇differential(q) ≈ Lmult(q)*jacobian(ExponentialMap,ϕ)
    @test Rotations.∇differential(q) ≈ Lmult(q)*jacobian(CayleyMap,ϕ)
    @test Rotations.∇differential(q) ≈ Lmult(q)*jacobian(MRPMap,ϕ)

    R1 = RotX
    R2 = UnitQuaternion
    r1 = rand(R1)
    m1 = SMatrix(r1)

    r2 = R2(r1)
    r2 ≈ m1
end
