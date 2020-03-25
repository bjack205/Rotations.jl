using ForwardDiff
import Rotations: ∇rotate, ∇composition1, ∇composition2, skew


@testset "$R basic tests" for R in (RodriguesParam, MRP)

    # Constructors
    @test R(1.0, 0.0, 0.0) isa R{Float64}
    @test R(1.0, 0, 0) isa R{Float64}
    @test R(1.0f0, 0f0, 0f0) isa R{Float32}
    @test R(1.0f0, 0, 0) isa R{Float32}
    @test R(1.0, 0f0, 0) isa R{Float64}
    @test R(1, 0, 0) isa R{Int}
    @test R{Float64}(1, 0, 0) isa R{Float64}
    @test R{Float64}(1f0, 0f0, 0f0) isa R{Float64}
    @test R{Float32}(1.0, 0, 0) isa R{Float32}

    # Copy constructors
    g = rand(R)
    @test R{Float32}(g) isa R{Float32}
    @test R{Float64}(rand(R{Float32})) isa R{Float64}

    # initializers
    @test rand(R) isa R{Float64}
    @test rand(R{Float32}) isa R{Float32}
    @test one(R) isa R{Float64}
    @test one(R{Float32}) isa R{Float32}
    @test SVector(one(R)) === @SVector [0,0,0.]


    # Math operations
    g = rand(R)
    @test norm(g) == sqrt(g.x^2 + g.y^2 + g.z^2)

    # Test Jacobians
    R = RodriguesParam
    g1 = rand(R)
    g2 = rand(R)
    r = @SVector rand(3)
    @test ForwardDiff.jacobian(g->R(g)*r, SVector(g1)) ≈ ∇rotate(g1, r)

    function compose(g2,g1)
        SVector(R(g2)*R(g1))
    end
    @test ForwardDiff.jacobian(g->compose(SVector(g2),g), SVector(g1)) ≈ ∇composition1(g2,g1)
    @test ForwardDiff.jacobian(g->compose(g,SVector(g1)), SVector(g2)) ≈ ∇composition2(g2,g1)

    g0 = R{Float64}(0,0,0)
    @test ∇composition1(g2, g0) ≈ Rotations.∇differential(g2)

    gval = SVector(g1)
    b = @SVector rand(3)
    @test ForwardDiff.jacobian(g->∇composition1(g2,R(g))'b, gval) ≈
        Rotations.∇²composition1(g2,g1,b)
    @test Rotations.∇²differential(g2, b) ≈
        Rotations.∇²composition1(g2, g0, b)

    # Test kinematics
    ω = @SVector rand(3)
    @test Rotations.kinematics(g1, ω) isa SVector{3}
end
