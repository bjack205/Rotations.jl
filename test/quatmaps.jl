using ForwardDiff
import Rotations: jacobian, map_type

@testset "Quaternion Maps" begin
    ϕ = @SVector rand(3)
    v = 0.1*@SVector rand(3)
    g = @SVector rand(3)
    p = SVector(rand(MRP))

    @testset "Forward Maps" begin
        # Exponential
        @test ForwardDiff.jacobian(x->SVector(ExponentialMap()(x)),ϕ) ≈ jacobian(ExponentialMap(),ϕ)

        ϕ = 1e-6*@SVector rand(3)
        @test ForwardDiff.jacobian(x->SVector(ExponentialMap()(x)),ϕ) ≈ jacobian(ExponentialMap(),ϕ)

        # Vector Part
        @test ForwardDiff.jacobian(x->SVector(QuatVecMap()(x)),v) ≈
            jacobian(QuatVecMap(), v)

        # Gibbs Vectors
        @test ForwardDiff.jacobian(x->SVector(CayleyMap()(x)),g) ≈ jacobian(CayleyMap(), g)

        # MRPs
        @test ForwardDiff.jacobian(x->SVector(MRPMap()(x)),p) ≈
            jacobian(MRPMap(), p)

        μ0 = 1/Rotations.scaling(QuatVecMap)
        jac_eye = [@SMatrix zeros(1,3); μ0*Diagonal(@SVector ones(3))];
        @test jacobian(ExponentialMap(), p*1e-10) ≈ jac_eye
        @test jacobian(MRPMap(), p*1e-10) ≈ jac_eye
        @test jacobian(CayleyMap(), p*1e-10) ≈ jac_eye
        @test jacobian(QuatVecMap(), p*1e-10) ≈ jac_eye
    end



############################################################################################
#                                 INVERSE RETRACTION MAPS
############################################################################################
    @testset "Inverse Maps" begin

        # Exponential Map
        Random.seed!(1);
        q = rand(UnitQuaternion)
        q = UnitQuaternion(q)
        qval = SVector(q)
        @test ExponentialMap()(q) == Rotations.scaling(ExponentialMap())*logm(q)
        @test ExponentialMap()(ExponentialMap()(q)) ≈ q
        @test ExponentialMap()(ExponentialMap()(ϕ)) ≈ ϕ

        function invmap(q)
            μ = Rotations.scaling(ExponentialMap())
            v = @SVector [q[2], q[3], q[4]]
            s = q[1]
            θ = norm(v)
            M = μ*2atan(θ, s)/θ
            return M*v
        end
        @test invmap(qval) ≈ Rotations.scaling(ExponentialMap())*logm(q)

        qI = QuatVecMap()(v*1e-5)
        @test ForwardDiff.jacobian(invmap, qval) ≈ jacobian(ExponentialMap(), q)
        @test ForwardDiff.jacobian(invmap, SVector(qI)) ≈ jacobian(ExponentialMap(), qI)

        b = @SVector rand(3)
        @test ForwardDiff.jacobian(q->jacobian(ExponentialMap(),
            UnitQuaternion(q,false))'b, qval) ≈
            Rotations.∇jacobian(ExponentialMap(), q, b)

        # Vector Part
        invmap(q) = @SVector [q[2], q[3], q[4]]
        @test QuatVecMap()(q) ≈ Rotations.scaling(QuatVecMap)*invmap(qval)
        @test ForwardDiff.jacobian(invmap, qval) ≈ jacobian(QuatVecMap(), q)
        @test QuatVecMap()(QuatVecMap()(q)) ≈ q
        @test QuatVecMap()(QuatVecMap()(v)) ≈ v

        @test ForwardDiff.jacobian(q->jacobian(QuatVecMap(),
            UnitQuaternion(q,false))'b, qval) ≈
            Rotations.∇jacobian(QuatVecMap(), q, b)

        # Cayley
        invmap(q) = 1/q[1] * @SVector [q[2], q[3], q[4]]
        @test CayleyMap()(q) ≈ Rotations.scaling(CayleyMap)*invmap(qval)
        @test ForwardDiff.jacobian(invmap, qval) ≈ jacobian(CayleyMap(), q)
        @test CayleyMap()(CayleyMap()(q)) ≈ q
        @test CayleyMap()(CayleyMap()(g)) ≈ g

        @test ForwardDiff.jacobian(q->jacobian(CayleyMap(),
            UnitQuaternion(q,false))'b, qval) ≈
            Rotations.∇jacobian(CayleyMap(), q, b)

        # MRP
        invmap(q) = Rotations.scaling(MRPMap)/(1+q[1]) * @SVector [q[2], q[3], q[4]]
        @test MRPMap()(q) ≈ invmap(qval)
        @test ForwardDiff.jacobian(invmap, qval) ≈ jacobian(MRPMap(), q)
        @test MRPMap()(MRPMap()(q)) ≈ q
        @test MRPMap()(MRPMap()(p)) ≈ p

        @test ForwardDiff.jacobian(q->jacobian(MRPMap(),
            UnitQuaternion(q,false))'b, qval) ≈
            Rotations.∇jacobian(MRPMap(), q, b)

        # Test near origin
        μ0 = Rotations.scaling(CayleyMap)
        jacT_eye = [@SMatrix zeros(1,3); μ0*Diagonal(@SVector ones(3))]';
        @test isapprox(jacobian(ExponentialMap(),qI), jacT_eye, atol=1e-5)
        @test isapprox(jacobian(QuatVecMap(),qI), jacT_eye, atol=1e-5)
        @test isapprox(jacobian(CayleyMap(),qI), jacT_eye, atol=1e-5)
        @test isapprox(jacobian(MRPMap(),qI), jacT_eye, atol=1e-5)

    end
end
