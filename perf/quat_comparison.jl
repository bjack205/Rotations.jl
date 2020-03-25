using BenchmarkTools
using Rotations
using StaticArrays
using Statistics
using Random
using Test

Rotations.params(q::Quat) = @SVector [q.w, q.x, q.y, q.z]
Rotations.params(spq::SPQuat) = @SVector [spq.x, spq.y, spq.z]

const comp = BenchmarkGroup()
comp["composition"] = BenchmarkGroup()
comp["rotation"] = BenchmarkGroup()
Random.seed!(1)

# ~~~~~~~~~~~~~~~~~~~~ Create two of each type of Quaternion ~~~~~~~~~~~~~~~~~~~~ #
q1 = rand(Quat)
q2 = rand(Quat)

u1 = UnitQuaternion(q1);
u2 = UnitQuaternion(q2);

# Test Equality
@test q1 == u1
@test u1 == q1

# Test composition
@test q1*q2 ≈ u1*u2
comp["composition"]["Quat"]           = @benchmarkable for i = 1:100; $q1 = $q1*$q2; end
comp["composition"]["UnitQuaternion"] = @benchmarkable for i = 1:100; $u1 = $u1*$u2; end

# Test rotation
r = @SVector rand(3)
@test q1*r ≈ u1*r
comp["rotation"]["Quat"]           = @benchmarkable for i = 1:100; $r = $q1*$r; end
comp["rotation"]["UnitQuaternion"] = @benchmarkable for i = 1:100; $r = $u1*$r; end



# ~~~~~~~~~~~~~~~~~~~~ MRP vs SPQuat ~~~~~~~~~~~~~~~~~~~~ #
p1 = rand(MRP)
p2 = rand(MRP)
s1 = SPQuat(params(p1)...)
s2 = SPQuat(params(p2)...)
@test p1 ≈ s1
@test p2 ≈ s2
@test params(p1) ≈ params(s1)
@test params(p2) ≈ params(s2)

# Test composition
@test p2*p1 ≈ s2*s1
comp["composition"]["SPQuat"] = @benchmarkable for i = 1:100; $s1 = $s1*$s2; end
comp["composition"]["MRP"]    = @benchmarkable for i = 1:100; $p1 = $p1*$p2; end

# Test rotation
r = @SVector rand(3)
@test p1*r ≈ s1*r
comp["rotation"]["SPQuat"]  = @benchmarkable for i = 1:100; $r = $s1*$r; end
comp["rotation"]["MRP"]     = @benchmarkable for i = 1:100; $r = $p1*$r; end

tune!(comp, verbose=true)
res = run(comp, verbose=true)

for (r1,r2) in ((Quat,UnitQuaternion),(SPQuat,MRP))
    println(r1, " vs ", r2)
    for key in keys(res)
        b1 = res[key][string(r1)]
        b2 = res[key][string(r2)]
        b = judge(median(b2), median(b1))
        println("  ", key, ": ", b)
    end
end
