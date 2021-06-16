import Base.@pure
using LinearAlgebra
using StaticArrays
using SpecialFunctions

const absolute_tolerance = 1e-12
const probability_tolerance = 1/128
const initially_equivariant = false
const σy = 20
const σx = 50
const σz = 5σy
const px = 0.1
const b = SA[0, 6σy, 0]
const s = SA[1, 1, 1]/sqrt(3)
const dist = 1e3
const n_trajectories = 50

@pure function ψG(t, x)
	σXt2 = σx^2 + im * t / 2
	σYt2 = σy^2 + im * t / 2
	σZt2 = σz^2 + im * t / 2
	√(σXt2 * σYt2 * σZt2) / (2*pi*σXt2*σYt2*σZt2)^(3/4) *
	exp( -1/4 * (
		(x[1] - px*t)^2 / σXt2 +
		 x[2]^2         / σYt2 +
		 x[3]^2         / σZt2
		) +
		im * (-px*px/2 + px*x[1])
	)
end

@pure function ∇ψG(t, x)
	σXt2 = σx^2 + im * t / 2
	σYt2 = σy^2 + im * t / 2
	σZt2 = σz^2 + im * t / 2
	SA[-(x[1] - px*t)/(2*σXt2) + px*im,
	   -(x[2]       )/(2*σYt2),
	   -(x[3]       )/(2*σZt2)] * ψG(t, x)
end

@pure function ψ(t, x)
	ψG(t, x - b) + ψG(t, x + b)
end

@pure function ∇ψ(t, x)
	∇ψG(t, x - b) + ∇ψG(t, x + b)
end

@pure function velocity(t, x)
	imag(∇ψ(t,x) / ψ(t,x)) + real(∇ψ(t, x) / ψ(t,x)) × s
end

@pure function vR(t, x)
	velocity(t, x) + s
end

@pure function vL(t,x)
	velocity(t, x) - s
end

@pure function rate(t, x)
	-2s⋅real(∇ψ(t,x) / ψ(t,x))
end

@pure function odestep(t, h, x, v)
	c1 = 0;
	c2 = 1/5;  a21 = 1/5;
	c3 = 3/10; a31 = 3/40;       a32 = 9/40;
	c4 = 3/5;  a41 = 3/10;       a42 = -9/10;   a43 = 6/5;
	c5 = 1;    a51 = -11/54;     a52 = 5/2;     a53 = -70/27;    a54 = 35/27;
	c6 = 7/8;  a61 = 1631/55296; a62 = 175/512; a63 = 575/13824; a64 = 44275/110592; a65 = 253/4096;
	b1 = 37/378; b2 = 0; b3 = 250/621; b4 = 125/594; b5 = 0; b6 = 512/1771;
	d1 = 2825/27648; d2 = 0; d3 = 18575/48384; d4 = 13525/55296; d5 = 277/14336; d6 = 1/4;
	hk1 = h*v(t, x)
	hk2 = h*v(t + c2*h, x + a21 * hk1)
	hk3 = h*v(t + c3*h, x + a31 * hk1 + a32 * hk2)
	hk4 = h*v(t + c4*h, x + a41 * hk1 + a42 * hk2 + a43 * hk3)
	hk5 = h*v(t + c5*h, x + a51 * hk1 + a52 * hk2 + a53 * hk3 + a54 * hk4)
	hk6 = h*v(t + c6*h, x + a61 * hk1 + a62 * hk2 + a63 * hk3 + a64 * hk4 + a65 * hk5)
	xout = x + b1 * hk1 + b2 * hk2 + b3 * hk3 + b4 * hk4 + b5 * hk5 + b6 * hk6
	xerr = (d1-b1) * hk1 + (d2-b2) * hk2 + (d3-b3) * hk3 + (d4-b4) * hk4 + (d5-b5) * hk5 + (d6-b6) * hk6

	err = norm(xerr, Inf)

	return xout, err
end

function trajectory(xi)
	h = 1.0

	steps = 1
	t = told = 0.0
	x = xold = xi
	ts = [t]
	xs = [x]
	χ = rand(Bool)

	while x[1] < dist
		if χ
			r1 = max(+rate(t,x), 0.0)
			xtry, err = odestep(t, h, x, vR)
		else
			r1 = max(-rate(t,x), 0.0)
			xtry, err = odestep(t, h, x, vL)
		end

		if err > absolute_tolerance
			h = 0.9 * h * max((absolute_tolerance / err)^(1/5), 0.1)
			continue
		end

		if χ
			r2 = max(+rate(t + h, xtry), 0.0)
		else
			r2 = max(-rate(t + h, xtry), 0.0)
		end

		r = (r1 + r2) / 2
		if r * h > probability_tolerance
			h = probability_tolerance / r
			continue
		end

		told = t
		xold = x
		t += h
		x = xtry
		steps += 1
		
		if rand() < r * h
			χ = !χ
			push!(ts, t)
			push!(xs, x)
		end

		h = 0.9 * h * min((absolute_tolerance / err)^(1/5), 10)
		if r * h > probability_tolerance
			h = probability_tolerance / r
		end
	end

	dt = (dist - xold[1]) / ((χ ? vR(told, xold) : vL(told, xold))[1])

	x, _ = odestep(told, dt,  xold, χ ? vR : vL)
	t = told + dt
	push!(ts, t)
	push!(xs, x)

	return ts, xs, steps
end

cd("/tmp/data")
foreach(rm,
	filter(
		endswith(".dat"),
		readdir()
	)
)
println(rand(["Allons-y", "Поехали", "Lad os gå"]))
trajectories = Threads.Atomic{Int}(0)
arrival_coordinates = Vector{SVector{4,Float64}}(undef, n_trajectories)

Threads.@threads for i in 1:n_trajectories

	if initially_equivariant
		δ = [σx,σy,σz].*randn(3)
	else
		# Distribute y-coordinates such that there is the same amount of
		# probability between them
		δuniform = ceil(i/2) / ((n_trajectories/2) + 1)
		δy = -√2 * σy * erfcinv(2 * δuniform)
		δ = SA[0.0, δy, 0.0]
	end
	
	if iseven(i)
		xi = SVector{3}(-b + δ)
	else
		xi = SVector{3}(+b + δ)
	end
	ts, xs, steps = trajectory(xi)
	open("traj$i.dat", "w") do io
		for i in 1:length(ts)
			println(io, ts[i], '\t', xs[i][1], '\t', xs[i][2], '\t', xs[i][3])
		end
	end
	arrival_coordinates[i] = [ts[end]; xs[end]]
	
	Threads.atomic_add!(trajectories, 1)
	print("\r$(trajectories[]) / $n_trajectories")
end
open("arrival_coordinates.dat", "w") do io
	for coord in arrival_coordinates
		println(io, coord[1], '\t', coord[2], '\t', coord[3], '\t', coord[4])
	end
end

