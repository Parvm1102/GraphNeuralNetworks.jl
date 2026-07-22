module GNNlibMooncakeExt

using GNNlib: GNNlib, propagate, copy_xj, e_mul_xj, w_mul_xj
using GNNGraphs: GNNGraph, adjacency_matrix, degree, edge_index, set_edge_weight
using LinearAlgebra: adjoint
using Statistics: mean
using Base: IEEEFloat
import Mooncake
using Mooncake: CoDual, DefaultCtx, NoRData, @is_primitive

# Reverse rule for the fast path `propagate(copy_xj, g, +, xj) == xj * A`.
# `A` is constant w.r.t. the inputs, so `dxj = dy * A'`.

@is_primitive DefaultCtx Tuple{
    typeof(propagate),
    typeof(copy_xj),
    GNNGraph,
    typeof(+),
    Nothing,
    AbstractMatrix{P},
    Nothing,
} where {P <: IEEEFloat}

function Mooncake.rrule!!(
    ::CoDual{typeof(propagate)},
    ::CoDual{typeof(copy_xj)},
    g::CoDual{<:GNNGraph},
    ::CoDual{typeof(+)},
    ::CoDual{Nothing},
    xj::CoDual{<:AbstractMatrix{P}},
    ::CoDual{Nothing},
) where {P <: IEEEFloat}
    pg = Mooncake.primal(g)
    pxj = Mooncake.primal(xj)
    A = adjacency_matrix(pg, P; weighted = false)
    y = pxj * A
    res = Mooncake.zero_fcodual(y)
    function propagate_copy_xj_add_pullback!!(::NoRData)
        dy = Mooncake.tangent(res)
        dxj = Mooncake.tangent(xj)
        dxj .+= dy * adjoint(A)
        return NoRData(), NoRData(), Mooncake.zero_rdata(pg),
               NoRData(), NoRData(), NoRData(), NoRData()
    end
    return res, propagate_copy_xj_add_pullback!!
end

# Reverse rule for the weighted fast path `propagate(e_mul_xj, g, +, xj, e) == xj * A(e)`.
# `e` enters `A`, so we also return `de_k = Σ_f xj[f, s_k] * dy[f, t_k]`.

@is_primitive DefaultCtx Tuple{
    typeof(propagate),
    typeof(e_mul_xj),
    GNNGraph,
    typeof(+),
    Nothing,
    AbstractMatrix{P},
    AbstractVector{P},
} where {P <: IEEEFloat}

function Mooncake.rrule!!(
    ::CoDual{typeof(propagate)},
    ::CoDual{typeof(e_mul_xj)},
    g::CoDual{<:GNNGraph},
    ::CoDual{typeof(+)},
    ::CoDual{Nothing},
    xj::CoDual{<:AbstractMatrix{P}},
    e::CoDual{<:AbstractVector{P}},
) where {P <: IEEEFloat}
    pg = Mooncake.primal(g)
    pxj = Mooncake.primal(xj)
    pe = Mooncake.primal(e)
    s, t = edge_index(pg)
    A = adjacency_matrix(set_edge_weight(pg, pe), P; weighted = true)
    y = pxj * A
    res = Mooncake.zero_fcodual(y)
    function propagate_e_mul_xj_add_pullback!!(::NoRData)
        dy = Mooncake.tangent(res)
        dxj = Mooncake.tangent(xj)
        de = Mooncake.tangent(e)
        dxj .+= dy * adjoint(A)
        de .+= vec(sum(view(pxj, :, s) .* view(dy, :, t); dims = 1))
        return NoRData(), NoRData(), Mooncake.zero_rdata(pg),
               NoRData(), NoRData(), NoRData(), NoRData()
    end
    return res, propagate_e_mul_xj_add_pullback!!
end

# Reverse rule for the weighted fast path `propagate(w_mul_xj, g, +, xj) == xj * A(w)`.
# `w` are the graph's own weights, so `dw` accumulates into the COO tangent `g.graph[3]`.

@is_primitive DefaultCtx Tuple{
    typeof(propagate),
    typeof(w_mul_xj),
    GNNGraph{<:Tuple},
    typeof(+),
    Nothing,
    AbstractMatrix{P},
    Nothing,
} where {P <: IEEEFloat}

# Edge-weight tangent accumulator (COO `graph[3]`), or `nothing` for a constant graph.
_w_mul_xj_dw(::Mooncake.NoFData) = nothing
function _w_mul_xj_dw(dg::Mooncake.FData)
    gt = dg.data.graph
    gt isa Tuple || return nothing
    dw = gt[3]
    return dw isa AbstractVector{<:IEEEFloat} ? dw : nothing
end

function Mooncake.rrule!!(
    ::CoDual{typeof(propagate)},
    ::CoDual{typeof(w_mul_xj)},
    g::CoDual{<:GNNGraph{<:Tuple}},
    ::CoDual{typeof(+)},
    ::CoDual{Nothing},
    xj::CoDual{<:AbstractMatrix{P}},
    ::CoDual{Nothing},
) where {P <: IEEEFloat}
    pg = Mooncake.primal(g)
    pxj = Mooncake.primal(xj)
    s, t = edge_index(pg)
    A = adjacency_matrix(pg, P; weighted = true)
    y = pxj * A
    res = Mooncake.zero_fcodual(y)
    dw = _w_mul_xj_dw(Mooncake.tangent(g))
    function propagate_w_mul_xj_add_pullback!!(::NoRData)
        dy = Mooncake.tangent(res)
        dxj = Mooncake.tangent(xj)
        dxj .+= dy * adjoint(A)
        if dw !== nothing
            dw .+= vec(sum(view(pxj, :, s) .* view(dy, :, t); dims = 1))
        end
        return NoRData(), NoRData(), Mooncake.zero_rdata(pg),
               NoRData(), NoRData(), NoRData(), NoRData()
    end
    return res, propagate_w_mul_xj_add_pullback!!
end

# The `mean` fast paths divide each node's aggregated messages by its unweighted
# in-degree. Isolated nodes get 0, matching `NNlib.scatter(mean, ...)`.
function inv_in_degree(g::GNNGraph, ::Type{P}) where {P}
    d = degree(g, P; dir = :in, edge_weight = false)
    return map(x -> ifelse(iszero(x), zero(P), inv(x)), d)
end

# Reverse rule for the fast path `propagate(copy_xj, g, mean, xj) == (xj * A) .* dinv'`.
# `A` and `dinv` are constant w.r.t. the inputs, so `dxj = (dy .* dinv') * A'`.

@is_primitive DefaultCtx Tuple{
    typeof(propagate),
    typeof(copy_xj),
    GNNGraph,
    typeof(mean),
    Nothing,
    AbstractMatrix{P},
    Nothing,
} where {P <: IEEEFloat}

function Mooncake.rrule!!(
    ::CoDual{typeof(propagate)},
    ::CoDual{typeof(copy_xj)},
    g::CoDual{<:GNNGraph},
    ::CoDual{typeof(mean)},
    ::CoDual{Nothing},
    xj::CoDual{<:AbstractMatrix{P}},
    ::CoDual{Nothing},
) where {P <: IEEEFloat}
    pg = Mooncake.primal(g)
    pxj = Mooncake.primal(xj)
    A = adjacency_matrix(pg, P; weighted = false)
    dinv = inv_in_degree(pg, P)
    y = (pxj * A) .* dinv'
    res = Mooncake.zero_fcodual(y)
    function propagate_copy_xj_mean_pullback!!(::NoRData)
        dy = Mooncake.tangent(res)
        dxj = Mooncake.tangent(xj)
        dxj .+= (dy .* dinv') * adjoint(A)
        return NoRData(), NoRData(), Mooncake.zero_rdata(pg),
               NoRData(), NoRData(), NoRData(), NoRData()
    end
    return res, propagate_copy_xj_mean_pullback!!
end

# Reverse rule for the weighted fast path
# `propagate(e_mul_xj, g, mean, xj, e) == (xj * A(e)) .* dinv'`.
# With the scaled cotangent `dym = dy .* dinv'`, the gradients are the same as
# for the `+` rule: `dxj = dym * A(e)'` and `de_k = Σ_f xj[f, s_k] * dym[f, t_k]`.

@is_primitive DefaultCtx Tuple{
    typeof(propagate),
    typeof(e_mul_xj),
    GNNGraph,
    typeof(mean),
    Nothing,
    AbstractMatrix{P},
    AbstractVector{P},
} where {P <: IEEEFloat}

function Mooncake.rrule!!(
    ::CoDual{typeof(propagate)},
    ::CoDual{typeof(e_mul_xj)},
    g::CoDual{<:GNNGraph},
    ::CoDual{typeof(mean)},
    ::CoDual{Nothing},
    xj::CoDual{<:AbstractMatrix{P}},
    e::CoDual{<:AbstractVector{P}},
) where {P <: IEEEFloat}
    pg = Mooncake.primal(g)
    pxj = Mooncake.primal(xj)
    pe = Mooncake.primal(e)
    s, t = edge_index(pg)
    A = adjacency_matrix(set_edge_weight(pg, pe), P; weighted = true)
    dinv = inv_in_degree(pg, P)
    y = (pxj * A) .* dinv'
    res = Mooncake.zero_fcodual(y)
    function propagate_e_mul_xj_mean_pullback!!(::NoRData)
        dym = Mooncake.tangent(res) .* dinv'
        dxj = Mooncake.tangent(xj)
        de = Mooncake.tangent(e)
        dxj .+= dym * adjoint(A)
        de .+= vec(sum(view(pxj, :, s) .* view(dym, :, t); dims = 1))
        return NoRData(), NoRData(), Mooncake.zero_rdata(pg),
               NoRData(), NoRData(), NoRData(), NoRData()
    end
    return res, propagate_e_mul_xj_mean_pullback!!
end

end # module
