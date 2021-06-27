module PairsDB

using MAT

# actual cycle duration of Daniel's data
const CYCLE_DURATION = 0.24997832635983264

const db_dir = joinpath(@__DIR__, "..", "data")

export DB, get_database, get_data, get_type, get_param,
    get_uniform_param, get_ids
# ============================================================================ #
struct DB
    db::Vector{Pair{Int16, Vector{String}}}
    typ::String
end

function Base.getindex(db::DB, k::Integer)
    k < 1 || k > length(db.db) && error("Index \"$(k)\" is out of range")
    return db.db[k]
end

function Base.getindex(db::DB; id::Integer=1)
    for k in eachindex(db.db)
        if db.db[k].first == id
            return db.db[k]
        end
    end
    return nothing
end

Base.IteratorSize(db::DB) = Base.HasLength()
Base.IteratorEltype(db::DB) = Base.HasEltype()
@inline Base.length(db::DB) = length(db.db)
@inline Base.eltype(db::DB) = eltype(db.db)

Base.iterate(db::DB) = iterate(db, 1)
function Base.iterate(db::DB, state::Integer)
    state > length(db) && return nothing
    return db.db[state], state + 1
end

@inline get_type(db::DB) = db.typ
@inline get_ids(db::DB) = first.(db)
# ============================================================================ #
function get_uniform_param(pair::Pair{Int16,Vector{String}}, name::String)
    out = get_param(pair, name)
    if !all(isequal(out[1]), out[2:end])
        val = NaN
        mx = -Inf
        for x in unique(out)
            n = sum(isequal(x), out)
            if n > mx
                val = x
                mx = n
            end
        end
        k = findfirst(isequal(val), out)
        @warn("Values of \"$(name)\" are inconsistenct across files\n   values: $(out)")
    else
        k = 1
    end
    return out[k]
end
# ============================================================================ #
function get_param(pair::Pair{Int16,Vector{String}}, name::String)
    p = get_param(pair.second[1], name)
    p == nothing && error("Parameter name \"$(name)\" is not valid")
    out = Vector{typeof(p)}(undef, length(pair.second))
    out[1] = p
    for k in 2:length(pair.second)
        out[k] = get_param(pair.second[k], name)
    end
    return out
end
# ============================================================================ #
function get_param(ifile::String, name::String)
    par = matopen(ifile, "r") do io
        return read(io, "parameters")
    end
    return haskey(par, name) ? par[name] : nothing
end
# ============================================================================ #
"""
## get\\_database(typ::String, condition::Function=id->true)
    * type: one of ["contrast", "msequence", "area", "spatial_frequency"]
    * condition(id::Int16) -> (true|false): whether or not to include a pair
    based on their id
"""
function get_database(typ::String, condition::Function=x->true)

    typ = lowercase(typ)

    files = find_files(db_dir, Regex("\\d{8}_\\d{3}_" * typ * "\\-\\d{3}\\.mat"))

    isempty(files) && error("No files match pattern \"$(typ)\"")

    ids = Vector{Int16}(undef, length(files))
    for k in eachindex(files)
        m = match(r"\d{8}_(?<id>\d{3})_.*", files[k])
        m == nothing && error("Failed to extract id from \"$(files[k])\"")
        ids[k] = parse(Int16, m[:id])
    end

    db = Vector{Pair{Int16, Vector{String}}}(undef, 0)
    processed = Vector{Int16}(undef, 0)
    for id in filter(condition, sort(ids))
        k = findall(isequal(id), ids)
        if !(id in processed)
            push!(db, id => files[k])
            push!(processed, id)
        end
    end
    return DB(db, typ)
end
# ============================================================================ #
function trial_indicies(ts::Vector{Float64}, evt::Real, dur::Real)
    k1 = findfirst(x-> x >= evt, ts)
    k1 == nothing && return 1:0

    k2 = findlast(x-> x <= (evt + dur), ts[k1:end])
    k2 == nothing && return 1:0 #k1:length(ts)

    return k1:(k2 + k1 -1)
end
# ============================================================================ #
function trial_timestamps(ret::Vector{Float64}, lgn::Vector{Float64},
    evt::Vector{Float64}, dur::Real, offset::Real=2.0, start::Real=0.0, t0::Real=0.0)

    ret_out = Vector{Float64}(undef, 0)
    lgn_out = Vector{Float64}(undef, 0)
    evt_out = Vector{Float64}(undef, length(evt))

    last_end = start
    for k in eachindex(evt)
        # NOTE: in Rathbun's data the grating phase was *NOT* reset between
        # trials, so we apply another offset to each spike (drift_offset) that
        # correct each spike time so that all are relative to the same grating
        # phase (whatever phase the grating started at) before adding the trial
        # offset and run offset, if t0 is not specified no correction is made
        drift_offset = t0 > 0.0 ? mod(evt[k] .- t0, CYCLE_DURATION) : 0.0

        evt_out[k] = last_end
        ridx = trial_indicies(ret, evt[k], dur)
        lidx = trial_indicies(lgn, evt[k], dur)
        if !isempty(ridx)
            append!(ret_out, (ret[ridx] .- evt[k]) .+ (last_end + drift_offset))
        end
        if !isempty(lidx)
            append!(lgn_out, (lgn[lidx] .- evt[k]) .+ (last_end + drift_offset))
        end
        lmx = isempty(lgn_out) ? 0.0 : lgn_out[end]
        rmx = isempty(ret_out) ? 0.0 : ret_out[end]
        last_end = max(rmx, lmx) + offset
    end
    return ret_out, lgn_out, evt_out
end
# ============================================================================ #
"""
## get\\_data(db::DB, idx::Integer=0; id::Integer=-1, cond::Function=x->true)
    * db::DB - database object
    * idx::Integer [0] - index of pair to get data for (trumps id option)
### Options:
    * id::Integer [-1] - id of pair to get data for
    * ffile::Function [x->true] - filter condition for selecting files based on their parameter dict
    * ftrial::Function [x->true] - filter condition for selecting trials based on their stimulus value

"""
function get_data(db::DB, idx::Integer=0; id::Integer=-1, ffile::Function=x->true, ftrial::Function=x->true)
    offset = 2.0

    idx < 1 && id < 0 && error("You *MUST* provide either an index or a pair id!")

    if idx < 1
        files = db[id=id].second
    else
        files = db[idx].second
        id = db[idx].first
    end

    ret = Vector{Float64}(undef, 0)
    lgn = Vector{Float64}(undef, 0)
    evt = Vector{Float64}(undef, 0)
    lab = Vector{Float64}(undef, 0)

    last_end = 0.0

    for file in files
        matopen(file, "r") do mf
            par = read(mf, "parameters")

            if ffile(par)
                values = read(mf, "values")

                if isempty(values)
                    append!(ret, read(mf, "retina") .+ last_end)
                    append!(lgn , read(mf, "lgn") .+ last_end)

                    if get_type(db) == "contrast"
                        append!(lab, par["contrast"])
                    elseif get_type(db) == "msequence"
                        append!(evt, read(mf, "stimulus") .+ last_end)
                        append!(lab, par["msequence"])
                    end

                    mx = max(ret[end], lgn[end])
                    mx = isempty(evt) ? mx : max(mx, evt[end])
                    last_end =  mx + offset

                else
                    b = ftrial.(values)

                    stim = read(mf, "stimulus")
                    evt_cur = stim[b]

                    if 100 < id < 200
                        # for Rathbun data, supply trial_timestamps with the time of the
                        # very first stimulus onset so it can calculate relative
                        # stimulus phase for the entire run
                        t0 = stim[1]
                    else
                        # not needed for Tucker's data (and does not apply to Marty's)
                        t0 = 0.0
                    end

                    rtmp, ltmp, etmp = trial_timestamps(
                        read(mf, "retina"),
                        read(mf, "lgn"),
                        evt_cur,
                        par["stimulus_duration"],
                        offset,
                        last_end,
                        t0
                    )

                    append!(ret, rtmp)
                    append!(lgn, ltmp)
                    append!(evt, etmp)
                    append!(lab, values[b])

                    re = isempty(ret) ? 0 : ret[end]
                    le = isempty(lgn) ? 0 : lgn[end]
                    ee = isempty(evt) ? 0 : evt[end]
                    last_end = maximum([re, le, ee]) + offset
                end
            end
        end
    end

    return ret, lgn, evt, lab
end
# ============================================================================ #
find_files(dir::AbstractString, re=r".*") = return do_match(dir, re, isfile)
# --------------------------------------------------------------------------- #
find_directories(dir::AbstractString, re=r".*") = return do_match(dir, re, isdir)
# =========================================================================== #
function do_match(dir::AbstractString, re::Regex, f::Function)
    if !isdir(dir)
        error("Input is not a vaild directory path")
    end
    files = [joinpath(dir, x) for x in readdir(dir)]
    return filter(x->occursin(re, x) && f(x), files)
end
# =========================================================================== #
end
