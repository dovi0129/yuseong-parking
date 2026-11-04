"""Arrival-time-block prizes, optional visits, unique physical stops, depot return."""
from __future__ import annotations

import time
from dataclasses import replace

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from .domain import BLOCKS, Plan, PlanConfig, PlanningData, Visit, hour_block
from .network import TravelNetwork

METHOD_LABELS = {"optimized": "최적화 순회", "b1": "신고 많은 순 순회", "b2": "동별 순회", "b3": "시간대 반영 순회"}


def make_visit(data, scores, config, spot_index, arrival, vehicle):
    row = data.spots.iloc[spot_index-1]
    return Visit(str(row.spot_id), int(arrival), vehicle,
        scores.get((str(row.spot_id), hour_block(int(arrival))), 0.),
        float(row.risk_weight) if config.use_risk else 1., str(row.label), float(row.lat), float(row.lon))


def baseline(data: PlanningData, network: TravelNetwork, config: PlanConfig, method: str):
    start = time.monotonic()
    scores = data.scores(config)
    service = round(config.service_min*60)
    begin, end = config.start_min*60, config.end_min*60
    plan = Plan(method, config, score_method=data.chosen_method)
    dongs={i:str(row.dong) for i,row in enumerate(data.spots.itertuples(),start=1)}
    if method in ("b1","b3"):
        # Top train-frequency pool bounded only by the theoretical service capacity.
        capacity = min(len(data.spots), ((end-begin)//service)*config.vehicles)
        if method=="b1":
            remaining = list(range(1, capacity+1))
        else:
            blocks=[b for lo,hi,b in BLOCKS if lo*3600<end and hi*3600>begin]
            ranked=[]
            for i,row in enumerate(data.spots.itertuples(),start=1):
                prize=sum(scores.get((str(row.spot_id),b),0.) for b in blocks)*(float(row.risk_weight) if config.use_risk else 1.)
                ranked.append((-prize,str(row.spot_id),i))
            remaining=[i for _,_,i in sorted(ranked)[:capacity]]
        dong_order = []
    elif method == "b2":
        remaining = list(range(1, len(data.spots)+1))
        dong_order = data.spots.groupby("dong").n_reports_train.sum().sort_values(ascending=False).index.tolist()
        dong_rank={str(d):rank for rank,d in enumerate(dong_order)}
    else:
        raise ValueError(method)
    for vehicle in range(config.vehicles):
        current, now = 0, begin
        while remaining:
            feasible = [i for i in remaining if now+network.seconds[current, i]+service+network.seconds[i, 0] <= end]
            if not feasible:
                break
            if method == "b2":
                dong=min((dongs[i] for i in feasible),key=dong_rank.__getitem__)
                feasible = [i for i in feasible if dongs[i]==dong]
            nxt = min(feasible, key=lambda i: (network.seconds[current, i], i))
            now += int(network.seconds[current, nxt])
            plan.travel_s += int(network.seconds[current, nxt])
            plan.visits.append(make_visit(data, scores, config, nxt, now, vehicle))
            now += service
            remaining.remove(nxt)
            current = nxt
        plan.travel_s += int(network.seconds[current, 0])
        plan.vehicle_returns_s.append(now+int(network.seconds[current, 0]))
    plan.elapsed_s = time.monotonic()-start
    validate_plan(plan, network)
    return plan


def optimize(data: PlanningData, network: TravelNetwork, config: PlanConfig):
    started = time.monotonic()
    begin, end = config.start_min*60, config.end_min*60
    duration = end-begin
    service = round(config.service_min*60)
    scores = data.scores(config)
    seed=baseline(data,network,config,"b3")
    # Each physical stop has one optional copy per overlapping score block.
    # Individual disjunction penalties supply the copy-specific prize. A separate
    # ActiveVar sum <= 1 prevents counting a physical stop twice across copies/K.
    copies = [(0, begin, end, 0.)]
    groups = {}
    for i, row in enumerate(data.spots.itertuples(), start=1):
        for lo, hi, block in BLOCKS:
            left, right = max(begin, lo*3600), min(end-service, hi*3600-1)
            prize = scores.get((str(row.spot_id), block), 0.)*(float(row.risk_weight) if config.use_risk else 1.)
            if left <= right and prize > 0 and network.seconds[0, i]+service+network.seconds[i, 0] <= duration:
                copies.append((i, left, right, prize))
                groups.setdefault(i, []).append(len(copies)-1)
    manager = pywrapcp.RoutingIndexManager(len(copies), config.vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def transit(a, b):
        ca, cb = manager.IndexToNode(a), manager.IndexToNode(b)
        return int(network.seconds[copies[ca][0], copies[cb][0]])+(service if ca else 0)

    def cost(a, b):
        return int(network.seconds[copies[manager.IndexToNode(a)][0], copies[manager.IndexToNode(b)][0]])

    transit_id = routing.RegisterTransitCallback(transit)
    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(cost))
    routing.AddDimension(transit_id, duration, end, False, "Time")
    dimension = routing.GetDimensionOrDie("Time")
    for v in range(config.vehicles):
        dimension.CumulVar(routing.Start(v)).SetRange(begin, begin)
        dimension.CumulVar(routing.End(v)).SetRange(begin, end)
        routing.AddVariableMinimizedByFinalizer(dimension.CumulVar(routing.End(v)))
    # Lexicographic objective: 1e-4 prize units dominate every feasible travel
    # difference; travel seconds only break ties at that precision.
    multiplier = config.vehicles*duration+1
    for copy_index, (_, lo, hi, prize) in enumerate(copies[1:], start=1):
        index = manager.NodeToIndex(copy_index)
        dimension.CumulVar(index).SetRange(lo, hi)
        routing.AddDisjunction([index], max(1, round(prize*10_000))*multiplier)
    for group in groups.values():
        routing.solver().Add(routing.solver().Sum([routing.ActiveVar(manager.NodeToIndex(i)) for i in group]) <= 1)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromMilliseconds(max(50,int(config.solver_seconds*1000)))
    routing.CloseModelWithParameters(params)
    id_to_spot={sid:i for i,sid in enumerate(network.ids)}
    copy_lookup={(spot,hour_block(left)):i for i,(spot,left,_,_) in enumerate(copies) if i}
    seed_routes=[]
    for vehicle in range(config.vehicles):
        seed_routes.append([copy_lookup[(id_to_spot[v.spot_id],v.block)]
            for v in seed.visits if v.vehicle==vehicle and (id_to_spot[v.spot_id],v.block) in copy_lookup])
    initial=routing.ReadAssignmentFromRoutes(seed_routes,True)
    if initial is None:
        raise RuntimeError("b3 초기해를 OR-Tools에 주입하지 못했습니다.")
    params.time_limit.FromMilliseconds(max(50,int((config.solver_seconds-(time.monotonic()-started))*1000)))
    solution = routing.SolveFromAssignmentWithParameters(initial,params)
    plan = Plan("optimized", config, score_method=data.chosen_method)
    if solution is None:
        # Do not silently replace an optimizer failure with a baseline.
        plan.status = "no_solution"
        plan.vehicle_returns_s = [begin]*config.vehicles
    else:
        for vehicle in range(config.vehicles):
            index = routing.Start(vehicle)
            while not routing.IsEnd(index):
                next_index = solution.Value(routing.NextVar(index))
                a, b = manager.IndexToNode(index), manager.IndexToNode(next_index)
                plan.travel_s += int(network.seconds[copies[a][0], copies[b][0]])
                if b:
                    plan.visits.append(make_visit(data, scores, config, copies[b][0],
                        solution.Value(dimension.CumulVar(next_index)), vehicle))
                index = next_index
            plan.vehicle_returns_s.append(solution.Value(dimension.CumulVar(index)))
    plan.elapsed_s = time.monotonic()-started
    # Preserve the feasible initial incumbent if quantization/tie-breaking or
    # the time limit yields a lower *true* weighted training objective.
    if solution is None or plan.expected_weighted+1e-10<seed.expected_weighted:
        plan=replace(seed,method="optimized",elapsed_s=time.monotonic()-started,
                     incumbent_source="b3_initial_incumbent")
    else:
        plan.incumbent_source="ortools_warm_start"
    plan.seed_expected_weighted=seed.expected_weighted
    plan.seed_expected_reports=seed.expected_reports
    plan.initial_assignment_used=True
    if plan.expected_weighted+1e-10<seed.expected_weighted:
        raise AssertionError("최적화의 위험 가중 기대값이 b3 초기해보다 작습니다.")
    validate_plan(plan, network)
    return plan


def validate_plan(plan: Plan, network: TravelNetwork):
    ids = [v.spot_id for v in plan.visits]
    if len(ids) != len(set(ids)):
        raise AssertionError("동일 지점의 중복 방문/중복 보상이 있습니다.")
    begin, end = plan.config.start_min*60, plan.config.end_min*60
    service = round(plan.config.service_min*60)
    lookup = {sid: i for i, sid in enumerate(network.ids)}
    travel = 0
    for vehicle in range(plan.config.vehicles):
        previous, now = 0, begin
        for visit in [v for v in plan.visits if v.vehicle == vehicle]:
            node = lookup[visit.spot_id]
            leg = int(network.seconds[previous, node])
            if visit.arrival_s < now+leg or visit.arrival_s+service > end:
                raise AssertionError("방문·점검 시간이 근무/이동시간 제약을 위반했습니다.")
            travel += leg
            previous, now = node, visit.arrival_s+service
        travel += int(network.seconds[previous, 0])
        if plan.vehicle_returns_s[vehicle] < now+network.seconds[previous, 0] or plan.vehicle_returns_s[vehicle] > end:
            raise AssertionError("구청 복귀시간이 제약을 위반했습니다.")
    if travel != plan.travel_s:
        raise AssertionError("총 이동시간이 실제 경로와 다릅니다.")


def solve(data, network, config, method="optimized"):
    if network.ids[1:] != data.spots.spot_id.tolist():
        raise ValueError("이동행렬과 지점 순서가 다릅니다.")
    return optimize(data, network, config) if method == "optimized" else baseline(data, network, config, method)
