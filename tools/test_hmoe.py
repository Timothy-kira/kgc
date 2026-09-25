"""H-MoE gate semantics vs the DeepSeek-V4.1-Flash reference Gate (inference/model.py L809-827)."""
import torch
import torch.nn.functional as F

from model.hmoe import HGate, HeuristicMoE


def ref_gate(x, weight, bias, topk, gate_temp=1.0, route_scale=1.5, norm=True):
    # verbatim logic of DSV4.1 Gate.forward (sqrtsoftplus branch)
    scores = F.linear(x.float(), weight.float()) / gate_temp
    scores = F.softplus(scores).sqrt()
    indices = (scores + bias).topk(topk, dim=-1)[1]
    weights = scores.gather(1, indices)
    if norm and topk > 1:
        weights /= weights.sum(dim=-1, keepdim=True) + 1e-20
    weights *= route_scale
    return weights, indices


def main():
    torch.manual_seed(0)
    n, d, E = 64, 32, 20
    x = torch.randn(n, d)
    for topk in (1, 3):
        g = HGate(d, E, topk=(topk,) * 3)
        with torch.no_grad():
            g.weight.normal_()
            g.bias.normal_()
        st = torch.randint(0, 3, (n,))
        w, i = g(x, st, torch.ones(n, E, dtype=torch.bool))
        for t in range(3):
            m = st == t
            rw, ri = ref_gate(x[m], g.weight, g.bias[t], topk)
            assert torch.equal(i[m], ri), "indices differ from DSV4.1"
            assert torch.allclose(w[m], rw, atol=1e-6), "weights differ from DSV4.1"
    # bias changes selection only, never the weight of a selected expert
    g = HGate(d, E, topk=(1, 1, 1))
    with torch.no_grad():
        g.weight.normal_()
    st = torch.zeros(n, dtype=torch.long)
    w0, i0 = g(x, st, torch.ones(n, E, dtype=torch.bool))
    g.bias[0, 5] = 100.0
    w1, i1 = g(x, st, torch.ones(n, E, dtype=torch.bool))
    assert (i1 == 5).all()
    assert torch.allclose(w1[:, 0], g.scores(x)[:, 5] * 1.5)
    # unavailable experts are never selected
    avail = torch.ones(n, E, dtype=torch.bool)
    avail[:, 5] = False
    w2, i2 = g(x, st, avail)
    assert not (i2 == 5).any()
    # pointer copy with a favoured expert reproduces that expert's proposal
    moe = HeuristicMoE(d, E)
    moe.favour(3, margin=5.0)
    props = torch.randint(0, 50, (n, E))
    enc = lambda c, s: torch.randn(len(c), d) * 0.0
    y, w, cand = moe(x, st, props, enc)
    logits = (torch.rand(n, 50) * 2 - 1) * 19.0          # CLM logits are scale*cos, |.| <= scale (~19)
    out = moe.pointer_logits(logits, w, cand)
    assert torch.equal(out.argmax(-1), props[:, 3]), "safe init must reproduce the favoured skill"
    print("HMOE_TESTS_OK")


if __name__ == "__main__":
    main()
