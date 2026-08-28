
import sys, types, json
import larp_agent as A

                               
class FakeBlock:
    def __init__(self, text): self.type="text"; self.text=text
class FakeMsg:
    def __init__(self, text): self.content=[FakeBlock(text)]
class FakeMessages:
    def __init__(self, outer): self.outer=outer
    def create(self, **kw):
        self.outer.calls += 1
                                                                                 
        if self.outer.calls == 1:
            return FakeMsg(json.dumps({"trios":[
                {"action":"pick","object":"meat_1","target":None},
                {"action":"pick","object":"tomato_1","target":None}]}))
        return FakeMsg(json.dumps({"trios":[
            {"action":"pick","object":"meat_1","target":None},
            {"action":"put","object":"meat_1","target":"plate_1"}]}))
class FakeClient:
    def __init__(self): self.calls=0; self.messages=FakeMessages(self)

fake = types.ModuleType("anthropic")
fake.Anthropic = lambda *a, **k: FakeClient()
sys.modules["anthropic"] = fake

scene = A.Scene(objects=["plate_1","meat_1","tomato_1","table_1"],
                on=[("plate_1","table_1"),("meat_1","table_1"),("tomato_1","table_1")])
agent = A.LARPAgent(model="claude-sonnet-4-6", max_repair_rounds=1)
res = agent.plan("put the meat on the plate", scene)
print("rounds used:", res["rounds"], "| valid:", res["valid"])
print("final ITG:", [t.tuple() for t in res["itg"].trios])
assert res["rounds"] == 1, "should have taken one repair round"
assert res["valid"], "should be valid after repair"
print("\nMOCK API TEST: PASS (broken plan -> self-repair -> valid)")
